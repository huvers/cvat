# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Bulk ingestion from S3 cloud storage using LeRobot directory layout.

Layout expected:
    s3://bucket/<procedure>/videos/chunk-XXX/observation.images.endoscope/episode_XXXXXX.mp4

Each episode becomes one CVAT task, tracked via Dataset/DatasetEpisode,
auto-classified with the procedure type, and optionally weak-labeled.
"""

import logging
import re
import shutil
from dataclasses import dataclass

from django.conf import settings
from django.db import transaction

from cvat.apps.engine.cloud_provider import db_storage_to_storage_instance
from cvat.apps.engine.models import (
    CloudStorage,
    Data,
    Dataset,
    DatasetEpisode,
    EpisodeStatus,
    Job,
    JobClassification,
    Label,
    Project,
    Segment,
    ServerFile,
    Task,
    User,
)

logger = logging.getLogger(__name__)

EPISODE_PATTERN = re.compile(r"episode_\d+\.mp4$", re.IGNORECASE)


@dataclass
class DiscoveredEpisode:
    """A video file found in S3."""
    s3_key: str
    episode_name: str
    chunk: str


def _list_files_recursive(storage, prefix: str) -> list[dict]:
    """
    Walk an S3 prefix recursively, returning all REG entries.

    ``storage.list_files`` uses a ``/`` delimiter, so it only returns one
    directory level at a time.  We recurse into every DIR entry until we
    reach leaf files.
    """
    entries = storage.list_files(prefix=prefix)
    files: list[dict] = []
    for entry in entries:
        if entry.get("type") == "DIR":
            dir_name = entry["name"].rstrip("/")
            child_prefix = f"{prefix}{dir_name}/"
            files.extend(_list_files_recursive(storage, child_prefix))
        else:
            # Reconstruct the full S3 key so callers don't need to know
            # about prefix stripping done inside list_files.
            entry_name = entry if isinstance(entry, str) else entry.get("name", "")
            full_key = f"{prefix}{entry_name}" if not entry_name.startswith(prefix) else entry_name
            files.append({"name": full_key, "type": "REG"})
    return files


def discover_episodes(cloud_storage: CloudStorage, s3_prefix: str) -> list[DiscoveredEpisode]:
    """
    Scan an S3 prefix for LeRobot-format episode videos.

    Looks for files matching:
        {s3_prefix}/videos/chunk-XXX/observation.images.endoscope/episode_XXXXXX.mp4
    """
    storage = db_storage_to_storage_instance(cloud_storage)
    prefix = f"{s3_prefix}/videos/"

    logger.info("Scanning s3://%s/%s for episodes", cloud_storage.resource, prefix)
    files = _list_files_recursive(storage, prefix)

    episodes = []
    for entry in files:
        key = entry if isinstance(entry, str) else entry.get("name", "")
        if EPISODE_PATTERN.search(key):
            parts = key.split("/")
            episode_name = parts[-1].replace(".mp4", "")
            chunk = ""
            for part in parts:
                if part.startswith("chunk-"):
                    chunk = part
                    break
            episodes.append(DiscoveredEpisode(
                s3_key=key,
                episode_name=episode_name,
                chunk=chunk,
            ))

    logger.info("Discovered %d episodes under %s", len(episodes), prefix)
    return episodes


def sync_dataset_episodes(dataset: Dataset) -> dict:
    """
    Scan S3 and update the DatasetEpisode table.
    New episodes get status='discovered'. Existing ones are unchanged.
    Returns counts of new vs existing episodes.
    """
    try:
        discovered = discover_episodes(dataset.cloud_storage, dataset.s3_prefix)
    except Exception as exc:
        logger.exception("Failed to scan S3 for dataset %d", dataset.id)
        return {"new": 0, "existing": 0, "error": str(exc)}

    existing_names = set(
        DatasetEpisode.objects.filter(dataset=dataset)
        .values_list("episode_name", flat=True)
    )

    new_count = 0
    for ep in discovered:
        if ep.episode_name not in existing_names:
            DatasetEpisode.objects.create(
                dataset=dataset,
                episode_name=ep.episode_name,
                s3_key=ep.s3_key,
                status=EpisodeStatus.DISCOVERED,
            )
            new_count += 1

    logger.info(
        "Dataset %d sync: %d new, %d existing",
        dataset.id, new_count, len(existing_names),
    )
    return {"new": new_count, "existing": len(existing_names)}


@transaction.atomic
def create_task_for_episode(
    episode: DatasetEpisode,
    *,
    cloud_storage: CloudStorage,
    project: Project | None,
    procedure_type: str,
    owner: User,
) -> Task | None:
    """Create a CVAT task for a single episode and link it."""
    if episode.task_id is not None:
        logger.info("Episode %r already has task %d, skipping", episode.episode_name, episode.task_id)
        return None

    task_name = f"{procedure_type}/{episode.episode_name}"

    # Check if task already exists by name (defensive)
    existing = Task.objects.filter(name=task_name, project=project).first()
    if existing:
        episode.task = existing
        episode.status = EpisodeStatus.INGESTED
        episode.save(update_fields=["task", "status", "updated_date"])
        return None

    # Create Data object pointing to cloud storage
    db_data = Data.objects.create(
        storage="cloud_storage",
        cloud_storage=cloud_storage,
        sorting_method="lexicographical",
    )
    db_data.make_dirs()

    ServerFile.objects.create(file=episode.s3_key, data=db_data)

    db_task = Task.objects.create(
        name=task_name,
        owner=owner,
        project=project,
        data=db_data,
        organization=cloud_storage.organization,
    )

    segment = Segment.objects.create(
        task=db_task,
        start_frame=0,
        stop_frame=0,
    )
    job = Job.objects.create(segment=segment)

    # Auto-classify with procedure type
    procedure_label = None
    if project:
        procedure_label = Label.objects.filter(project=project, name=procedure_type).first()
    if procedure_label is None:
        procedure_label = Label.objects.filter(task=db_task, name=procedure_type).first()

    if procedure_label:
        JobClassification.objects.get_or_create(
            job=job, label=procedure_label, defaults={"owner": owner},
        )

    # Link episode to task
    episode.task = db_task
    episode.status = EpisodeStatus.INGESTED
    episode.save(update_fields=["task", "status", "updated_date"])

    logger.info("Created task %r (id=%d) for episode %s", task_name, db_task.id, episode.s3_key)
    return db_task


def run_bulk_ingest(
    dataset_id: int | None = None,
    *,
    # Legacy params (used when dataset_id is None)
    cloud_storage_id: int | None = None,
    procedure_prefix: str | None = None,
    project_id: int | None = None,
    owner_id: int | None = None,
    trigger_weak_labeling: bool = False,
) -> dict:
    """
    Main entry point (called as an RQ job).

    If dataset_id is provided, uses the Dataset entity.
    Otherwise falls back to legacy params and creates a Dataset.
    """

    # Resolve or create dataset
    if dataset_id:
        try:
            dataset = Dataset.objects.get(id=dataset_id)
        except Dataset.DoesNotExist:
            logger.error("Dataset %d not found", dataset_id)
            return {"created": 0, "skipped": 0, "errors": 1, "error": "Dataset not found"}
        owner = dataset.owner
    else:
        # Legacy path: create dataset from params
        if not cloud_storage_id or not procedure_prefix or not owner_id:
            return {"created": 0, "skipped": 0, "errors": 1, "error": "Missing required params"}
        try:
            cloud_storage = CloudStorage.objects.get(id=cloud_storage_id)
            owner = User.objects.get(id=owner_id)
            project = Project.objects.get(id=project_id) if project_id else None
        except (CloudStorage.DoesNotExist, User.DoesNotExist, Project.DoesNotExist) as exc:
            logger.error("Lookup failed: %s", exc)
            return {"created": 0, "skipped": 0, "errors": 1, "error": str(exc)}

        procedure_type = procedure_prefix.rstrip("/").split("/")[-1]
        dataset, _ = Dataset.objects.get_or_create(
            cloud_storage=cloud_storage,
            s3_prefix=procedure_prefix,
            defaults={
                "name": procedure_type,
                "procedure_type": procedure_type,
                "project": project,
                "owner": owner,
            },
        )

    # Sync episodes from S3
    sync_result = sync_dataset_episodes(dataset)
    if "error" in sync_result:
        return {"created": 0, "skipped": 0, "errors": 1, "error": sync_result["error"]}

    # Ingest discovered episodes
    episodes = DatasetEpisode.objects.filter(
        dataset=dataset, status=EpisodeStatus.DISCOVERED,
    )

    created = 0
    skipped = 0
    errors = 0

    for episode in episodes:
        try:
            task = create_task_for_episode(
                episode,
                cloud_storage=dataset.cloud_storage,
                project=dataset.project,
                procedure_type=dataset.procedure_type,
                owner=owner,
            )
            if task is None:
                skipped += 1
            else:
                created += 1
        except Exception:
            logger.exception("Failed to create task for %s", episode.s3_key)
            errors += 1

    logger.info(
        "Bulk ingest complete for dataset %d: %d created, %d skipped, %d errors",
        dataset.id, created, skipped, errors,
    )

    # Trigger weak labeling for newly ingested episodes
    if trigger_weak_labeling and created > 0:
        try:
            import django_rq
            queue = django_rq.get_queue(settings.CVAT_QUEUES.AUTO_ANNOTATION.value)
            for episode in DatasetEpisode.objects.filter(dataset=dataset, status=EpisodeStatus.INGESTED):
                if episode.task_id:
                    for job in Job.objects.filter(segment__task_id=episode.task_id):
                        queue.enqueue(
                            "cvat.apps.engine.weak_labeling.run_weak_labeling",
                            job_id=job.id,
                            job_timeout=900,
                        )
        except Exception:
            logger.exception("Failed to enqueue weak labeling for dataset %d", dataset.id)

    # Enqueue video processing for newly created tasks
    if created > 0:
        try:
            processing = enqueue_dataset_processing(dataset.id)
            logger.info(
                "Enqueued %d episodes for video processing",
                processing["enqueued"],
            )
        except Exception:
            logger.exception("Failed to enqueue video processing for dataset %d", dataset.id)

    return {
        "dataset_id": dataset.id,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "sync": sync_result,
    }


def _cleanup_raw_video(db_data: Data) -> None:
    """Delete downloaded source video, keeping manifest and chunks."""
    raw_dir = db_data.get_upload_dirname()
    if not raw_dir.exists():
        return
    for f in raw_dir.iterdir():
        if f.name == "manifest.jsonl":
            continue
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            f.unlink()
            logger.info("Evicted raw file: %s (%.0f MB)", f.name, size_mb)
        elif f.is_dir():
            shutil.rmtree(f)


def process_episode_data(episode_id: int) -> dict:
    """
    Process a single episode's video through CVAT's create_thread pipeline.

    Downloads video from S3, extracts frames, builds static chunks on disk,
    then deletes the source video to reclaim space (process-and-evict).
    The video remains streamable from the pre-built chunks.
    """
    from cvat.apps.engine.task import create_thread

    try:
        episode = DatasetEpisode.objects.select_related("task__data").get(id=episode_id)
    except DatasetEpisode.DoesNotExist:
        return {"error": f"Episode {episode_id} not found"}

    if not episode.task_id:
        return {"error": "Episode has no task"}

    db_task = episode.task
    db_data = db_task.data

    # Skip if already processed (size > 0 means frames were extracted)
    if db_data.size and db_data.size > 0:
        logger.info("Episode %d already processed (task %d, %d frames)", episode_id, db_task.id, db_data.size)
        return {"status": "already_processed", "task_id": db_task.id}

    logger.info(
        "Processing episode %d: task %d, s3_key=%s",
        episode_id, db_task.id, episode.s3_key,
    )

    # Delete placeholder segments/jobs — create_thread will create real ones
    db_task.segment_set.all().delete()

    # Build the data dict that create_thread expects
    data_dict = {
        "chunk_size": None,
        "image_quality": 70,
        "start_frame": 0,
        "stop_frame": 0,
        "frame_filter": "",
        "sorting_method": "lexicographical",
        "storage_method": "file_system",
        "storage": "cloud_storage",
        "client_files": [],
        "server_files": [episode.s3_key],
        "remote_files": [],
        "server_files_exclude": [],
        "use_zip_chunks": False,
        "use_cache": False,
        "copy_data": False,
        "filename_pattern": None,
        "job_file_mapping": None,
        "validation_params": {},
    }

    try:
        create_thread(db_task.pk, data_dict)
    except Exception:
        logger.exception("create_thread failed for task %d", db_task.id)
        raise

    # Refresh to get updated size
    db_data.refresh_from_db()

    # Evict the downloaded source video — chunks on disk are sufficient
    _cleanup_raw_video(db_data)

    logger.info(
        "Episode %d processed: task %d, %d frames, chunks on disk",
        episode_id, db_task.id, db_data.size,
    )
    return {"status": "processed", "task_id": db_task.id, "frames": db_data.size}


def enqueue_dataset_processing(dataset_id: int) -> dict:
    """Enqueue video processing jobs for all ingested but unprocessed episodes."""
    import django_rq

    dataset = Dataset.objects.get(id=dataset_id)

    episodes = (
        DatasetEpisode.objects
        .filter(dataset=dataset, status=EpisodeStatus.INGESTED, task__isnull=False)
        .select_related("task__data")
    )

    queue = django_rq.get_queue(settings.CVAT_QUEUES.IMPORT_DATA.value)
    enqueued = 0
    skipped = 0

    for ep in episodes:
        if ep.task.data.size and ep.task.data.size > 0:
            skipped += 1
            continue

        queue.enqueue(
            "cvat.apps.engine.bulk_ingest.process_episode_data",
            episode_id=ep.id,
            job_timeout=3600,
        )
        enqueued += 1

    logger.info(
        "Dataset %d: enqueued %d episodes for processing, skipped %d already processed",
        dataset_id, enqueued, skipped,
    )
    return {"enqueued": enqueued, "skipped": skipped}
