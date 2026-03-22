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


def discover_episodes(cloud_storage: CloudStorage, s3_prefix: str) -> list[DiscoveredEpisode]:
    """
    Scan an S3 prefix for LeRobot-format episode videos.

    Looks for files matching:
        {s3_prefix}/videos/chunk-XXX/observation.images.endoscope/episode_XXXXXX.mp4
    """
    storage = db_storage_to_storage_instance(cloud_storage)
    prefix = f"{s3_prefix}/videos/"

    logger.info("Scanning s3://%s/%s for episodes", cloud_storage.resource, prefix)
    files = storage.list_files(prefix=prefix)

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

    return {
        "dataset_id": dataset.id,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "sync": sync_result,
    }
