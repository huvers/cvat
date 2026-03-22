# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Bulk ingestion from S3 cloud storage using LeRobot directory layout.

Layout expected:
    s3://bucket/<procedure>/videos/chunk-XXX/observation.images.endoscope/episode_XXXXXX.mp4

Each episode becomes one CVAT task, auto-classified with the procedure type,
and optionally weak-labeled if a matching temporal model exists.
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


def discover_episodes(cloud_storage: CloudStorage, procedure_prefix: str) -> list[DiscoveredEpisode]:
    """
    Scan an S3 prefix for LeRobot-format episode videos.

    Looks for files matching:
        {procedure_prefix}/videos/chunk-XXX/observation.images.endoscope/episode_XXXXXX.mp4
    """
    storage = db_storage_to_storage_instance(cloud_storage)
    prefix = f"{procedure_prefix}/videos/"

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


@transaction.atomic
def create_task_for_episode(
    episode: DiscoveredEpisode,
    *,
    cloud_storage: CloudStorage,
    project: Project | None,
    procedure_type: str,
    owner: User,
) -> Task:
    """Create a CVAT task for a single episode video."""
    task_name = f"{procedure_type}/{episode.episode_name}"

    # Check if task already exists (idempotent)
    existing = Task.objects.filter(name=task_name, project=project).first()
    if existing:
        logger.info("Task %r already exists (id=%d), skipping", task_name, existing.id)
        return None

    # Create Data object pointing to cloud storage
    db_data = Data.objects.create(
        storage="cloud_storage",
        cloud_storage=cloud_storage,
        sorting_method="lexicographical",
    )
    db_data.make_dirs()

    # Register the S3 key as a server file
    ServerFile.objects.create(file=episode.s3_key, data=db_data)

    # Create the task
    db_task = Task.objects.create(
        name=task_name,
        owner=owner,
        project=project,
        data=db_data,
        organization=cloud_storage.organization,
    )

    # Create segment + job (CVAT normally does this in background processing,
    # but we create a placeholder so the job is immediately visible)
    segment = Segment.objects.create(
        task=db_task,
        start_frame=0,
        stop_frame=0,  # Will be updated when data is processed
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
    else:
        logger.warning(
            "Label %r not found for project/task — skipping auto-classification for %s",
            procedure_type, task_name,
        )

    logger.info("Created task %r (id=%d) for episode %s", task_name, db_task.id, episode.s3_key)
    return db_task


def run_bulk_ingest(
    cloud_storage_id: int,
    procedure_prefix: str,
    project_id: int | None,
    owner_id: int,
    trigger_weak_labeling: bool = False,
) -> dict:
    """
    Main entry point (called as an RQ job).

    Discovers episodes in S3, creates tasks, auto-classifies,
    and optionally triggers weak labeling.
    """
    try:
        cloud_storage = CloudStorage.objects.get(id=cloud_storage_id)
    except CloudStorage.DoesNotExist:
        logger.error("Cloud storage %d not found", cloud_storage_id)
        return {"created": 0, "skipped": 0, "errors": 1, "error": "Cloud storage not found"}

    try:
        owner = User.objects.get(id=owner_id)
    except User.DoesNotExist:
        logger.error("User %d not found", owner_id)
        return {"created": 0, "skipped": 0, "errors": 1, "error": "User not found"}

    project = None
    if project_id:
        try:
            project = Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            logger.error("Project %d not found", project_id)
            return {"created": 0, "skipped": 0, "errors": 1, "error": "Project not found"}

    # Derive procedure type from prefix (last path component)
    procedure_type = procedure_prefix.rstrip("/").split("/")[-1]

    try:
        episodes = discover_episodes(cloud_storage, procedure_prefix)
    except Exception as exc:
        logger.exception("Failed to scan S3 for episodes")
        return {"created": 0, "skipped": 0, "errors": 1, "error": f"S3 scan failed: {exc}"}

    if not episodes:
        logger.warning("No episodes found under %s", procedure_prefix)
        return {"created": 0, "skipped": 0, "errors": 0}

    created = 0
    skipped = 0
    errors = 0

    for episode in episodes:
        try:
            task = create_task_for_episode(
                episode,
                cloud_storage=cloud_storage,
                project=project,
                procedure_type=procedure_type,
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
        "Bulk ingest complete: %d created, %d skipped, %d errors",
        created, skipped, errors,
    )

    # Optionally trigger weak labeling for all new jobs
    if trigger_weak_labeling and created > 0:
        import django_rq
        queue = django_rq.get_queue(settings.CVAT_QUEUES.AUTO_ANNOTATION.value)
        for episode in episodes:
            task_name = f"{procedure_type}/{episode.episode_name}"
            task = Task.objects.filter(name=task_name, project=project).first()
            if task:
                for job in Job.objects.filter(segment__task=task):
                    queue.enqueue(
                        "cvat.apps.engine.weak_labeling.run_weak_labeling",
                        job_id=job.id,
                        job_timeout=900,
                    )

    return {"created": created, "skipped": skipped, "errors": errors}
