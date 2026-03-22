# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Weak labeling pipeline: run temporal AI models to pre-populate phase/task
intervals for surgeon review.

When a job is created and its procedure type matches a registered model,
the pipeline runs inference and writes predicted intervals with
``source='auto'`` so they are visually distinct from manual annotations.
"""

import logging

import requests
from django.db import transaction

from cvat.apps.engine.models import (
    Job,
    JobClassification,
    Label,
    LabeledInterval,
    SourceType,
    SurgeryModel,
)

logger = logging.getLogger(__name__)


def get_procedure_types(job: Job) -> list[str]:
    """Return the list of classification label names for a job."""
    return list(
        JobClassification.objects.filter(job_id=job.id)
        .select_related("label")
        .values_list("label__name", flat=True)
    )


def find_matching_models(procedure_types: list[str], model_type: str | None = None) -> list[SurgeryModel]:
    """Find active models that match any of the given procedure types."""
    qs = SurgeryModel.objects.filter(
        procedure_type__in=procedure_types,
        is_active=True,
    )
    if model_type:
        qs = qs.filter(model_type=model_type)
    return list(qs)


def _call_model_endpoint(model: SurgeryModel, job: Job) -> list[dict]:
    """
    Call the model inference endpoint.

    The endpoint receives job metadata and returns predicted intervals::

        POST {endpoint_url}
        {
            "job_id": 42,
            "task_id": 7,
            "start_frame": 0,
            "stop_frame": 9000,
            "video_url": "/api/jobs/42/data?type=chunk&number=0&quality=original",
            "config": {...}
        }

    Expected response::

        {
            "intervals": [
                {"label": "Preparation", "start_frame": 0, "end_frame": 300},
                {"label": "Calot Triangle Dissection", "start_frame": 301, "end_frame": 1200},
                ...
            ]
        }
    """
    payload = {
        "job_id": job.id,
        "task_id": job.segment.task_id,
        "start_frame": job.segment.start_frame,
        "stop_frame": job.segment.stop_frame,
        "config": model.config,
    }

    response = requests.post(model.endpoint_url, json=payload, timeout=(10, 600))
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(f"Model endpoint returned non-JSON response: {exc}") from exc

    intervals = data.get("intervals", [])
    if not isinstance(intervals, list):
        raise ValueError(f"Model returned non-list intervals: {type(intervals)}")

    return intervals


def _resolve_label(job: Job, label_name: str) -> Label | None:
    """Find a label by name in the job's task (or project)."""
    task = job.segment.task
    label = Label.objects.filter(task=task, name=label_name).first()
    if label is None and task.project_id:
        label = Label.objects.filter(project_id=task.project_id, name=label_name).first()
    return label


@transaction.atomic
def _write_predicted_intervals(job: Job, predictions: list[dict], model: SurgeryModel) -> int:
    """
    Write model predictions as auto-sourced intervals.
    Returns the number of intervals created.
    """
    created = 0
    for pred in predictions:
        if not isinstance(pred, dict):
            logger.warning("Skipping non-dict prediction: %r", pred)
            continue

        if "start_frame" not in pred or "end_frame" not in pred:
            logger.warning("Skipping prediction missing start_frame/end_frame: %r", pred)
            continue

        label = _resolve_label(job, pred.get("label", ""))
        if label is None:
            logger.warning(
                "Skipping prediction with unknown label %r for job %d",
                pred.get("label"), job.id,
            )
            continue

        LabeledInterval.objects.create(
            job=job,
            label=label,
            frame=pred["start_frame"],
            end_frame=pred["end_frame"],
            source=SourceType.AUTO,
            group=0,
        )
        created += 1

    logger.info(
        "Wrote %d auto intervals for job %d from model %s",
        created, job.id, model.name,
    )
    return created


def run_weak_labeling(job_id: int) -> None:
    """
    Main entry point (called as an RQ job).

    Looks up the job's procedure type classifications, finds matching
    temporal models, runs inference, and writes predicted intervals.
    """
    try:
        job = Job.objects.select_related("segment__task").get(id=job_id)
    except Job.DoesNotExist:
        logger.warning("Job %d not found, skipping weak labeling", job_id)
        return

    procedure_types = get_procedure_types(job)
    if not procedure_types:
        logger.info("Job %d has no classifications, skipping weak labeling", job_id)
        return

    # Find temporal models (phase + task classifiers)
    models = find_matching_models(procedure_types, model_type=None)
    temporal_models = [m for m in models if m.model_type in ('phase_classifier', 'task_classifier')]

    if not temporal_models:
        logger.info("No temporal models found for %s, skipping weak labeling for job %d", procedure_types, job_id)
        return

    for model in temporal_models:
        try:
            logger.info("Running %s (%s) on job %d", model.name, model.model_type, job_id)
            predictions = _call_model_endpoint(model, job)
            _write_predicted_intervals(job, predictions, model)

            # Emit inference event via surgery_events
            from cvat.apps.engine.surgery_events import emit_inference_requested
            emit_inference_requested(job, model.name, model.model_type)

        except Exception:
            logger.exception("Weak labeling failed for model %s on job %d", model.name, job_id)
