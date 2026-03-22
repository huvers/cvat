# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 segmentation pipeline: run anatomy segmentation models on selected
frames or temporal windows of a surgical video.

Supports multiple frame selection strategies:
  - "all": every Nth frame (configurable step)
  - "intervals": only frames within annotated phase intervals
  - "window": a specific temporal window {start, end, step}
  - explicit list of frame numbers

Model predictions (masks/polygons per frame) are written as LabeledShape
objects with source='auto' and the model's confidence score.
"""

import logging
from typing import Any

import requests
from django.db import transaction

from cvat.apps.engine.models import (
    Job,
    Label,
    LabeledInterval,
    LabeledShape,
    ShapeType,
    SourceType,
    SurgeryModel,
)
from cvat.apps.engine.weak_labeling import find_matching_models, get_procedure_types

logger = logging.getLogger(__name__)

DEFAULT_FRAME_STEP = 60  # every 60 frames = 1 per second at 60fps


# ── Frame selection strategies ───────────────────────────────────────────────

def _select_frames_all(job: Job, step: int = DEFAULT_FRAME_STEP) -> list[int]:
    """Sample every Nth frame across the full job."""
    start = job.segment.start_frame
    stop = job.segment.stop_frame
    return list(range(start, stop + 1, step))


def _select_frames_intervals(job: Job, step: int = DEFAULT_FRAME_STEP) -> list[int]:
    """Sample frames only within annotated phase intervals."""
    intervals = LabeledInterval.objects.filter(job=job).values_list("frame", "end_frame")
    frames = set()
    for start, end in intervals:
        frames.update(range(start, end + 1, step))
    return sorted(frames)


def _select_frames_window(
    job: Job, start: int, end: int, step: int = DEFAULT_FRAME_STEP,
) -> list[int]:
    """Sample frames within a specific temporal window."""
    job_start = job.segment.start_frame
    job_stop = job.segment.stop_frame
    start = max(start, job_start)
    end = min(end, job_stop)
    return list(range(start, end + 1, step))


def resolve_frames(job: Job, frame_spec: Any) -> list[int]:
    """
    Resolve a frame specification into a list of frame numbers.

    frame_spec can be:
      - "all" or {"mode": "all", "step": 60}
      - "intervals" or {"mode": "intervals", "step": 60}
      - {"mode": "window", "start": 100, "end": 500, "step": 10}
      - [100, 200, 300] (explicit list)
    """
    if isinstance(frame_spec, list):
        return sorted(frame_spec)

    if isinstance(frame_spec, str):
        frame_spec = {"mode": frame_spec}

    mode = frame_spec.get("mode", "all")
    step = frame_spec.get("step", DEFAULT_FRAME_STEP)

    if mode == "all":
        return _select_frames_all(job, step)
    elif mode == "intervals":
        return _select_frames_intervals(job, step)
    elif mode == "window":
        return _select_frames_window(
            job,
            start=frame_spec.get("start", job.segment.start_frame),
            end=frame_spec.get("end", job.segment.stop_frame),
            step=step,
        )
    else:
        raise ValueError(f"Unknown frame selection mode: {mode}")


# ── Model endpoint call ──────────────────────────────────────────────────────

def _call_segmentation_endpoint(
    model: SurgeryModel, job: Job, frames: list[int],
) -> list[dict]:
    """
    Call the SAM3 segmentation endpoint.

    Request::

        POST {endpoint_url}
        {
            "job_id": 42,
            "task_id": 7,
            "frames": [0, 60, 120, ...],
            "config": {...}
        }

    Expected response::

        {
            "predictions": [
                {
                    "frame": 0,
                    "masks": [
                        {
                            "label": "gallbladder",
                            "rle": [0, 5, 10, 3, ...],
                            "bbox": [left, top, right, bottom],
                            "score": 0.95
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    # Send frames in batches to avoid oversized payloads
    BATCH_SIZE = 100
    all_predictions = []

    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i + BATCH_SIZE]
        payload = {
            "job_id": job.id,
            "task_id": job.segment.task_id,
            "frames": batch,
            "config": model.config,
        }

        response = requests.post(model.endpoint_url, json=payload, timeout=(10, 600))
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(f"Segmentation endpoint returned non-JSON: {exc}") from exc

        predictions = data.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError(f"Expected list of predictions, got {type(predictions)}")

        all_predictions.extend(predictions)

    return all_predictions


# ── Write predictions ────────────────────────────────────────────────────────

def _resolve_label(job: Job, label_name: str) -> Label | None:
    """Find a label by name in the job's task or project."""
    task = job.segment.task
    label = Label.objects.filter(task=task, name=label_name).first()
    if label is None and task.project_id:
        label = Label.objects.filter(project_id=task.project_id, name=label_name).first()
    return label


@transaction.atomic
def _write_segmentation_predictions(
    job: Job, predictions: list[dict], model: SurgeryModel,
) -> int:
    """
    Write SAM3 mask predictions as LabeledShape objects.
    Returns the number of shapes created.
    """
    created = 0

    for frame_pred in predictions:
        if not isinstance(frame_pred, dict):
            logger.warning("Skipping non-dict frame prediction: %r", frame_pred)
            continue

        frame = frame_pred.get("frame")
        masks = frame_pred.get("masks", [])

        if frame is None or not isinstance(masks, list):
            logger.warning("Skipping prediction with missing frame or masks: %r", frame_pred)
            continue

        for mask in masks:
            if not isinstance(mask, dict):
                continue

            label_name = mask.get("label", "")
            label = _resolve_label(job, label_name)
            if label is None:
                logger.warning(
                    "Skipping mask with unknown label %r on frame %d",
                    label_name, frame,
                )
                continue

            rle = mask.get("rle", [])
            bbox = mask.get("bbox", [0, 0, 0, 0])
            score = mask.get("score", 1.0)

            # CVAT mask format: RLE data followed by bbox [left, top, right, bottom]
            if len(bbox) == 4:
                points = rle + bbox
            else:
                points = rle

            # Determine shape type from prediction
            shape_type = mask.get("type", "mask")
            if shape_type == "polygon":
                points = mask.get("points", [])
            elif shape_type == "rectangle":
                points = mask.get("points", bbox)

            LabeledShape.objects.create(
                job=job,
                label=label,
                frame=frame,
                type=shape_type,
                points=points,
                source=SourceType.AUTO,
                score=score,
                occluded=False,
                outside=False,
                z_order=0,
                rotation=0,
            )
            created += 1

    logger.info(
        "Wrote %d auto shapes for job %d from model %s",
        created, job.id, model.name,
    )
    return created


# ── Pipeline entry points ────────────────────────────────────────────────────

def run_segmentation(
    job_id: int,
    frame_spec: Any = "all",
    model_name: str | None = None,
) -> dict:
    """
    Main entry point (called as an RQ job).

    Runs SAM3 segmentation on the specified frames and writes mask predictions.
    If model_name is not specified, finds matching models by procedure type.
    """
    try:
        job = Job.objects.select_related("segment__task").get(id=job_id)
    except Job.DoesNotExist:
        logger.warning("Job %d not found, skipping segmentation", job_id)
        return {"error": "Job not found"}

    # Resolve frames
    try:
        frames = resolve_frames(job, frame_spec)
    except (ValueError, TypeError) as exc:
        logger.error("Invalid frame spec for job %d: %s", job_id, exc)
        return {"error": f"Invalid frame spec: {exc}"}

    if not frames:
        logger.info("No frames to process for job %d", job_id)
        return {"frames": 0, "shapes": 0}

    # Find segmentation model
    if model_name:
        models = list(SurgeryModel.objects.filter(name=model_name, is_active=True))
    else:
        procedure_types = get_procedure_types(job)
        models = find_matching_models(procedure_types, model_type="anatomy_segmenter")

    if not models:
        logger.info("No segmentation models found for job %d", job_id)
        return {"frames": len(frames), "shapes": 0, "error": "No matching models"}

    total_shapes = 0
    for model in models:
        try:
            logger.info(
                "Running %s on job %d (%d frames)",
                model.name, job_id, len(frames),
            )
            predictions = _call_segmentation_endpoint(model, job, frames)
            shapes = _write_segmentation_predictions(job, predictions, model)
            total_shapes += shapes
        except Exception:
            logger.exception("Segmentation failed for model %s on job %d", model.name, job_id)

    return {
        "frames": len(frames),
        "shapes": total_shapes,
        "models": [m.name for m in models],
    }
