# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Dataset-level export in two formats:

1. **COCO** — for SAM3 segmentation fine-tuning
   Standard COCO format with RLE masks, categories from ontology labels,
   and image references per annotated frame.

2. **Temporal** — for phase/task classification model training
   JSON with intervals, classifications, and transcripts per episode.

Both produce one coherent dataset file across all episodes, not per-job.
"""

import json
import logging
from datetime import datetime

from django.db.models import Q

from django.contrib.auth.models import User

from cvat.apps.engine.models import (
    Dataset,
    DatasetEpisode,
    Job,
    JobClassification,
    JobTranscript,
    Label,
    LabeledInterval,
    LabeledShape,
    ShapeType,
)

logger = logging.getLogger(__name__)

SURGERY_FPS = 60


def _get_annotator_metadata(job: Job) -> dict:
    """Get the annotator's expertise info for a job."""
    assignee = job.assignee
    if not assignee:
        return {"username": None, "role": None, "expertise_level": None}
    profile = getattr(assignee, "profile", None)
    return {
        "username": assignee.username,
        "role": getattr(profile, "role", "") if profile else "",
        "expertise_level": getattr(profile, "expertise_level", "") if profile else "",
        "specialty": getattr(profile, "specialty", "") if profile else "",
        "institution": getattr(profile, "institution", "") if profile else "",
    }


def _frame_to_seconds(frame: int, start_frame: int = 0) -> float:
    return round((frame - start_frame) / SURGERY_FPS, 3)


# ═══════════════════════════════════════════════════════════════════════
# COCO FORMAT — for SAM3 segmentation fine-tuning
# ═══════════════════════════════════════════════════════════════════════

def _build_coco_categories(dataset: Dataset) -> tuple[list[dict], dict[int, int]]:
    """
    Build COCO categories from project labels.
    Returns (categories_list, label_id_to_category_id_map).
    """
    if dataset.project_id:
        labels = Label.objects.filter(
            project_id=dataset.project_id, parent__isnull=True,
        ).order_by("id")
    else:
        labels = Label.objects.none()

    categories = []
    label_to_cat = {}
    for idx, label in enumerate(labels, start=1):
        categories.append({
            "id": idx,
            "name": label.name,
            "supercategory": dataset.procedure_type,
        })
        label_to_cat[label.id] = idx

    return categories, label_to_cat


def export_coco(dataset: Dataset) -> dict:
    """
    Export all segmentation annotations in COCO format.

    Produces a standard COCO JSON with:
      - images: one entry per (episode, frame) that has shape annotations
      - annotations: one entry per mask/polygon/rectangle shape
      - categories: from project ontology labels
    """
    categories, label_to_cat = _build_coco_categories(dataset)

    episodes = DatasetEpisode.objects.filter(
        dataset=dataset,
        task__isnull=False,
    ).select_related("task")

    images = []
    annotations = []
    image_id = 0
    ann_id = 0

    # Track (task_id, frame) → image_id to avoid duplicate image entries
    frame_image_map: dict[tuple[int, int], int] = {}

    for episode in episodes:
        task = episode.task
        if not task:
            continue

        jobs = Job.objects.filter(segment__task=task).select_related("segment")

        for job in jobs:
            start_frame = job.segment.start_frame
            stop_frame = job.segment.stop_frame

            shapes = LabeledShape.objects.filter(
                job=job,
            ).order_by("frame")

            for shape in shapes:
                if shape.label_id not in label_to_cat:
                    continue

                # Create image entry if not already present for this frame
                frame_key = (task.id, shape.frame)
                if frame_key not in frame_image_map:
                    image_id += 1
                    frame_image_map[frame_key] = image_id
                    images.append({
                        "id": image_id,
                        "file_name": f"{episode.episode_name}/frame_{shape.frame:06d}.jpg",
                        "width": 0,  # Populated by training pipeline from actual frame
                        "height": 0,
                        "episode": episode.episode_name,
                        "frame": shape.frame,
                        "time_seconds": _frame_to_seconds(shape.frame, start_frame),
                        "task_id": task.id,
                        "job_id": job.id,
                    })

                img_id = frame_image_map[frame_key]
                ann_id += 1

                annotator = _get_annotator_metadata(job)

                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label_to_cat[shape.label_id],
                    "iscrowd": 0,
                    "score": shape.score,
                    "source": shape.source,
                    "annotator": annotator,
                }

                if shape.type == ShapeType.MASK:
                    # CVAT mask points: [...rle_data, left, top, right, bottom]
                    points = list(shape.points)
                    if len(points) >= 5:
                        bbox = points[-4:]  # [left, top, right, bottom]
                        rle = points[:-4]
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        annotation["segmentation"] = {
                            "counts": rle,
                            "size": [int(h), int(w)],
                        }
                        annotation["bbox"] = [bbox[0], bbox[1], w, h]
                        annotation["area"] = w * h
                    else:
                        annotation["segmentation"] = {"counts": points, "size": [0, 0]}
                        annotation["bbox"] = [0, 0, 0, 0]
                        annotation["area"] = 0

                elif shape.type == ShapeType.POLYGON:
                    points = list(shape.points)
                    annotation["segmentation"] = [points]
                    xs = points[0::2]
                    ys = points[1::2]
                    if xs and ys:
                        x0, y0 = min(xs), min(ys)
                        w, h = max(xs) - x0, max(ys) - y0
                        annotation["bbox"] = [x0, y0, w, h]
                        annotation["area"] = w * h
                    else:
                        annotation["bbox"] = [0, 0, 0, 0]
                        annotation["area"] = 0

                elif shape.type == ShapeType.RECTANGLE:
                    points = list(shape.points)
                    if len(points) == 4:
                        x0, y0, x1, y1 = points
                        w, h = x1 - x0, y1 - y0
                        annotation["bbox"] = [x0, y0, w, h]
                        annotation["area"] = w * h
                        annotation["segmentation"] = [[x0, y0, x1, y0, x1, y1, x0, y1]]
                    else:
                        annotation["bbox"] = [0, 0, 0, 0]
                        annotation["area"] = 0
                        annotation["segmentation"] = []

                else:
                    continue  # Skip unsupported shape types

                annotations.append(annotation)

    return {
        "info": {
            "description": f"{dataset.name} - {dataset.procedure_type}",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "CVAT Surgery Platform",
            "date_created": datetime.now().isoformat(),
            "dataset_id": dataset.id,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL FORMAT — for phase/task classification model training
# ═══════════════════════════════════════════════════════════════════════

def export_temporal(dataset: Dataset) -> dict:
    """
    Export all temporal annotations (intervals, classifications, transcripts)
    in a training-friendly JSON format.

    Produces one entry per episode with:
      - intervals (phases/tasks) with labels and time ranges
      - classifications (procedure metadata)
      - transcripts (corrected text + word timestamps)
    """
    episodes_data = []

    episodes = DatasetEpisode.objects.filter(
        dataset=dataset,
        task__isnull=False,
    ).select_related("task")

    for episode in episodes:
        task = episode.task
        if not task:
            continue

        jobs = Job.objects.filter(segment__task=task).select_related("segment")

        episode_entry = {
            "episode_name": episode.episode_name,
            "s3_key": episode.s3_key,
            "status": episode.status,
            "task_id": task.id,
            "annotators": [],
            "intervals": [],
            "classifications": [],
            "transcripts": [],
        }

        for job in jobs:
            start_frame = job.segment.start_frame
            annotator = _get_annotator_metadata(job)
            if annotator["username"] and annotator not in episode_entry["annotators"]:
                episode_entry["annotators"].append(annotator)
            stop_frame = job.segment.stop_frame

            # Intervals
            intervals = LabeledInterval.objects.filter(
                job=job,
            ).select_related("label").order_by("frame")

            for interval in intervals:
                episode_entry["intervals"].append({
                    "label": interval.label.name,
                    "start_frame": interval.frame,
                    "end_frame": interval.end_frame,
                    "start_seconds": _frame_to_seconds(interval.frame, start_frame),
                    "end_seconds": _frame_to_seconds(interval.end_frame, start_frame),
                    "source": interval.source,
                    "annotator": annotator["username"],
                    "annotator_expertise": annotator["expertise_level"],
                })

            # Classifications
            classifications = JobClassification.objects.filter(
                job=job,
            ).select_related("label")

            for cls in classifications:
                episode_entry["classifications"].append(cls.label.name)

            # Transcripts
            transcripts = JobTranscript.objects.filter(
                narration__job=job,
                status="completed",
            )

            for transcript in transcripts:
                episode_entry["transcripts"].append({
                    "corrected_transcript": transcript.corrected_transcript,
                    "raw_transcript": transcript.raw_transcript,
                    "word_timestamps": transcript.word_timestamps,
                })

        episodes_data.append(episode_entry)

    return {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "procedure_type": dataset.procedure_type,
            "episode_count": len(episodes_data),
            "exported_at": datetime.now().isoformat(),
        },
        "episodes": episodes_data,
    }


# ═══════════════════════════════════════════════════════════════════════
# Combined export
# ═══════════════════════════════════════════════════════════════════════

def export_dataset(dataset_id: int, fmt: str = "both") -> dict:
    """
    Export a full dataset in the specified format.

    Args:
        dataset_id: Dataset ID
        fmt: "coco", "temporal", or "both"

    Returns dict with the requested format(s).
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id)
    except Dataset.DoesNotExist:
        return {"error": f"Dataset {dataset_id} not found"}

    result = {"dataset_id": dataset_id, "name": dataset.name}

    if fmt in ("coco", "both"):
        logger.info("Exporting COCO format for dataset %d", dataset_id)
        result["coco"] = export_coco(dataset)
        logger.info(
            "COCO export: %d images, %d annotations, %d categories",
            len(result["coco"]["images"]),
            len(result["coco"]["annotations"]),
            len(result["coco"]["categories"]),
        )

    if fmt in ("temporal", "both"):
        logger.info("Exporting temporal format for dataset %d", dataset_id)
        result["temporal"] = export_temporal(dataset)
        logger.info(
            "Temporal export: %d episodes",
            len(result["temporal"]["episodes"]),
        )

    return result
