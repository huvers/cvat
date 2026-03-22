# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Surgery-specific webhook event helpers.

These emit domain-level events beyond the generic CRUD webhooks,
carrying rich surgical context for downstream automation.
"""

import logging

from django.db import transaction

from cvat.apps.engine.models import (
    Job,
    JobClassification,
    JobTranscript,
    LabeledInterval,
)
from cvat.apps.webhooks.event_type import event_name
from cvat.apps.webhooks.signals import batch_add_to_queue, get_sender, select_webhooks

logger = logging.getLogger(__name__)


def emit_procedure_submitted(job: Job) -> None:
    """
    Emit a ``submitted:procedure`` webhook with rich surgery context.

    Called after a surgeon finishes a job (state → completed).
    The payload includes classifications, interval summary, and transcript status
    so downstream consumers can trigger exports, QA sampling, or model retraining.
    """
    event = event_name("submitted", "procedure")
    webhooks = select_webhooks(job, event)
    if not webhooks:
        return

    classifications = list(
        JobClassification.objects.filter(job_id=job.id)
        .select_related("label")
        .values_list("label__name", flat=True)
    )

    interval_count = LabeledInterval.objects.filter(job_id=job.id).count()

    transcripts = list(
        JobTranscript.objects.filter(narration__job_id=job.id)
        .values("id", "status", "narration_id")
    )

    payload = {
        "event": event,
        "procedure": {
            "job_id": job.id,
            "task_id": job.segment.task_id,
            "stage": job.stage,
            "state": job.state,
            "classifications": classifications,
            "interval_count": interval_count,
            "transcripts": [
                {
                    "id": t["id"],
                    "narration_id": t["narration_id"],
                    "status": t["status"],
                }
                for t in transcripts
            ],
        },
        "sender": get_sender(job),
    }

    transaction.on_commit(
        lambda: batch_add_to_queue(webhooks, payload),
        robust=True,
    )


def emit_inference_requested(job: Job, model_name: str, model_type: str) -> None:
    """
    Emit an event signaling that model inference should run on this job.

    Used by the weak labeling pipeline: when a job is created and a matching
    temporal model exists for the procedure type, this event notifies the
    inference service.
    """
    # This uses the existing update:job webhook but with a custom payload
    # Downstream consumers filter on the "inference_requested" field
    event = event_name("update", "job")
    webhooks = select_webhooks(job, event)
    if not webhooks:
        return

    payload = {
        "event": event,
        "job": {
            "id": job.id,
            "task_id": job.segment.task_id,
        },
        "inference_requested": {
            "model_name": model_name,
            "model_type": model_type,
        },
        "sender": get_sender(job),
    }

    transaction.on_commit(
        lambda: batch_add_to_queue(webhooks, payload),
        robust=True,
    )
