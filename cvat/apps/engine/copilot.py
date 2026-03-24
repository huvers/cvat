# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Surgery Copilot: LLM-powered assistant that provides context-aware
suggestions to surgeons during annotation.

The copilot gathers full job context (transcript, model predictions,
quality metrics, ontology, classifications) and asks an LLM to produce
structured, actionable suggestions.
"""

import json
import logging

import requests

from cvat.apps.engine.llm import get_llm_request_settings
from cvat.apps.engine.models import (
    Job,
    JobClassification,
    JobTranscript,
    Label,
    LabeledInterval,
)
from cvat.apps.engine.surgery_metrics import compute_job_metrics

logger = logging.getLogger(__name__)

COPILOT_SYSTEM_PROMPT = """\
You are a surgical annotation copilot embedded in a video annotation platform.
You help surgeons annotate surgical procedure videos efficiently and accurately.

Your role:
1. Identify the highest-impact actions the surgeon should take next
2. Detect quality issues (coverage gaps, overlapping phases, transcript mismatches)
3. Suggest specific intervals to create, adjust, or review
4. Help the surgeon work faster by leveraging model predictions and transcript context

Rules:
- Be concise. Surgeons have limited time.
- Every suggestion must be actionable with a single click.
- Prioritize by impact: coverage gaps > boundary errors > missing classifications > polish.
- Reference specific timestamps (mm:ss format) so the surgeon can verify visually.
- If auto-predicted intervals look correct, suggest batch-accepting them.
- If the transcript mentions a phase change, suggest creating an interval there.

Respond with a JSON object containing an array of suggestions:
{
  "suggestions": [
    {
      "type": "create_interval",
      "priority": "high",
      "message": "...",
      "action": {
        "label": "Calot Triangle Dissection",
        "start_frame": 18000,
        "end_frame": 27000
      }
    },
    {
      "type": "review",
      "priority": "medium",
      "message": "...",
      "action": {
        "seek_frame": 12000
      }
    },
    {
      "type": "accept_predictions",
      "priority": "low",
      "message": "...",
      "action": {
        "count": 5
      }
    }
  ],
  "summary": "Brief status overview for the surgeon"
}

Valid suggestion types: create_interval, adjust_interval, review, accept_predictions,
add_classification, flag_issue, seek_video, submit_ready.
"""

FPS = 60  # 50fps recorded, upsampled to 60fps


def _frame_to_time(frame: int, start_frame: int = 0) -> str:
    """Convert frame number to mm:ss display string."""
    seconds = (frame - start_frame) / FPS
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def gather_job_context(job: Job) -> dict:
    """
    Gather all available context for a job into a structured dict
    suitable for LLM consumption.
    """
    start_frame = job.segment.start_frame
    stop_frame = job.segment.stop_frame
    total_frames = max(stop_frame - start_frame + 1, 1)
    duration_str = _frame_to_time(stop_frame, start_frame)

    # Classifications
    classifications = list(
        JobClassification.objects.filter(job_id=job.id)
        .select_related("label")
        .values_list("label__name", flat=True)
    )

    # Intervals (current annotations)
    intervals = list(
        LabeledInterval.objects.filter(job=job)
        .select_related("label")
        .order_by("frame")
        .values("id", "frame", "end_frame", "label__name", "source")
    )

    formatted_intervals = [
        {
            "id": i["id"],
            "label": i["label__name"],
            "start": _frame_to_time(i["frame"], start_frame),
            "end": _frame_to_time(i["end_frame"], start_frame),
            "start_frame": i["frame"],
            "end_frame": i["end_frame"],
            "source": i["source"],
        }
        for i in intervals
    ]

    auto_intervals = [i for i in formatted_intervals if i["source"] in ("auto", "semi-auto")]
    manual_intervals = [i for i in formatted_intervals if i["source"] not in ("auto", "semi-auto")]

    # Coverage gaps
    metrics = compute_job_metrics(job)

    # Available labels (ontology)
    task = job.segment.task
    if task.project_id:
        labels = list(Label.objects.filter(project_id=task.project_id, parent__isnull=True).values_list("name", flat=True))
    else:
        labels = list(Label.objects.filter(task=task, parent__isnull=True).values_list("name", flat=True))

    # Transcripts
    transcripts = list(
        JobTranscript.objects.filter(narration__job_id=job.id)
        .values("id", "status", "corrected_transcript", "word_timestamps")
    )

    transcript_excerpts = []
    for t in transcripts:
        if t["status"] == "completed" and t["corrected_transcript"]:
            text = t["corrected_transcript"][:2000]
            words = t["word_timestamps"] or []
            # Extract key moments (first word of each ~30s segment)
            key_moments = []
            last_time = -30
            for w in words:
                if isinstance(w, dict) and w.get("start", 0) >= last_time + 30:
                    key_moments.append({
                        "time": _frame_to_time(start_frame + int(w["start"] * FPS), start_frame),
                        "frame": start_frame + int(w["start"] * FPS),
                        "text": w.get("word", ""),
                    })
                    last_time = w["start"]
            transcript_excerpts.append({
                "text": text,
                "key_moments": key_moments[:20],
            })

    return {
        "job_id": job.id,
        "procedure_type": classifications[0] if classifications else "Unknown",
        "classifications": classifications,
        "duration": duration_str,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "fps": FPS,
        "available_labels": labels,
        "manual_intervals": manual_intervals,
        "auto_intervals": auto_intervals,
        "coverage_pct": metrics["coverage"]["coverage_pct"],
        "gap_count": metrics["coverage"]["gap_count"],
        "completeness": metrics["completeness"],
        "open_issues": metrics["issues"]["open"],
        "transcript_excerpts": transcript_excerpts,
    }


def get_copilot_suggestions(job: Job) -> dict:
    """
    Main entry point: gather context, call LLM, return structured suggestions.
    """
    context = gather_job_context(job)

    user_message = json.dumps(context, indent=2)

    llm_url, llm_model, llm_headers = get_llm_request_settings(
        url_setting="COPILOT_LLM_URL",
        model_setting="COPILOT_LLM_MODEL",
        api_key_setting="COPILOT_LLM_API_KEY",
    )

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            llm_url,
            json=payload,
            headers=llm_headers or None,
            timeout=(10, 60),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        suggestions = json.loads(content)
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.exception("Copilot LLM call failed for job %d", job.id)
        suggestions = {
            "suggestions": [],
            "summary": f"Copilot unavailable: {type(exc).__name__}",
            "error": True,
        }

    # Attach context summary for the UI
    suggestions["context"] = {
        "procedure_type": context["procedure_type"],
        "coverage_pct": context["coverage_pct"],
        "gap_count": context["gap_count"],
        "interval_count": len(context["manual_intervals"]) + len(context["auto_intervals"]),
        "auto_count": len(context["auto_intervals"]),
        "completeness": context["completeness"],
        "duration": context["duration"],
    }

    return suggestions
