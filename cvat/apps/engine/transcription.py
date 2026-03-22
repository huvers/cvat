# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Two-stage narration-to-text pipeline:
  1. ASR via NVIDIA Parakeet-TDT-0.6B-v2 → raw transcript + word timestamps
  2. LLM post-correction with surgical context → corrected transcript

Configuration (django settings):
  TRANSCRIPTION_ASR_URL  – URL of the NeMo ASR service (e.g. http://nemo:8888/asr)
  TRANSCRIPTION_LLM_URL  – URL of the LLM correction service (e.g. http://llm:8000/v1/chat/completions)
  TRANSCRIPTION_LLM_MODEL – Model name for the LLM (e.g. "meta/llama-3.1-8b-instruct")
"""

import json
import logging

import requests
from django.conf import settings

from cvat.apps.engine.models import (
    JobClassification,
    JobNarration,
    JobTranscript,
    TranscriptStatus,
)

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_ASR_URL = "http://localhost:8888/asr"
DEFAULT_LLM_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_LLM_MODEL = "meta/llama-3.1-8b-instruct"


# ── ASR stage ────────────────────────────────────────────────────────────────

def _run_asr(audio_path: str) -> dict:
    """
    Send audio to Parakeet-TDT-0.6B-v2 NeMo service.
    Returns {"text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.1}, ...]}.
    """
    asr_url = getattr(settings, "TRANSCRIPTION_ASR_URL", DEFAULT_ASR_URL)

    with open(audio_path, "rb") as f:
        response = requests.post(
            asr_url,
            files={"audio": f},
            timeout=600,
        )
    response.raise_for_status()
    return response.json()


# ── LLM correction stage ────────────────────────────────────────────────────

LLM_SYSTEM_PROMPT = (
    "You are a medical transcription editor. You receive a raw ASR transcript "
    "of a surgeon narrating a surgical procedure. The transcript may contain "
    "misspelled medical terms, incorrect anatomical names, or garbled instrument "
    "names. Correct ONLY terminology errors — do not change sentence structure, "
    "add content, or remove filler words. Return only the corrected transcript."
)


def _build_llm_context(narration: JobNarration) -> str:
    """Build surgical context from video-level classifications."""
    classifications = JobClassification.objects.filter(
        job_id=narration.job_id,
    ).select_related("label")

    if not classifications.exists():
        return ""

    labels = [c.label.name for c in classifications]
    return f"Procedure context: {', '.join(labels)}."


def _run_llm_correction(raw_text: str, context: str) -> str:
    """
    Send raw transcript + surgical context to LLM for terminology correction.
    Uses OpenAI-compatible chat completions API.
    """
    llm_url = getattr(settings, "TRANSCRIPTION_LLM_URL", DEFAULT_LLM_URL)
    llm_model = getattr(settings, "TRANSCRIPTION_LLM_MODEL", DEFAULT_LLM_MODEL)

    user_message = raw_text
    if context:
        user_message = f"{context}\n\nTranscript:\n{raw_text}"

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    response = requests.post(llm_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Pipeline entry point ─────────────────────────────────────────────────────

def process_narration(narration_id: int) -> None:
    """
    Main pipeline: ASR → LLM correction → save transcript.
    Called as an RQ job after narration upload.
    """
    try:
        narration = JobNarration.objects.select_related("job").get(id=narration_id)
    except JobNarration.DoesNotExist:
        logger.warning("Narration %d not found, skipping transcription", narration_id)
        return

    transcript, _ = JobTranscript.objects.get_or_create(
        narration=narration,
        defaults={"status": TranscriptStatus.PENDING},
    )

    transcript.status = TranscriptStatus.PROCESSING
    transcript.save(update_fields=["status", "updated_date"])

    try:
        # Stage 1: ASR
        logger.info("Starting ASR for narration %d", narration_id)
        asr_result = _run_asr(narration.file.path)
        raw_text = asr_result.get("text", "")
        word_timestamps = asr_result.get("words", [])

        transcript.raw_transcript = raw_text
        transcript.word_timestamps = word_timestamps
        transcript.save(update_fields=["raw_transcript", "word_timestamps", "updated_date"])

        # Stage 2: LLM correction
        logger.info("Starting LLM correction for narration %d", narration_id)
        context = _build_llm_context(narration)
        corrected = _run_llm_correction(raw_text, context)

        transcript.corrected_transcript = corrected
        transcript.status = TranscriptStatus.COMPLETED
        transcript.model_info = {
            "asr_url": getattr(settings, "TRANSCRIPTION_ASR_URL", DEFAULT_ASR_URL),
            "asr_model": "nvidia/parakeet-tdt-0.6b-v2",
            "llm_url": getattr(settings, "TRANSCRIPTION_LLM_URL", DEFAULT_LLM_URL),
            "llm_model": getattr(settings, "TRANSCRIPTION_LLM_MODEL", DEFAULT_LLM_MODEL),
            "procedure_context": context,
        }
        transcript.save(update_fields=[
            "corrected_transcript", "status", "model_info", "updated_date",
        ])
        logger.info("Transcription completed for narration %d", narration_id)

    except Exception as exc:
        logger.exception("Transcription failed for narration %d", narration_id)
        transcript.status = TranscriptStatus.FAILED
        transcript.error_message = f"{type(exc).__name__}: {exc}"
        transcript.save(update_fields=["status", "error_message", "updated_date"])
