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

import logging
import os
import time

import requests

from cvat.apps.engine.llm import DEFAULT_LLM_MODEL, DEFAULT_LLM_URL, get_llm_request_settings
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
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, doubles each retry


# ── Retry helper ─────────────────────────────────────────────────────────────

def _retry_request(fn, *, retries=MAX_RETRIES, backoff=RETRY_BACKOFF):
    """Call fn() with exponential backoff on connection/timeout errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            wait = backoff * (2 ** attempt)
            logger.warning(
                "Request failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, retries, wait, exc,
            )
            time.sleep(wait)
        except requests.HTTPError:
            raise  # Don't retry 4xx/5xx — they're deterministic
    raise last_exc


# ── ASR stage ────────────────────────────────────────────────────────────────

def _run_asr(audio_path: str) -> dict:
    """
    Send audio to Parakeet-TDT-0.6B-v2 NeMo service.
    Returns {"text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.1}, ...]}.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Narration audio file not found: {audio_path}")

    asr_url = getattr(settings, "TRANSCRIPTION_ASR_URL", DEFAULT_ASR_URL)

    def _do_request():
        with open(audio_path, "rb") as f:
            response = requests.post(
                asr_url,
                files={"audio": f},
                timeout=(10, 600),  # (connect_timeout, read_timeout)
            )
        response.raise_for_status()
        return response.json()

    result = _retry_request(_do_request)

    # Validate response structure
    if not isinstance(result, dict):
        raise ValueError(f"ASR returned non-dict response: {type(result)}")
    if "text" not in result:
        logger.warning("ASR response missing 'text' field, using empty string")

    return result


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
    if not raw_text.strip():
        return raw_text  # Nothing to correct

    llm_url, llm_model, llm_headers = get_llm_request_settings(
        url_setting="TRANSCRIPTION_LLM_URL",
        model_setting="TRANSCRIPTION_LLM_MODEL",
        api_key_setting="TRANSCRIPTION_LLM_API_KEY",
    )

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

    def _do_request():
        response = requests.post(
            llm_url,
            json=payload,
            headers=llm_headers or None,
            timeout=(10, 120),
        )
        response.raise_for_status()
        return response.json()

    data = _retry_request(_do_request)

    # Validate response structure
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected LLM response structure: {exc}") from exc


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
