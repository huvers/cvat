# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Final

from django.conf import settings

DEFAULT_LLM_URL: Final[str] = "http://localhost:8000/v1/chat/completions"
DEFAULT_LLM_MODEL: Final[str] = "meta/llama-3.1-8b-instruct"


def get_llm_request_settings(
    *, url_setting: str, model_setting: str, api_key_setting: str
) -> tuple[str, str, dict[str, str]]:
    url = getattr(settings, url_setting, "") or DEFAULT_LLM_URL
    model = getattr(settings, model_setting, "") or DEFAULT_LLM_MODEL
    api_key = getattr(settings, api_key_setting, "")

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return url, model, headers
