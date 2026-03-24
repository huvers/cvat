# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from django.test import SimpleTestCase, override_settings

from cvat.apps.engine.llm import DEFAULT_LLM_MODEL, DEFAULT_LLM_URL, get_llm_request_settings


class LLMRequestSettingsTest(SimpleTestCase):
    @override_settings(
        COPILOT_LLM_URL="",
        COPILOT_LLM_MODEL="",
        COPILOT_LLM_API_KEY="",
    )
    def test_defaults_without_api_key(self):
        url, model, headers = get_llm_request_settings(
            url_setting="COPILOT_LLM_URL",
            model_setting="COPILOT_LLM_MODEL",
            api_key_setting="COPILOT_LLM_API_KEY",
        )

        self.assertEqual(url, DEFAULT_LLM_URL)
        self.assertEqual(model, DEFAULT_LLM_MODEL)
        self.assertEqual(headers, {})

    @override_settings(
        COPILOT_LLM_URL="https://inference-api.nvidia.com/v1/chat/completions",
        COPILOT_LLM_MODEL="azure/openai/gpt-5.4",
        COPILOT_LLM_API_KEY="nvapi-test",
    )
    def test_service_specific_settings_add_bearer_header(self):
        url, model, headers = get_llm_request_settings(
            url_setting="COPILOT_LLM_URL",
            model_setting="COPILOT_LLM_MODEL",
            api_key_setting="COPILOT_LLM_API_KEY",
        )

        self.assertEqual(url, "https://inference-api.nvidia.com/v1/chat/completions")
        self.assertEqual(model, "azure/openai/gpt-5.4")
        self.assertEqual(headers, {"Authorization": "Bearer nvapi-test"})
