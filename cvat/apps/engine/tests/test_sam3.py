# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase
from PIL import Image
from rest_framework.exceptions import ValidationError

from cvat.apps.engine.models import Label, Project, SurgeryModel, SurgeryModelType
from cvat.apps.engine.sam3 import (
    SAM3_ATLAS_CONFIDENCE_THRESHOLD,
    SAM3_ATLAS_SCORE_THRESHOLD,
    SAM3_ATLAS_VARIANT_KEY,
    SAM3_DEFAULT_CONFIDENCE_THRESHOLD,
    SAM3_DEFAULT_DEVICE,
    SAM3_DEFAULT_PRIORITY,
    SAM3_DEFAULT_SCORE_THRESHOLD,
    create_job_sam3_session,
    create_job_sam3_video_session,
    get_job_sam3_models,
    infer_job_sam3_points,
    mask_png_b64_to_cvat_rle,
    normalize_sidecar_base_url,
    parse_sam3_model_config,
    sync_job_sam3_labels,
)


def _encode_mask(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class SAM3HelpersTest(SimpleTestCase):
    def test_normalize_sidecar_base_url_strips_predict_suffix(self):
        self.assertEqual(
            normalize_sidecar_base_url("http://sam3:8090/predict"),
            "http://sam3:8090",
        )
        self.assertEqual(
            normalize_sidecar_base_url("http://sam3:8090/sam3/predict"),
            "http://sam3:8090/sam3",
        )

    def test_mask_png_b64_to_cvat_rle_converts_full_frame_mask_to_tight_rle(self):
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:3, 3:5] = 1

        result = mask_png_b64_to_cvat_rle(_encode_mask(mask))

        self.assertEqual(result, [0, 4, 3, 1, 4, 2])

    def test_mask_png_b64_to_cvat_rle_returns_none_for_empty_mask(self):
        mask = np.zeros((4, 4), dtype=np.uint8)

        self.assertIsNone(mask_png_b64_to_cvat_rle(_encode_mask(mask)))
        self.assertIsNone(mask_png_b64_to_cvat_rle(None))

    def test_parse_sam3_model_config_uses_defaults(self):
        model = SurgeryModel(
            name="sam3-default",
            procedure_type="cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={"checkpoint_path": "/models/checkpoint.pt"},
        )

        config = parse_sam3_model_config(model)

        self.assertEqual(config.display_name, "sam3-default")
        self.assertEqual(config.variant_key, None)
        self.assertEqual(config.priority, SAM3_DEFAULT_PRIORITY)
        self.assertFalse(config.is_default)
        self.assertEqual(config.checkpoint_path, "/models/checkpoint.pt")
        self.assertEqual(config.device, SAM3_DEFAULT_DEVICE)
        self.assertEqual(config.confidence_threshold, SAM3_DEFAULT_CONFIDENCE_THRESHOLD)
        self.assertEqual(config.score_threshold, SAM3_DEFAULT_SCORE_THRESHOLD)
        self.assertEqual(config.default_prompt_key, None)
        self.assertEqual(config.prompt_options, ())

    def test_parse_sam3_model_config_requires_checkpoint_path(self):
        model = SurgeryModel(
            name="sam3-invalid",
            procedure_type="cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={},
        )

        with self.assertRaises(ValidationError):
            parse_sam3_model_config(model)

    def test_parse_sam3_model_config_parses_prompt_options(self):
        model = SurgeryModel(
            name="sam3-prompts",
            procedure_type="cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/checkpoint.pt",
                "default_prompt_key": "gallbladder",
                "prompt_options": [
                    {
                        "key": "gallbladder",
                        "value": "gallbladder",
                        "display_name": "Gallbladder",
                        "prompt": "gallbladder",
                        "color": "#00FF80",
                    },
                ],
            },
        )

        config = parse_sam3_model_config(model)

        self.assertEqual(config.default_prompt_key, "gallbladder")
        self.assertEqual(len(config.prompt_options), 1)
        self.assertEqual(config.prompt_options[0].display_name, "Gallbladder")
        self.assertEqual(config.prompt_options[0].prompt, "gallbladder")
        self.assertEqual(config.prompt_options[0].auto_mask_priority, None)
        self.assertEqual(config.prompt_options[0].component_mode, "union")
        self.assertEqual(config.prompt_options[0].min_component_area, 1024)

    def test_parse_sam3_model_config_parses_prompt_auto_mask_priority(self):
        model = SurgeryModel(
            name="sam3-priority",
            procedure_type="cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/checkpoint.pt",
                "prompt_options": [
                    {
                        "key": "cystic_duct",
                        "display_name": "Cystic Duct",
                        "prompt": "cystic duct",
                        "auto_mask_priority": 10,
                    },
                ],
            },
        )

        config = parse_sam3_model_config(model)

        self.assertEqual(config.prompt_options[0].auto_mask_priority, 10)
        self.assertEqual(config.prompt_options[0].component_mode, "union")

    def test_parse_sam3_model_config_uses_embedded_atlas_prompt_inventory(self):
        model = SurgeryModel(
            name="sam3-atlas",
            procedure_type="cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/atlas.pt",
                "variant_key": SAM3_ATLAS_VARIANT_KEY,
                "confidence_threshold": 0.7,
                "score_threshold": 0.7,
                "default_prompt_key": "gallbladder",
                "prompt_options": [
                    {
                        "key": "wrong_prompt",
                        "display_name": "Wrong Prompt",
                        "prompt": "wrong prompt",
                    },
                ],
            },
        )

        config = parse_sam3_model_config(model)

        self.assertEqual(config.confidence_threshold, SAM3_ATLAS_CONFIDENCE_THRESHOLD)
        self.assertEqual(config.score_threshold, SAM3_ATLAS_SCORE_THRESHOLD)
        self.assertEqual(config.default_prompt_key, "gallbladder")
        self.assertEqual(len(config.prompt_options), 45)
        self.assertEqual(config.prompt_options[0].prompt, "tools")
        self.assertTrue(any(option.prompt == "pancreas" for option in config.prompt_options))
        self.assertFalse(any(option.prompt == "wrong prompt" for option in config.prompt_options))

    @patch("cvat.apps.engine.sam3.find_matching_models")
    @patch("cvat.apps.engine.sam3.get_procedure_types")
    def test_get_job_sam3_models_returns_matching_variants(self, mock_get_procedure_types, mock_find_matching_models):
        mock_get_procedure_types.return_value = ["Cholecystectomy"]
        mock_find_matching_models.return_value = [
            SurgeryModel(
                id=1,
                name="sam3-dsad",
                procedure_type="Cholecystectomy",
                model_type=SurgeryModelType.ANATOMY_SEGMENTER,
                endpoint_url="http://sam3:8090",
                config={
                    "checkpoint_path": "/models/dsad.pt",
                    "priority": 20,
                },
            ),
            SurgeryModel(
                id=2,
                name="sam3-cholec8k",
                procedure_type="Cholecystectomy",
                model_type=SurgeryModelType.ANATOMY_SEGMENTER,
                endpoint_url="http://sam3:8090",
                config={
                    "checkpoint_path": "/models/cholec8k.pt",
                    "default": True,
                    "priority": 10,
                    "prompt_options": [
                        {
                            "key": "gallbladder",
                            "display_name": "Gallbladder",
                            "prompt": "gallbladder",
                            "auto_mask_priority": 60,
                        },
                    ],
                },
            ),
        ]

        result = get_job_sam3_models(MagicMock(id=4793))

        self.assertEqual([item["id"] for item in result], [2, 1])
        self.assertTrue(all(item["is_matching_procedure"] for item in result))
        self.assertEqual(result[0]["prompt_options"][0]["auto_mask_priority"], 60)
        self.assertEqual(result[0]["prompt_options"][0]["component_mode"], "union")

    @patch("cvat.apps.engine.sam3.SurgeryModel.objects.filter")
    @patch("cvat.apps.engine.sam3.find_matching_models")
    @patch("cvat.apps.engine.sam3.get_procedure_types")
    def test_get_job_sam3_models_falls_back_to_all_active_variants(
        self,
        mock_get_procedure_types,
        mock_find_matching_models,
        mock_filter,
    ):
        mock_get_procedure_types.return_value = ["Appendectomy"]
        mock_find_matching_models.return_value = []
        mock_filter.return_value = [
            SurgeryModel(
                id=3,
                name="sam3-cholec8k",
                procedure_type="Cholecystectomy",
                model_type=SurgeryModelType.ANATOMY_SEGMENTER,
                endpoint_url="http://sam3:8090",
                config={
                    "checkpoint_path": "/models/cholec8k.pt",
                    "default": True,
                    "priority": 10,
                },
            ),
            SurgeryModel(
                id=4,
                name="sam3-dsad",
                procedure_type="Cholecystectomy",
                model_type=SurgeryModelType.ANATOMY_SEGMENTER,
                endpoint_url="http://sam3:8090",
                config={
                    "checkpoint_path": "/models/dsad.pt",
                    "priority": 20,
                },
            ),
        ]

        result = get_job_sam3_models(MagicMock(id=4793))

        self.assertEqual([item["id"] for item in result], [3, 4])
        self.assertTrue(all(not item["is_matching_procedure"] for item in result))

    @patch("cvat.apps.engine.serializers.ProjectWriteSerializer.update_child_objects_on_labels_update")
    @patch("cvat.apps.engine.serializers.LabelSerializer.update_labels")
    @patch("cvat.apps.engine.sam3._get_parent_labels")
    @patch("cvat.apps.engine.sam3._get_job_label_parent")
    @patch("cvat.apps.engine.sam3.resolve_job_sam3_model")
    def test_sync_job_sam3_labels_creates_missing_mask_labels(
        self,
        mock_resolve_model,
        mock_get_parent,
        mock_get_parent_labels,
        mock_update_labels,
        mock_update_children,
    ):
        parent = Project(id=17, name="demo")
        parent.touch = MagicMock()
        mock_get_parent.return_value = parent
        mock_get_parent_labels.return_value = [
            Label(name="Gallbladder", type="mask"),
            Label(name="Existing Polygon", type="polygon"),
        ]
        mock_resolve_model.return_value = (
            SurgeryModel(
                id=1,
                name="sam3",
                procedure_type="Cholecystectomy",
                model_type=SurgeryModelType.ANATOMY_SEGMENTER,
                endpoint_url="http://sam3:8090",
            ),
            parse_sam3_model_config(
                SurgeryModel(
                    name="sam3-prompts",
                    procedure_type="Cholecystectomy",
                    model_type=SurgeryModelType.ANATOMY_SEGMENTER,
                    endpoint_url="http://sam3:8090",
                    config={
                        "checkpoint_path": "/models/checkpoint.pt",
                        "prompt_options": [
                            {
                                "key": "gallbladder",
                                "display_name": "Gallbladder",
                                "prompt": "gallbladder",
                            },
                            {
                                "key": "cystic_duct",
                                "display_name": "Cystic Duct",
                                "prompt": "cystic duct",
                            },
                            {
                                "key": "existing_polygon",
                                "display_name": "Existing Polygon",
                                "prompt": "existing polygon",
                            },
                        ],
                    },
                )
            ),
        )

        result = sync_job_sam3_labels(MagicMock(id=4793), model_id=1)

        self.assertEqual(result["created_labels"], ["Cystic Duct"])
        self.assertEqual(result["existing_labels"], ["Gallbladder"])
        self.assertEqual(
            result["conflicts"],
            [
                {
                    "prompt_key": "existing_polygon",
                    "prompt_display_name": "Existing Polygon",
                    "label_name": "Existing Polygon",
                    "label_type": "polygon",
                }
            ],
        )
        self.assertTrue(result["reload_required"])
        mock_update_labels.assert_called_once_with(
            [{"name": "Cystic Duct", "type": "mask", "attributes": []}],
            parent_instance=parent,
        )
        parent.touch.assert_called_once()
        mock_update_children.assert_called_once_with(parent)

    @patch("cvat.apps.engine.sam3.request_sidecar")
    @patch("cvat.apps.engine.sam3.resolve_job_label")
    @patch("cvat.apps.engine.sam3.resolve_job_sam3_model")
    def test_infer_job_sam3_points_forwards_prompt_and_score_threshold(
        self,
        mock_resolve_model,
        mock_resolve_label,
        mock_request_sidecar,
    ):
        model = SurgeryModel(
            id=7,
            name="sam3-cholec8k",
            procedure_type="Cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/cholec8k.pt",
                "score_threshold": 0.3,
            },
        )
        mock_resolve_model.return_value = (model, parse_sam3_model_config(model))
        mock_request_sidecar.return_value = {
            "confidence": 0.91,
            "mask_png_b64": _encode_mask(np.array([[1]], dtype=np.uint8)),
            "logits_token": "token-1",
        }

        result = infer_job_sam3_points(
            MagicMock(id=4793),
            model_id=7,
            session_id="session-1",
            label_name="Gallbladder",
            prompt="gallbladder",
            points=[{"x": 12, "y": 24, "label": 1}],
            logits_token=None,
            multimask_output=True,
            score_threshold=0.42,
        )

        self.assertEqual(result["confidence"], 0.91)
        self.assertEqual(result["logits_token"], "token-1")
        mock_resolve_label.assert_called_once()
        mock_request_sidecar.assert_called_once_with(
            "http://sam3:8090",
            method="POST",
            path="/infer/points",
            json_payload={
                "session_id": "session-1",
                "points": [{"x": 12.0, "y": 24.0, "label": 1}],
                "logits_token": None,
                "initial_mask_rle": None,
                "multimask_output": True,
                "prompt": "gallbladder",
                "score_threshold": 0.42,
            },
        )

    @patch("cvat.apps.engine.sam3.request_sidecar")
    @patch("cvat.apps.engine.sam3.resolve_job_label")
    @patch("cvat.apps.engine.sam3.resolve_job_sam3_model")
    def test_infer_job_sam3_points_forwards_initial_mask_rle(
        self,
        mock_resolve_model,
        mock_resolve_label,
        mock_request_sidecar,
    ):
        model = SurgeryModel(
            id=8,
            name="sam3-cholec8k",
            procedure_type="Cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/cholec8k.pt",
                "score_threshold": 0.3,
            },
        )
        mock_resolve_model.return_value = (model, parse_sam3_model_config(model))
        mock_request_sidecar.return_value = {
            "confidence": 0.4,
            "mask_png_b64": None,
            "logits_token": None,
        }

        infer_job_sam3_points(
            MagicMock(id=4793),
            model_id=8,
            session_id="session-2",
            label_name="Liver Ligament",
            prompt="liver ligament",
            points=[{"x": 100, "y": 200, "label": 1}],
            logits_token=None,
            initial_mask_rle=[0, 4, 3, 1, 4, 2],
            multimask_output=True,
            score_threshold=0.3,
        )

        mock_resolve_label.assert_called_once()
        mock_request_sidecar.assert_called_once_with(
            "http://sam3:8090",
            method="POST",
            path="/infer/points",
            json_payload={
                "session_id": "session-2",
                "points": [{"x": 100.0, "y": 200.0, "label": 1}],
                "logits_token": None,
                "initial_mask_rle": [0, 4, 3, 1, 4, 2],
                "multimask_output": True,
                "prompt": "liver ligament",
                "score_threshold": 0.3,
            },
        )

    @patch("cvat.apps.engine.sam3.request_sidecar")
    @patch("cvat.apps.engine.sam3.load_job_frame_rgb")
    @patch("cvat.apps.engine.sam3.resolve_job_sam3_model")
    def test_create_job_sam3_session_forwards_variant_key(
        self,
        mock_resolve_model,
        mock_load_job_frame_rgb,
        mock_request_sidecar,
    ):
        model = SurgeryModel(
            id=8,
            name="sam3-atlas",
            procedure_type="Cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/atlas.pt",
                "device": "cuda",
                "variant_key": SAM3_ATLAS_VARIANT_KEY,
            },
        )
        mock_resolve_model.return_value = (model, parse_sam3_model_config(model))
        mock_load_job_frame_rgb.return_value = np.zeros((4, 5, 3), dtype=np.uint8)
        mock_request_sidecar.return_value = {
            "session_id": "session-1",
            "width": 5,
            "height": 4,
        }

        result = create_job_sam3_session(
            MagicMock(id=4793),
            model_id=8,
            frame=0,
        )

        self.assertEqual(result["session_id"], "session-1")
        self.assertEqual(mock_request_sidecar.call_args.args[0], "http://sam3:8090")
        self.assertEqual(mock_request_sidecar.call_args.kwargs["path"], "/sessions")
        self.assertEqual(
            mock_request_sidecar.call_args.kwargs["data"]["checkpoint_path"],
            "/models/atlas.pt",
        )
        self.assertEqual(
            mock_request_sidecar.call_args.kwargs["data"]["variant_key"],
            SAM3_ATLAS_VARIANT_KEY,
        )

    @patch("cvat.apps.engine.sam3.request_sidecar")
    @patch("cvat.apps.engine.sam3.TaskFrameProvider")
    @patch("cvat.apps.engine.sam3.resolve_job_sam3_model")
    def test_create_job_sam3_video_session_forwards_checkpoint_path(
        self,
        mock_resolve_model,
        mock_task_frame_provider,
        mock_request_sidecar,
    ):
        model = SurgeryModel(
            id=9,
            name="sam3-atlas",
            procedure_type="Cholecystectomy",
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            endpoint_url="http://sam3:8090",
            config={
                "checkpoint_path": "/models/atlas.pt",
                "device": "cuda",
            },
        )
        mock_resolve_model.return_value = (model, parse_sam3_model_config(model))

        frame_result = MagicMock()
        frame_result.data = np.zeros((4, 5, 3), dtype=np.uint8)
        provider = mock_task_frame_provider.return_value
        provider.get_frame.return_value = frame_result

        mock_request_sidecar.return_value = {
            "session_id": "video-session-1",
            "num_frames": 2,
            "frame_index_map": {10: 0, 11: 1},
            "width": 5,
            "height": 4,
        }

        job = MagicMock(id=4793)
        job.segment.frame_set = {10, 11}
        job.segment.task = MagicMock()

        result = create_job_sam3_video_session(
            job,
            model_id=9,
            start_frame=10,
            stop_frame=11,
        )

        self.assertEqual(result["session_id"], "video-session-1")
        self.assertEqual(mock_request_sidecar.call_args.args[0], "http://sam3:8090")
        self.assertEqual(mock_request_sidecar.call_args.kwargs["path"], "/sessions/video")
        self.assertEqual(
            mock_request_sidecar.call_args.kwargs["data"]["checkpoint_path"],
            "/models/atlas.pt",
        )
        self.assertEqual(
            mock_request_sidecar.call_args.kwargs["data"]["device"],
            "cuda",
        )
