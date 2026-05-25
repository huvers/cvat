# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
import yaml

from cvat.apps.engine.models import SurgeryModel, SurgeryModelType
from cvat.apps.engine.sam3 import (
    SAM3_DEFAULT_COMPONENT_MODE,
    SAM3_DEFAULT_COMPONENT_RULES,
    SAM3_DEFAULT_CONFIDENCE_THRESHOLD,
    SAM3_DEFAULT_SCORE_THRESHOLD,
    SAM3_DEFAULT_SPLIT_MIN_COMPONENT_AREA,
    SAM3_DEFAULT_UNION_MIN_COMPONENT_AREA,
)


DEFAULT_CHECKPOINTS = [
    {
        "name": "Altas_Sam3",
        "display_name": "Altas_Sam3",
        "variant_key": "atlas_3_1",
        "priority": 0,
        "default": True,
        "checkpoint_name": "Altas_Sam3.pt",
    },
    {
        "name": "sam3_cholec8k",
        "display_name": "SAM3 Cholec8k FT",
        "variant_key": "cholec8k",
        "priority": 10,
        "default": False,
        "checkpoint_name": "checkpoint_8_cholec8k_best.pt",
        "prompt_config_name": "annotation_cholecseg8k.yaml",
    },
    {
        "name": "sam3_dsad",
        "display_name": "SAM3 DSAD FT",
        "variant_key": "dsad",
        "priority": 20,
        "default": False,
        "checkpoint_name": "dsad_ft_stg2_best.pt",
        "prompt_config_name": "dresden.yaml",
    },
]

OPTIONAL_BASE_CHECKPOINT = {
    "name": "sam3_base",
    "display_name": "SAM3 Base",
    "variant_key": "base",
    "priority": 30,
    "default": False,
    "checkpoint_name": "checkpoint.pt",
}

VIDEO_TRACKER_CHECKPOINT = {
    "name": "sam3.1_multiplex",
    "display_name": "SAM3.1 Video Tracker",
    "variant_key": "multiplex",
    "priority": 5,
    "default": False,
    "checkpoint_name": "sam3.1_multiplex.pt",
}

EMBEDDED_PROMPT_CONFIGS: dict[str, tuple[list[dict[str, str | int | None]], str | None]] = {
    "annotation_cholecseg8k.yaml": (
        [
            {"key": "abdominal_wall", "value": "abdominal_wall", "display_name": "Abdominal Wall", "prompt": "abdominal wall", "color": "#FF8000"},
            {"key": "blood", "value": "blood", "display_name": "Blood", "prompt": "blood", "color": "#FC4681"},
            {"key": "connective_tissue", "value": "connective_tissue", "display_name": "Connective Tissue", "prompt": "connective tissue", "color": "#0F7B2C"},
            {"key": "cystic_duct", "value": "cystic_duct", "display_name": "Cystic Duct", "prompt": "cystic duct", "color": "#00FFFF"},
            {"key": "fat", "value": "fat", "display_name": "Fat", "prompt": "fat", "color": "#80BF80"},
            {"key": "gallbladder", "value": "gallbladder", "display_name": "Gallbladder", "prompt": "gallbladder", "color": "#00FF80"},
            {"key": "gastrointestinal_tract", "value": "gastrointestinal_tract", "display_name": "Gastrointestinal Tract", "prompt": "gastrointestinal tract", "color": "#D384FD"},
            {"key": "grasper", "value": "grasper", "display_name": "Grasper", "prompt": "grasper", "color": "#FF00FF"},
            {"key": "hepatic_vein", "value": "hepatic_vein", "display_name": "Hepatic Vein", "prompt": "hepatic vein", "color": "#00FF00"},
            {"key": "l_hook_electrocautery", "value": "l_hook_electrocautery", "display_name": "L-hook Electrocautery", "prompt": "l-hook electrocautery", "color": "#B6003B"},
            {"key": "liver", "value": "liver", "display_name": "Liver", "prompt": "liver", "color": "#0080FF"},
            {"key": "liver_ligament", "value": "liver_ligament", "display_name": "Liver Ligament", "prompt": "liver ligament", "color": "#3406AB"},
        ],
        "gallbladder",
    ),
    "dresden.yaml": (
        [
            {"key": "abdominal_wall", "value": "abdominal_wall", "display_name": "Abdominal Wall", "prompt": "abdominal_wall", "color": "#E07A5F"},
            {"key": "colon", "value": "colon", "display_name": "Colon", "prompt": "colon", "color": "#81B29A"},
            {"key": "inferior_mesenteric_artery", "value": "inferior_mesenteric_artery", "display_name": "Inferior Mesenteric Artery", "prompt": "inferior_mesenteric_artery", "color": "#F2CC8F"},
            {"key": "intestinal_veins", "value": "intestinal_veins", "display_name": "Intestinal Veins", "prompt": "intestinal_veins", "color": "#3D405B"},
            {"key": "liver", "value": "liver", "display_name": "Liver", "prompt": "liver", "color": "#F94144"},
            {"key": "pancreas", "value": "pancreas", "display_name": "Pancreas", "prompt": "pancreas", "color": "#F3722C"},
            {"key": "small_intestine", "value": "small_intestine", "display_name": "Small Intestine", "prompt": "small_intestine", "color": "#577590"},
            {"key": "spleen", "value": "spleen", "display_name": "Spleen", "prompt": "spleen", "color": "#90BE6D"},
            {"key": "stomach", "value": "stomach", "display_name": "Stomach", "prompt": "stomach", "color": "#43AA8B"},
            {"key": "ureter", "value": "ureter", "display_name": "Ureter", "prompt": "ureter", "color": "#4D908E"},
            {"key": "vesicular_glands", "value": "vesicular_glands", "display_name": "Vesicular Glands", "prompt": "vesicular_glands", "color": "#277DA1"},
        ],
        "liver",
    ),
}

AUTO_MASK_PRIORITY_OVERRIDES: dict[str, dict[str, int]] = {
    "annotation_cholecseg8k.yaml": {
        "cystic_duct": 10,
        "hepatic_vein": 20,
        "liver_ligament": 30,
        "gallbladder": 40,
        "gastrointestinal_tract": 50,
        "fat": 60,
        "connective_tissue": 70,
        "blood": 80,
        "abdominal_wall": 90,
        "liver": 100,
        "grasper": 130,
        "l_hook_electrocautery": 140,
    },
    "dresden.yaml": {
        "ureter": 10,
        "inferior_mesenteric_artery": 20,
        "intestinal_veins": 30,
        "vesicular_glands": 40,
        "pancreas": 50,
        "spleen": 60,
        "stomach": 70,
        "colon": 80,
        "small_intestine": 90,
        "liver": 100,
        "abdominal_wall": 110,
    },
}


def apply_auto_mask_priorities(
    config_name: str,
    prompt_options: list[dict[str, str | int | None]],
) -> list[dict[str, str | int | None]]:
    priority_overrides = AUTO_MASK_PRIORITY_OVERRIDES.get(config_name, {})
    prioritized_options: list[dict[str, str | int | None]] = []
    for index, option in enumerate(prompt_options):
        option_copy = dict(option)
        if option_copy.get("auto_mask_priority") is None:
            key = str(option_copy.get("value") or option_copy.get("key") or "").strip()
            option_copy["auto_mask_priority"] = priority_overrides.get(key, (index + 1) * 100)
        prioritized_options.append(option_copy)
    return prioritized_options


def apply_component_rules(
    config_name: str,
    prompt_options: list[dict[str, str | int | None]],
) -> list[dict[str, str | int | None]]:
    variant_key = "cholec8k" if "cholec" in config_name else "dsad" if "dresden" in config_name else config_name
    component_rules = SAM3_DEFAULT_COMPONENT_RULES.get(variant_key, {})
    normalized_options: list[dict[str, str | int | None]] = []
    for option in prompt_options:
        option_copy = dict(option)
        key = str(option_copy.get("value") or option_copy.get("key") or "").strip()
        rule = component_rules.get(key, {})
        component_mode = str(option_copy.get("component_mode") or rule.get("component_mode") or SAM3_DEFAULT_COMPONENT_MODE)
        option_copy["component_mode"] = component_mode
        if option_copy.get("min_component_area") is None:
            option_copy["min_component_area"] = int(
                rule.get("min_component_area")
                or (SAM3_DEFAULT_SPLIT_MIN_COMPONENT_AREA if component_mode == "components" else SAM3_DEFAULT_UNION_MIN_COMPONENT_AREA)
            )
        normalized_options.append(option_copy)
    return normalized_options


def apply_prompt_defaults(
    config_name: str,
    prompt_options: list[dict[str, str | int | None]],
) -> list[dict[str, str | int | None]]:
    return apply_component_rules(config_name, apply_auto_mask_priorities(config_name, prompt_options))


def load_prompt_options(config_path: Path) -> tuple[list[dict[str, str | int | None]], str | None]:
    if not config_path.exists():
        embedded = EMBEDDED_PROMPT_CONFIGS.get(config_path.name)
        if embedded is not None:
            prompt_options, default_prompt_key = embedded
            return apply_prompt_defaults(config_path.name, prompt_options), default_prompt_key
        raise CommandError(f"Missing prompt config: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    classes = payload.get("classes") or {}
    if not isinstance(classes, dict):
        raise CommandError(f"Prompt config {config_path} has an invalid classes section")

    prompt_options: list[dict[str, str | int | None]] = []
    for class_key, spec in classes.items():
        if not isinstance(spec, dict):
            continue

        prompt = str(spec.get("prompt") or spec.get("value") or class_key).strip()
        if not prompt:
            continue

        value = str(spec.get("value") or class_key).strip()
        prompt_options.append(
            {
                "key": value,
                "value": value or None,
                "display_name": str(spec.get("name") or value or prompt).strip(),
                "prompt": prompt,
                "color": str(spec.get("color")).strip() if spec.get("color") else None,
            }
        )

    default_class = str(payload.get("default_class") or "").strip() or None
    default_prompt_key = None
    if default_class:
        default_spec = classes.get(default_class)
        if isinstance(default_spec, dict):
            default_prompt_key = str(default_spec.get("value") or default_class).strip() or None
        else:
            default_prompt_key = default_class

    return apply_prompt_defaults(config_path.name, prompt_options), default_prompt_key


class Command(BaseCommand):
    help = "Seed default SAM3 anatomy_segmenter registry rows for one or more procedure types."

    def add_arguments(self, parser):
        parser.add_argument(
            "--procedure-type",
            action="append",
            dest="procedure_types",
            required=True,
            help="Procedure type to attach the default SAM3 variants to. Repeat for multiple procedure types.",
        )
        parser.add_argument(
            "--endpoint-url",
            default="http://sam3:8090",
            help="Base URL for the SAM3 sidecar service.",
        )
        parser.add_argument(
            "--checkpoint-dir",
            default="/opt/checkpoints",
            help="Directory containing the SAM3 checkpoint variants.",
        )
        parser.add_argument(
            "--config-dir",
            default="/opt/sam3/configs",
            help="Directory containing the SAM3 prompt config YAMLs.",
        )
        parser.add_argument(
            "--device",
            choices=["cpu", "cuda"],
            default="cuda",
            help="Default device to store in model config.",
        )
        parser.add_argument(
            "--include-base",
            action="store_true",
            help="Also seed the base checkpoint.pt variant.",
        )
        parser.add_argument(
            "--include-video-tracker",
            action="store_true",
            help="Also seed the SAM3.1 multiplex video tracker model.",
        )

    def handle(self, *args, **options):
        checkpoint_dir = Path(options["checkpoint_dir"])
        config_dir = Path(options["config_dir"])
        endpoint_url = options["endpoint_url"]
        procedure_types: list[str] = options["procedure_types"]
        device = options["device"]
        checkpoints = [*DEFAULT_CHECKPOINTS, *( [OPTIONAL_BASE_CHECKPOINT] if options["include_base"] else [] )]

        if not checkpoint_dir.exists():
            self.stdout.write(
                self.style.WARNING(
                    f"Checkpoint directory {checkpoint_dir} is not mounted here. "
                    "Writing registry paths without local existence checks."
                )
            )

        for procedure_type in procedure_types:
            for checkpoint in checkpoints:
                checkpoint_path = checkpoint_dir / checkpoint["checkpoint_name"]
                if checkpoint_dir.exists() and not checkpoint_path.exists():
                    raise CommandError(f"Missing checkpoint: {checkpoint_path}")

                prompt_options: list[dict[str, str | int | None]] = []
                default_prompt_key: str | None = None
                prompt_config_name = checkpoint.get("prompt_config_name")
                if prompt_config_name:
                    prompt_options, default_prompt_key = load_prompt_options(config_dir / prompt_config_name)

                model, created = SurgeryModel.objects.update_or_create(
                    name=checkpoint["name"],
                    procedure_type=procedure_type,
                    defaults={
                        "model_type": SurgeryModelType.ANATOMY_SEGMENTER,
                        "endpoint_url": endpoint_url,
                        "is_active": True,
                        "config": {
                            "checkpoint_path": str(checkpoint_path),
                            "display_name": checkpoint["display_name"],
                            "variant_key": checkpoint["variant_key"],
                            "priority": checkpoint["priority"],
                            "default": checkpoint["default"],
                            "device": device,
                            "confidence_threshold": SAM3_DEFAULT_CONFIDENCE_THRESHOLD,
                            "score_threshold": SAM3_DEFAULT_SCORE_THRESHOLD,
                            "prompt_options": prompt_options,
                            "default_prompt_key": default_prompt_key,
                        },
                    },
                )
                action = "Created" if created else "Updated"
                self.stdout.write(
                    self.style.SUCCESS(
                        f"{action} {model.name} for procedure type '{procedure_type}'"
                    )
                )

        if options.get("include_video_tracker"):
            ckpt = VIDEO_TRACKER_CHECKPOINT
            checkpoint_path = checkpoint_dir / ckpt["checkpoint_name"]
            for procedure_type in procedure_types:
                model, created = SurgeryModel.objects.update_or_create(
                    name=ckpt["name"],
                    procedure_type=procedure_type,
                    defaults={
                        "model_type": SurgeryModelType.VIDEO_TRACKER,
                        "endpoint_url": endpoint_url,
                        "is_active": True,
                        "config": {
                            "checkpoint_path": str(checkpoint_path),
                            "display_name": ckpt["display_name"],
                            "variant_key": ckpt["variant_key"],
                            "priority": ckpt["priority"],
                            "default": ckpt["default"],
                            "device": device,
                        },
                    },
                )
                action = "Created" if created else "Updated"
                self.stdout.write(
                    self.style.SUCCESS(
                        f"{action} {model.name} (video tracker) for procedure type '{procedure_type}'"
                    )
                )
