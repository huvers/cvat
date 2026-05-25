from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import numpy as np
import requests
from PIL import Image
from rest_framework.exceptions import APIException, ValidationError

from cvat.apps.engine.frame_provider import FrameOutputType, TaskFrameProvider
from cvat.apps.engine.models import (
    FrameQuality,
    Job,
    Label,
    LabelType,
    Project,
    SurgeryModel,
    SurgeryModelType,
    Task,
)
from cvat.apps.engine.weak_labeling import find_matching_models, get_procedure_types


SAM3_DEFAULT_PRIORITY = 100
SAM3_DEFAULT_DEVICE = "cuda"
SAM3_DEFAULT_CONFIDENCE_THRESHOLD = 0.7
SAM3_DEFAULT_SCORE_THRESHOLD = 0.7
SAM3_ATLAS_VARIANT_KEY = "atlas_3_1"
SAM3_ATLAS_CONFIDENCE_THRESHOLD = 0.3
SAM3_ATLAS_SCORE_THRESHOLD = 0.3
SAM3_ATLAS_DEFAULT_PROMPT_KEY = "liver"
SAM3_ALLOWED_LABEL_TYPES = {LabelType.MASK.value, LabelType.ANY.value}
SAM3_DEFAULT_COMPONENT_MODE = "union"
SAM3_DEFAULT_UNION_MIN_COMPONENT_AREA = 1024
SAM3_DEFAULT_SPLIT_MIN_COMPONENT_AREA = 192
SAM3_ATLAS_PROMPTS: tuple[str, ...] = (
    "tools",
    "vein",
    "artery",
    "nerve",
    "small intestine",
    "colon",
    "abdominal wall",
    "diaphragm",
    "omentum",
    "aorta",
    "vena cava",
    "liver",
    "cystic duct",
    "gallbladder",
    "hepatic vein",
    "hepatic ligament",
    "cystic plate",
    "stomach",
    "ductus choledochus",
    "mesenterium",
    "ductus hepaticus",
    "spleen",
    "uterus",
    "ovary",
    "oviduct",
    "prostate",
    "urethra",
    "ligated plexus",
    "seminal vesicles",
    "catheter",
    "bladder",
    "kidney",
    "lung",
    "airway",
    "esophagus",
    "pericardium",
    "azygos vein",
    "thoracic duct",
    "nerves",
    "ureter",
    "non anatomical structures",
    "mesocolon",
    "adrenal gland",
    "pancreas",
    "duodenum",
)
SAM3_DEFAULT_AUTO_MASK_PRIORITIES: dict[str, dict[str, int]] = {
    "cholec8k": {
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
    "dsad": {
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
SAM3_DEFAULT_COMPONENT_RULES: dict[str, dict[str, dict[str, int | str]]] = {
    "cholec8k": {
        "abdominal_wall": {"component_mode": "union", "min_component_area": 1600},
        "blood": {"component_mode": "union", "min_component_area": 1200},
        "connective_tissue": {"component_mode": "union", "min_component_area": 1200},
        "cystic_duct": {"component_mode": "components", "min_component_area": 96},
        "fat": {"component_mode": "union", "min_component_area": 1200},
        "gallbladder": {"component_mode": "union", "min_component_area": 384},
        "gastrointestinal_tract": {"component_mode": "union", "min_component_area": 640},
        "grasper": {"component_mode": "components", "min_component_area": 384},
        "hepatic_vein": {"component_mode": "components", "min_component_area": 96},
        "l_hook_electrocautery": {"component_mode": "components", "min_component_area": 384},
        "liver": {"component_mode": "union", "min_component_area": 1600},
        "liver_ligament": {"component_mode": "components", "min_component_area": 128},
    },
    "dsad": {
        "abdominal_wall": {"component_mode": "union", "min_component_area": 1600},
        "colon": {"component_mode": "union", "min_component_area": 1024},
        "inferior_mesenteric_artery": {"component_mode": "components", "min_component_area": 96},
        "intestinal_veins": {"component_mode": "components", "min_component_area": 128},
        "liver": {"component_mode": "union", "min_component_area": 1024},
        "pancreas": {"component_mode": "union", "min_component_area": 640},
        "small_intestine": {"component_mode": "union", "min_component_area": 1200},
        "spleen": {"component_mode": "union", "min_component_area": 640},
        "stomach": {"component_mode": "union", "min_component_area": 640},
        "ureter": {"component_mode": "components", "min_component_area": 96},
        "vesicular_glands": {"component_mode": "components", "min_component_area": 128},
    },
}


class SAM3ProxyError(APIException):
    status_code = 502
    default_detail = "Failed to contact the SAM3 sidecar"
    default_code = "sam3_proxy_error"

    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail=detail)
        self.status_code = status_code


@dataclass(frozen=True)
class SAM3PromptOption:
    key: str
    value: str | None
    display_name: str
    prompt: str
    color: str | None
    auto_mask_priority: int | None
    component_mode: str | None
    min_component_area: int | None


@dataclass(frozen=True)
class SAM3ModelConfig:
    display_name: str
    variant_key: str | None
    priority: int
    is_default: bool
    checkpoint_path: str
    device: str
    confidence_threshold: float
    score_threshold: float
    default_prompt_key: str | None
    prompt_options: tuple[SAM3PromptOption, ...]


def _normalize_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Expected an integer value, got {value!r}") from exc


def _normalize_float(value: Any, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Expected a numeric value, got {value!r}") from exc


def _normalize_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Expected an integer value, got {value!r}") from exc


def _normalize_optional_component_mode(value: Any) -> str | None:
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"union", "components"}:
        raise ValidationError(f"Expected component mode 'union' or 'components', got {value!r}")
    return normalized


def normalize_prompt_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _prompt_key(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_")


def _atlas_prompt_options() -> tuple[tuple[SAM3PromptOption, ...], str]:
    return (
        tuple(
            SAM3PromptOption(
                key=_prompt_key(prompt),
                value=_prompt_key(prompt),
                display_name=prompt.title(),
                prompt=prompt,
                color=None,
                auto_mask_priority=None,
                component_mode=None,
                min_component_area=None,
            )
            for prompt in SAM3_ATLAS_PROMPTS
        ),
        SAM3_ATLAS_DEFAULT_PROMPT_KEY,
    )


def _parse_prompt_options(config: dict[str, Any]) -> tuple[tuple[SAM3PromptOption, ...], str | None]:
    raw_prompt_options = config.get("prompt_options") or []
    if not isinstance(raw_prompt_options, list):
        raise ValidationError("config.prompt_options must be a list if provided")

    normalized_options: list[SAM3PromptOption] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(raw_prompt_options):
        if not isinstance(item, dict):
            raise ValidationError(f"config.prompt_options[{index}] must be an object")

        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            raise ValidationError(f"config.prompt_options[{index}] is missing a prompt")

        key = str(item.get("key") or item.get("value") or prompt).strip()
        if not key:
            raise ValidationError(f"config.prompt_options[{index}] is missing a key")
        if key in seen_keys:
            raise ValidationError(f"config.prompt_options contains a duplicate key '{key}'")
        seen_keys.add(key)

        display_name = str(item.get("display_name") or item.get("name") or item.get("value") or prompt).strip()
        value = str(item.get("value")).strip() if item.get("value") not in (None, "") else None
        color = str(item.get("color")).strip() if item.get("color") not in (None, "") else None
        normalized_options.append(
            SAM3PromptOption(
                key=key,
                value=value,
                display_name=display_name,
                prompt=prompt,
                color=color,
                auto_mask_priority=_normalize_optional_int(item.get("auto_mask_priority")),
                component_mode=_normalize_optional_component_mode(item.get("component_mode")),
                min_component_area=_normalize_optional_int(item.get("min_component_area")),
            )
        )

    default_prompt_key = config.get("default_prompt_key")
    if default_prompt_key not in (None, ""):
        default_prompt_key = str(default_prompt_key).strip()
        if default_prompt_key and default_prompt_key not in seen_keys:
            default_prompt_key = None
    else:
        default_prompt_key = None

    return tuple(normalized_options), default_prompt_key


def _iter_prompt_tokens(option: SAM3PromptOption) -> set[str]:
    return {
        normalize_prompt_token(str(value))
        for value in (option.display_name, option.prompt, option.value, option.key)
        if value
    }


def _apply_default_auto_mask_priorities(
    prompt_options: tuple[SAM3PromptOption, ...],
    *,
    variant_key: str | None,
) -> tuple[SAM3PromptOption, ...]:
    if not variant_key:
        return prompt_options

    overrides = SAM3_DEFAULT_AUTO_MASK_PRIORITIES.get((variant_key or "").strip().lower(), {})
    normalized_options: list[SAM3PromptOption] = []
    for index, option in enumerate(prompt_options):
        if option.auto_mask_priority is not None:
            normalized_options.append(option)
            continue

        fallback_priority = overrides.get(
            str(option.value or option.key).strip(),
            (index + 1) * 100,
        )
        normalized_options.append(
            SAM3PromptOption(
                key=option.key,
                value=option.value,
                display_name=option.display_name,
                prompt=option.prompt,
                color=option.color,
                auto_mask_priority=fallback_priority,
                component_mode=option.component_mode,
                min_component_area=option.min_component_area,
            )
        )
    return tuple(normalized_options)


def _apply_default_component_rules(
    prompt_options: tuple[SAM3PromptOption, ...],
    *,
    variant_key: str | None,
) -> tuple[SAM3PromptOption, ...]:
    rules = SAM3_DEFAULT_COMPONENT_RULES.get((variant_key or "").strip().lower(), {})
    normalized_options: list[SAM3PromptOption] = []
    for option in prompt_options:
        rule = rules.get(str(option.value or option.key).strip(), {})
        component_mode = option.component_mode or str(rule.get("component_mode") or SAM3_DEFAULT_COMPONENT_MODE)
        min_component_area = option.min_component_area
        if min_component_area is None:
            default_min_area = rule.get("min_component_area")
            if default_min_area is None:
                default_min_area = (
                    SAM3_DEFAULT_SPLIT_MIN_COMPONENT_AREA if component_mode == "components"
                    else SAM3_DEFAULT_UNION_MIN_COMPONENT_AREA
                )
            min_component_area = int(default_min_area)

        normalized_options.append(
            SAM3PromptOption(
                key=option.key,
                value=option.value,
                display_name=option.display_name,
                prompt=option.prompt,
                color=option.color,
                auto_mask_priority=option.auto_mask_priority,
                component_mode=component_mode,
                min_component_area=min_component_area,
            )
        )
    return tuple(normalized_options)


def _get_job_label_parent(job: Job) -> Project | Task:
    task = job.segment.task
    return task.project if task.project_id else task


def _get_parent_labels(parent: Project | Task) -> list[Label]:
    filters: dict[str, Any] = {"parent__isnull": True}
    if isinstance(parent, Project):
        filters["project_id"] = parent.id
    else:
        filters["task_id"] = parent.id
    return list(Label.objects.filter(**filters).order_by("id"))


def parse_sam3_model_config(model: SurgeryModel) -> SAM3ModelConfig:
    config = model.config or {}
    checkpoint_path = str(config.get("checkpoint_path") or "").strip()
    if not checkpoint_path:
        raise ValidationError(f"SAM3 model {model.id} is missing config.checkpoint_path")

    device = str(config.get("device") or SAM3_DEFAULT_DEVICE).strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValidationError(f"SAM3 model {model.id} has invalid config.device '{device}'")

    variant_key = str(config["variant_key"]).strip() if config.get("variant_key") else None
    prompt_options, default_prompt_key = _parse_prompt_options(config)
    if (variant_key or "").strip().lower() == SAM3_ATLAS_VARIANT_KEY:
        atlas_prompt_options, atlas_default_prompt_key = _atlas_prompt_options()
        configured_default_prompt_key = str(config.get("default_prompt_key") or "").strip() or None
        prompt_options = atlas_prompt_options
        atlas_prompt_keys = {option.key for option in atlas_prompt_options}
        default_prompt_key = (
            configured_default_prompt_key
            if configured_default_prompt_key in atlas_prompt_keys
            else atlas_default_prompt_key
        )
    prompt_options = _apply_default_auto_mask_priorities(
        prompt_options,
        variant_key=variant_key,
    )
    prompt_options = _apply_default_component_rules(
        prompt_options,
        variant_key=variant_key,
    )

    return SAM3ModelConfig(
        display_name=str(config.get("display_name") or model.name),
        variant_key=variant_key,
        priority=_normalize_int(config.get("priority"), default=SAM3_DEFAULT_PRIORITY),
        is_default=_normalize_bool(config.get("default"), default=False),
        checkpoint_path=checkpoint_path,
        device=device,
        confidence_threshold=(
            SAM3_ATLAS_CONFIDENCE_THRESHOLD
            if (variant_key or "").strip().lower() == SAM3_ATLAS_VARIANT_KEY
            else _normalize_float(
                config.get("confidence_threshold"),
                default=SAM3_DEFAULT_CONFIDENCE_THRESHOLD,
            )
        ),
        score_threshold=(
            SAM3_ATLAS_SCORE_THRESHOLD
            if (variant_key or "").strip().lower() == SAM3_ATLAS_VARIANT_KEY
            else _normalize_float(
                config.get("score_threshold"),
                default=SAM3_DEFAULT_SCORE_THRESHOLD,
            )
        ),
        default_prompt_key=default_prompt_key,
        prompt_options=prompt_options,
    )


def get_job_sam3_models(job: Job) -> list[dict[str, Any]]:
    procedure_types = get_procedure_types(job)
    if procedure_types:
        matched_models = find_matching_models(
            procedure_types,
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
        )
    else:
        matched_models = []

    if matched_models:
        candidate_models = matched_models
        matching_procedure_types = {procedure_type.casefold() for procedure_type in procedure_types}
    else:
        candidate_models = list(
            SurgeryModel.objects.filter(
                is_active=True,
                model_type=SurgeryModelType.ANATOMY_SEGMENTER,
            )
        )
        matching_procedure_types = set()

    unique_models: dict[int, SurgeryModel] = {}
    for model in candidate_models:
        unique_models[model.id] = model

    response: list[dict[str, Any]] = []
    for model in unique_models.values():
        config = parse_sam3_model_config(model)
        response.append(
            {
                "id": model.id,
                "name": model.name,
                "display_name": config.display_name,
                "procedure_type": model.procedure_type,
                "variant_key": config.variant_key,
                "is_default": config.is_default,
                "is_matching_procedure": model.procedure_type.casefold() in matching_procedure_types,
                "priority": config.priority,
                "device": config.device,
                "confidence_threshold": config.confidence_threshold,
                "score_threshold": config.score_threshold,
                "default_prompt_key": config.default_prompt_key,
                "prompt_options": [
                    {
                        "key": option.key,
                        "value": option.value,
                        "display_name": option.display_name,
                        "prompt": option.prompt,
                        "color": option.color,
                        "auto_mask_priority": option.auto_mask_priority,
                        "component_mode": option.component_mode,
                        "min_component_area": option.min_component_area,
                    }
                    for option in config.prompt_options
                ],
            }
        )

    response.sort(
        key=lambda item: (
            0 if item["is_matching_procedure"] else 1,
            0 if item["is_default"] else 1,
            item["priority"],
            item["name"].lower(),
            item["id"],
        )
    )
    return response


def resolve_job_sam3_model(job: Job, model_id: int) -> tuple[SurgeryModel, SAM3ModelConfig]:
    matched_models = {
        model["id"]: model for model in get_job_sam3_models(job)
    }
    if model_id not in matched_models:
        raise ValidationError(f"SAM3 model {model_id} is not available for job {job.id}")

    try:
        model = SurgeryModel.objects.get(
            id=model_id,
            is_active=True,
            model_type=SurgeryModelType.ANATOMY_SEGMENTER,
        )
    except SurgeryModel.DoesNotExist as exc:
        raise ValidationError(f"SAM3 model {model_id} is not available for job {job.id}") from exc
    return model, parse_sam3_model_config(model)


def resolve_job_label(job: Job, label_name: str) -> Label:
    task = job.segment.task
    label = Label.objects.filter(task=task, name=label_name).first()
    if label is None and task.project_id:
        label = Label.objects.filter(project_id=task.project_id, name=label_name).first()
    if label is None:
        raise ValidationError(f"Label '{label_name}' does not exist for job {job.id}")
    if label.type not in SAM3_ALLOWED_LABEL_TYPES:
        raise ValidationError(
            f"Label '{label_name}' must be of type 'mask' or 'any' for SAM3 inference"
        )
    return label


def sync_job_sam3_labels(job: Job, *, model_id: int) -> dict[str, Any]:
    _, config = resolve_job_sam3_model(job, model_id)
    parent = _get_job_label_parent(job)
    existing_labels = _get_parent_labels(parent)

    normalized_existing_labels: list[tuple[Label, str]] = [
        (label, normalize_prompt_token(label.name))
        for label in existing_labels
    ]

    labels_to_create: list[dict[str, Any]] = []
    created_names: list[str] = []
    existing_names: list[str] = []
    conflicts: list[dict[str, str]] = []

    for option in config.prompt_options:
        target_name = option.display_name.strip()
        if not target_name:
            continue

        prompt_tokens = _iter_prompt_tokens(option)
        matched_label = next(
            (
                label for label, normalized_name in normalized_existing_labels
                if normalized_name in prompt_tokens
            ),
            None,
        )

        if matched_label is not None:
            if matched_label.type in SAM3_ALLOWED_LABEL_TYPES:
                existing_names.append(matched_label.name)
            else:
                conflicts.append(
                    {
                        "prompt_key": option.key,
                        "prompt_display_name": option.display_name,
                        "label_name": matched_label.name,
                        "label_type": matched_label.type,
                    }
                )
            continue

        labels_to_create.append(
            {
                "name": target_name,
                "type": LabelType.MASK.value,
                "attributes": [],
            }
        )
        created_names.append(target_name)
        normalized_existing_labels.append(
            (
                Label(name=target_name, type=LabelType.MASK.value),
                normalize_prompt_token(target_name),
            )
        )

    if labels_to_create:
        from cvat.apps.engine.serializers import LabelSerializer, ProjectWriteSerializer, TaskWriteSerializer

        LabelSerializer.update_labels(labels_to_create, parent_instance=parent)
        parent.touch()
        if isinstance(parent, Project):
            ProjectWriteSerializer(parent).update_child_objects_on_labels_update(parent)
        else:
            TaskWriteSerializer(parent).update_child_objects_on_labels_update(parent)

    return {
        "created_labels": created_names,
        "existing_labels": sorted(set(existing_names)),
        "conflicts": conflicts,
        "reload_required": bool(labels_to_create),
    }


def validate_job_frame(job: Job, frame: int) -> int:
    try:
        frame = int(frame)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Field 'frame' must be an integer") from exc

    if frame not in job.segment.frame_set:
        raise ValidationError(f"Frame {frame} is outside job {job.id}")
    return frame


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return frame[:, :, ::-1].copy()


def load_job_frame_rgb(job: Job, frame: int) -> np.ndarray:
    frame = validate_job_frame(job, frame)
    provider = TaskFrameProvider(job.segment.task)
    frame_data = provider.get_frame(
        frame,
        quality=FrameQuality.ORIGINAL,
        out_type=FrameOutputType.NUMPY_ARRAY,
    ).data
    return _bgr_to_rgb(np.asarray(frame_data))


def normalize_sidecar_base_url(endpoint_url: str) -> str:
    parsed = urlparse(endpoint_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/predict"):
        path = path[: -len("/predict")]
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def _rle(flat_mask: np.ndarray) -> list[int]:
    flat_mask = flat_mask.astype(np.uint8, copy=False)
    if flat_mask.size == 0:
        return []

    pairwise_unequal = flat_mask[1:] != flat_mask[:-1]
    runs = np.diff(np.nonzero(pairwise_unequal)[0], prepend=-1, append=len(flat_mask) - 1)
    rle = runs.tolist()
    if flat_mask[0] != 0:
        rle.insert(0, 0)
    return [int(value) for value in rle]


def mask_png_b64_to_cvat_rle(mask_png_b64: str | None) -> list[int] | None:
    if not mask_png_b64:
        return None

    payload = base64.b64decode(mask_png_b64)
    with Image.open(io.BytesIO(payload)) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 127

    if not np.any(mask):
        return None

    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    tight_mask = mask[top : bottom + 1, left : right + 1]
    return _rle(tight_mask.reshape(-1)) + [left, top, right, bottom]


def request_sidecar(
    endpoint_url: str,
    *,
    method: str,
    path: str,
    json_payload: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    timeout: tuple[int, int] = (10, 900),
) -> Any:
    url = urljoin(f"{normalize_sidecar_base_url(endpoint_url).rstrip('/')}/", path.lstrip("/"))
    try:
        response = requests.request(
            method=method,
            url=url,
            json=json_payload,
            data=data,
            files=files,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise SAM3ProxyError(str(exc), status_code=502) from exc

    if not response.ok:
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or "Unknown sidecar error"}
        detail = payload.get("detail") or payload.get("error") or response.text
        raise SAM3ProxyError(str(detail), status_code=response.status_code)

    try:
        return response.json()
    except ValueError as exc:
        raise SAM3ProxyError("SAM3 sidecar returned non-JSON content", status_code=502) from exc


def create_job_sam3_session(
    job: Job,
    *,
    model_id: int,
    frame: int,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    model, config = resolve_job_sam3_model(job, model_id)
    confidence_threshold_value = (
        _normalize_float(confidence_threshold, default=config.confidence_threshold)
        if confidence_threshold is not None
        else config.confidence_threshold
    )
    frame_rgb = load_job_frame_rgb(job, frame)
    image = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/sessions",
        data={
            "checkpoint_path": config.checkpoint_path,
            "device": config.device,
            "confidence_threshold": str(confidence_threshold_value),
            "variant_key": config.variant_key or "",
        },
        files={"image": ("frame.png", buffer.getvalue(), "image/png")},
    )
    return response


def delete_job_sam3_session(job: Job, *, model_id: int, session_id: str) -> dict[str, Any]:
    model, _ = resolve_job_sam3_model(job, model_id)
    return request_sidecar(
        model.endpoint_url,
        method="DELETE",
        path=f"/sessions/{session_id}",
    )


def infer_job_sam3_text(
    job: Job,
    *,
    model_id: int,
    session_id: str,
    labels: list[str],
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    model, config = resolve_job_sam3_model(job, model_id)
    validated_prompts: list[str] = []
    seen: set[str] = set()
    for prompt in labels:
        prompt = str(prompt).strip()
        if not prompt or prompt in seen:
            continue
        validated_prompts.append(prompt)
        seen.add(prompt)

    score_threshold_value = (
        _normalize_float(score_threshold, default=config.score_threshold)
        if score_threshold is not None
        else config.score_threshold
    )

    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/infer/text",
        json_payload={
            "session_id": session_id,
            "labels": validated_prompts,
            "score_threshold": score_threshold_value,
            "max_masks": 0,
        },
    )

    results: list[dict[str, Any]] = []
    for item in response:
        results.append(
            {
                "label_name": item["label_name"],
                "confidence": item.get("confidence", 0.0),
                "mask_rle": mask_png_b64_to_cvat_rle(item.get("mask_png_b64")),
            }
        )
    return [item for item in results if item["mask_rle"] is not None]


def infer_job_sam3_box(
    job: Job,
    *,
    model_id: int,
    session_id: str,
    label_name: str,
    bbox: list[float],
    negative: bool,
    prompt: str | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    model, config = resolve_job_sam3_model(job, model_id)
    resolve_job_label(job, label_name)
    if len(bbox) != 4:
        raise ValidationError("Field 'bbox' must contain four coordinates")
    prompt_text = str(prompt or label_name).strip()
    score_threshold_value = (
        _normalize_float(score_threshold, default=config.score_threshold)
        if score_threshold is not None
        else config.score_threshold
    )

    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/infer/box",
        json_payload={
            "session_id": session_id,
            "label_name": prompt_text,
            "bbox": [float(value) for value in bbox],
            "negative": bool(negative),
            "score_threshold": score_threshold_value,
            "max_masks": 0,
        },
    )

    return {
        "confidence": response.get("confidence", 0.0),
        "mask_rle": mask_png_b64_to_cvat_rle(response.get("mask_png_b64")),
    }


def infer_job_sam3_points(
    job: Job,
    *,
    model_id: int,
    session_id: str,
    points: list[dict[str, Any]],
    logits_token: str | None,
    initial_mask_rle: list[int] | None = None,
    multimask_output: bool,
    label_name: str | None = None,
    prompt: str | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    model, config = resolve_job_sam3_model(job, model_id)
    normalized_points: list[dict[str, Any]] = []
    for point in points:
        label = int(point.get("label", 1))
        if label not in {0, 1}:
            raise ValidationError("Point labels must be 0 or 1")
        normalized_points.append(
            {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "label": label,
            }
        )

    if label_name is not None:
        resolve_job_label(job, str(label_name))

    prompt_text = str(prompt).strip() if prompt is not None else ""
    prompt_text = prompt_text or None
    score_threshold_value = (
        _normalize_float(score_threshold, default=config.score_threshold)
        if score_threshold is not None
        else config.score_threshold
    )

    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/infer/points",
        json_payload={
            "session_id": session_id,
            "points": normalized_points,
            "logits_token": logits_token,
            "initial_mask_rle": initial_mask_rle,
            "multimask_output": bool(multimask_output),
            "prompt": prompt_text,
            "score_threshold": score_threshold_value,
        },
    )

    return {
        "confidence": response.get("confidence", 0.0),
        "mask_rle": mask_png_b64_to_cvat_rle(response.get("mask_png_b64")),
        "logits_token": response.get("logits_token"),
    }


# ── Video tracking (SAM 3.1 multiplex) ──────────────────────────────


def _stream_sidecar(
    endpoint_url: str,
    *,
    path: str,
    json_payload: dict[str, Any],
    timeout: tuple[int, int] = (10, 1800),
):
    """POST to sidecar and yield NDJSON lines as parsed dicts."""
    url = urljoin(f"{normalize_sidecar_base_url(endpoint_url).rstrip('/')}/", path.lstrip("/"))
    request_error = requests.RequestException
    try:
        response = requests.post(url, json=json_payload, timeout=timeout, stream=True)
    except request_error as exc:
        raise SAM3ProxyError(str(exc), status_code=502) from exc

    if not response.ok:
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or "Unknown sidecar error"}
        detail = payload.get("detail") or payload.get("error") or response.text
        raise SAM3ProxyError(str(detail), status_code=response.status_code)

    import json
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)
    except request_error as exc:
        raise SAM3ProxyError(str(exc), status_code=502) from exc
    finally:
        response.close()


def _masks_to_cvat_rle(masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert mask_png_b64 entries to CVAT RLE."""
    results = []
    for entry in masks:
        rle = mask_png_b64_to_cvat_rle(entry.get("mask_png_b64"))
        if rle is not None:
            results.append({
                "obj_id": entry.get("obj_id"),
                "mask_rle": rle,
            })
    return results


def create_job_sam3_video_session(
    job: Job,
    *,
    model_id: int,
    start_frame: int,
    stop_frame: int,
    step: int = 1,
) -> dict[str, Any]:
    """Create a SAM3.1 video tracking session from a range of job frames."""
    model, config = resolve_job_sam3_model(job, model_id)

    start_frame = validate_job_frame(job, start_frame)
    stop_frame = validate_job_frame(job, stop_frame)
    if stop_frame < start_frame:
        raise ValidationError("stop_frame must be >= start_frame")
    step = max(1, int(step))

    frame_indices = list(range(start_frame, stop_frame + 1, step))
    frame_indices = [f for f in frame_indices if f in job.segment.frame_set]
    if not frame_indices:
        raise ValidationError("No valid frames in the specified range")

    # Load frames as JPEGs to keep multipart uploads small enough for interactive use.
    provider = TaskFrameProvider(job.segment.task)
    files = []
    for frame_num in frame_indices:
        frame_data = provider.get_frame(
            frame_num,
            quality=FrameQuality.ORIGINAL,
            out_type=FrameOutputType.NUMPY_ARRAY,
        ).data
        rgb = _bgr_to_rgb(np.asarray(frame_data))
        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        files.append(("images", (f"frame_{frame_num}.jpg", buffer.getvalue(), "image/jpeg")))

    import json as _json
    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/sessions/video",
        data={
            "frame_indices": _json.dumps(frame_indices),
            "device": config.device,
            "checkpoint_path": config.checkpoint_path,
        },
        files=files,
        timeout=(120, 300),
    )
    return response


def delete_job_sam3_video_session(
    job: Job, *, model_id: int, session_id: str,
) -> dict[str, Any]:
    model, _ = resolve_job_sam3_model(job, model_id)
    return request_sidecar(
        model.endpoint_url,
        method="DELETE",
        path=f"/sessions/video/{session_id}",
    )


def add_job_sam3_video_prompt(
    job: Job,
    *,
    model_id: int,
    session_id: str,
    frame: int,
    text: str | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    bounding_boxes: list[list[float]] | None = None,
    bounding_box_labels: list[int] | None = None,
    obj_id: int | None = None,
    output_prob_thresh: float = 0.5,
) -> dict[str, Any]:
    model, _ = resolve_job_sam3_model(job, model_id)
    frame = validate_job_frame(job, frame)

    payload: dict[str, Any] = {
        "session_id": session_id,
        "frame_index": frame,
        "output_prob_thresh": output_prob_thresh,
    }
    if text is not None:
        payload["text"] = str(text)
    if points is not None:
        payload["points"] = points
    if point_labels is not None:
        payload["point_labels"] = point_labels
    if bounding_boxes is not None:
        payload["bounding_boxes"] = bounding_boxes
    if bounding_box_labels is not None:
        payload["bounding_box_labels"] = bounding_box_labels
    if obj_id is not None:
        payload["obj_id"] = obj_id

    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/video/prompt",
        json_payload=payload,
    )

    masks = response.get("masks", [])
    return {
        "frame_index": response.get("frame_index", frame),
        "masks": _masks_to_cvat_rle(masks),
    }


def propagate_job_sam3_video(
    job: Job,
    *,
    model_id: int,
    session_id: str,
    direction: str = "both",
    start_frame: int | None = None,
    max_frames: int | None = None,
    output_prob_thresh: float = 0.5,
):
    """Stream propagation results. Yields dicts with frame_index and masks (CVAT RLE)."""
    model, _ = resolve_job_sam3_model(job, model_id)

    payload: dict[str, Any] = {
        "session_id": session_id,
        "direction": direction,
        "output_prob_thresh": output_prob_thresh,
    }
    if start_frame is not None:
        payload["start_frame_index"] = start_frame
    if max_frames is not None:
        payload["max_frames"] = max_frames

    for chunk in _stream_sidecar(
        model.endpoint_url,
        path="/video/propagate",
        json_payload=payload,
    ):
        masks = chunk.get("masks", [])
        yield {
            "frame_index": chunk.get("frame_index"),
            "masks": _masks_to_cvat_rle(masks),
        }


def remove_job_sam3_video_object(
    job: Job,
    *,
    model_id: int,
    session_id: str,
    obj_id: int,
    frame: int = 0,
) -> dict[str, Any]:
    model, _ = resolve_job_sam3_model(job, model_id)

    response = request_sidecar(
        model.endpoint_url,
        method="POST",
        path="/video/remove-object",
        json_payload={
            "session_id": session_id,
            "obj_id": obj_id,
            "frame_index": frame,
        },
    )
    masks = response.get("masks", [])
    return {
        "frame_index": response.get("frame_index", frame),
        "masks": _masks_to_cvat_rle(masks),
    }
