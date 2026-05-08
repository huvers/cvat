from __future__ import annotations

import base64
import gc
import io
import json
import os
import sys
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from scipy import ndimage

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch is required for the SAM3 sidecar") from exc


_DEFAULT_SAM3_ROOT = Path(
    (Path(__file__).resolve().parent.parent.parent / ".." / "sam3").resolve()
)
SAM3_ROOT = Path(
    os.environ.get("SAM3_ROOT",
                   os.environ.get("MEDICAL_SAM3_ROOT", str(_DEFAULT_SAM3_ROOT)))
)
# SAM3.1 repo has the sam3 package directly under the repo root,
# while old Medical-SAM3 had an extra nesting level (sam3/sam3/).
_candidate = SAM3_ROOT / "sam3"
SAM3_CODE_ROOT = SAM3_ROOT if (_candidate / "__init__.py").exists() else _candidate
if str(SAM3_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_CODE_ROOT))

from sam3 import build_sam3_image_model, build_sam3_predictor  # noqa: E402
from sam3.model.box_ops import box_xywh_to_cxcywh  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.visualization_utils import normalize_bbox  # noqa: E402


SESSION_TTL_SECONDS = 15 * 60
MAX_SESSIONS = 32
MAX_LOGITS = 128
MAX_VIDEO_SESSIONS = 4
VIDEO_SESSION_TTL_SECONDS = 20 * 60
VIDEO_MODEL_LOCK_TIMEOUT_SECONDS = 10
CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.ckpt")
DEFAULT_CHECKPOINT_DIR = SAM3_ROOT / "checkpoints"
DEFAULT_BASE_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "checkpoint.pt"
ATLAS_VARIANT_KEY = "atlas_3_1"
ATLAS_IMAGE_RESOLUTION = 1008
POINT_BOX_PADDING = 32
POINT_FOCUS_PADDING = 128
NEGATIVE_POINT_RADIUS = 24
POINT_COMPONENT_SNAP_DISTANCE = 48
MIN_MASK_COMPONENT_AREA = 64
PRIOR_LOCALIZATION_AREA_RATIO = 4.0
PRIOR_LOCALIZATION_DILATION = 16


T = TypeVar("T")


class LruTtlStore(Generic[T]):
    def __init__(self, *, maxsize: int, ttl_seconds: int) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.Lock()

    def _monotonic(self) -> float:
        import time

        return time.monotonic()

    def _expire_locked(self) -> list[T]:
        expired: list[T] = []
        now = self._monotonic()
        expired_keys = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired_keys:
            _, value = self._entries.pop(key)
            expired.append(value)
        return expired

    def get(self, key: str) -> T | None:
        with self._lock:
            self._expire_locked()
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            now = self._monotonic()
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries[key] = (now + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> tuple[list[T], list[T]]:
        with self._lock:
            expired = self._expire_locked()
            self._entries[key] = (self._monotonic() + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            evicted: list[T] = []
            while len(self._entries) > self.maxsize:
                _, (_, old_value) = self._entries.popitem(last=False)
                evicted.append(old_value)
            return expired, evicted

    def pop(self, key: str) -> T | None:
        with self._lock:
            self._expire_locked()
            entry = self._entries.pop(key, None)
            if entry is None:
                return None
            return entry[1]

    def delete_many(self, keys: set[str]) -> None:
        with self._lock:
            for key in keys:
                self._entries.pop(key, None)

    def clear(self) -> list[T]:
        with self._lock:
            values = [value for _, value in self._entries.values()]
            self._entries.clear()
            return values


def available_devices() -> list[str]:
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]


def normalize_device(device: str) -> str:
    device = (device or "").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail=f"Unsupported device '{device}'")
    if device == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available on this host")
    return device


def normalize_variant_key(variant_key: str | None) -> str | None:
    if variant_key is None:
        return None
    normalized = str(variant_key).strip().lower()
    return normalized or None


def video_autocast(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_dir() -> Path:
    return Path(os.environ.get("SAM3_CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR))).resolve()


def list_checkpoints() -> list[dict[str, str]]:
    root = checkpoint_dir()
    if not root.exists():
        return []

    checkpoints: list[dict[str, str]] = []
    for pattern in CHECKPOINT_PATTERNS:
        for checkpoint in sorted(root.rglob(pattern)):
            checkpoints.append(
                {
                    "checkpoint_path": str(checkpoint.resolve()),
                    "display_name": checkpoint.stem.replace("_", " "),
                }
            )

    deduped: dict[str, dict[str, str]] = {}
    for checkpoint in checkpoints:
        deduped[checkpoint["checkpoint_path"]] = checkpoint
    return list(deduped.values())


def validate_checkpoint_path(checkpoint_path: str) -> str:
    resolved = Path(checkpoint_path).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint does not exist: {checkpoint_path}")

    allowed_root = checkpoint_dir()
    if allowed_root.exists():
        try:
            resolved.relative_to(allowed_root.resolve())
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Checkpoint is outside the allowed directory: {checkpoint_path}",
            ) from exc

    return str(resolved)


def encode_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) > 0).astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_upload(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def generate_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.ndim > 2:
        binary_mask = np.squeeze(binary_mask)
    if not np.any(binary_mask):
        return None

    rows = np.where(np.any(binary_mask, axis=1))[0]
    cols = np.where(np.any(binary_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def squeeze_mask_to_2d(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None

    array = np.asarray(mask)
    if array.ndim == 2:
        return array

    squeezed = np.squeeze(array)
    return squeezed if squeezed.ndim == 2 else None


def is_binary_like_mask(mask: np.ndarray) -> bool:
    array = np.asarray(mask, dtype=np.float32)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False

    return bool(np.all(
        np.logical_or.reduce((
            np.isclose(array, 0.0),
            np.isclose(array, 1.0),
            np.isclose(array, 255.0),
        ))
    ))


def decode_cvat_rle_to_full_mask(mask_rle: list[float] | list[int], width: int, height: int) -> np.ndarray | None:
    if not mask_rle or len(mask_rle) < 5:
        return None

    left, top, right, bottom = [int(round(value)) for value in mask_rle[-4:]]
    mask_width = (right - left) + 1
    mask_height = (bottom - top) + 1
    if mask_width <= 0 or mask_height <= 0:
        return None

    flat_size = mask_width * mask_height
    if flat_size <= 0:
        return None

    local_mask = np.zeros(flat_size, dtype=np.uint8)
    write_index = 0
    fill_value = 0
    for run in mask_rle[:-4]:
        run_length = max(0, int(round(run)))
        if write_index >= flat_size:
            break
        end_index = min(flat_size, write_index + run_length)
        if fill_value:
            local_mask[write_index:end_index] = 1
        write_index = end_index
        fill_value = 1 - fill_value

    if not np.any(local_mask):
        return None

    local_mask = local_mask.reshape(mask_height, mask_width)
    full_mask = np.zeros((height, width), dtype=np.uint8)

    src_left = max(0, -left)
    src_top = max(0, -top)
    dst_left = max(0, left)
    dst_top = max(0, top)
    dst_right = min(width, right + 1)
    dst_bottom = min(height, bottom + 1)
    copy_width = dst_right - dst_left
    copy_height = dst_bottom - dst_top
    if copy_width <= 0 or copy_height <= 0:
        return None

    full_mask[dst_top:dst_bottom, dst_left:dst_right] = local_mask[
        src_top:src_top + copy_height,
        src_left:src_left + copy_width,
    ]
    return full_mask if np.any(full_mask) else None


def _component_at(mask: np.ndarray, start_y: int, start_x: int) -> np.ndarray | None:
    if not mask[start_y, start_x]:
        return None

    height, width = mask.shape
    component = np.zeros_like(mask, dtype=bool)
    stack = [(start_y, start_x)]
    component[start_y, start_x] = True

    while stack:
        y, x = stack.pop()
        for next_y, next_x in (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        ):
            if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not component[next_y, next_x]:
                component[next_y, next_x] = True
                stack.append((next_y, next_x))

    return component.astype(np.uint8)


def select_bootstrap_mask(
    mask: np.ndarray,
    *,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
) -> np.ndarray | None:
    binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
    if binary_mask.ndim > 2:
        binary_mask = np.squeeze(binary_mask)
    if not np.any(binary_mask):
        return None

    height, width = binary_mask.shape

    def clamp_point(point: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        return y, x

    positive_points = [clamp_point(point) for point, label in zip(point_coords, point_labels) if int(label) == 1]
    negative_points = [clamp_point(point) for point, label in zip(point_coords, point_labels) if int(label) == 0]

    for point_y, point_x in positive_points:
        component = _component_at(binary_mask.astype(bool), point_y, point_x)
        if component is None:
            continue
        if any(component[neg_y, neg_x] for neg_y, neg_x in negative_points):
            continue
        return component

    filtered_mask = binary_mask.copy()
    for neg_y, neg_x in negative_points:
        component = _component_at(filtered_mask.astype(bool), neg_y, neg_x)
        if component is not None:
            filtered_mask[component.astype(bool)] = 0

    if np.any(filtered_mask):
        return filtered_mask.astype(np.uint8)
    return binary_mask


def clamp_point(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return y, x


def nearest_component_id(
    labels: np.ndarray,
    mask: np.ndarray,
    point_y: int,
    point_x: int,
    *,
    max_distance: int,
) -> int | None:
    if mask[point_y, point_x]:
        label_id = int(labels[point_y, point_x])
        return label_id if label_id > 0 else None

    distances, nearest_indices = ndimage.distance_transform_edt(~mask, return_indices=True)
    if distances[point_y, point_x] > max_distance:
        return None

    nearest_y = int(nearest_indices[0, point_y, point_x])
    nearest_x = int(nearest_indices[1, point_y, point_x])
    label_id = int(labels[nearest_y, nearest_x])
    return label_id if label_id > 0 else None


def select_components_for_points(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    *,
    max_snap_distance: int = POINT_COMPONENT_SNAP_DISTANCE,
) -> tuple[np.ndarray | None, set[int]]:
    binary_mask = np.asarray(mask, dtype=bool)
    if not np.any(binary_mask) or not points:
        return None, set()

    labels, _ = ndimage.label(binary_mask)
    height, width = binary_mask.shape
    selected_ids: set[int] = set()
    for point in points:
        point_y, point_x = clamp_point(point, width, height)
        label_id = nearest_component_id(labels, binary_mask, point_y, point_x, max_distance=max_snap_distance)
        if label_id is not None:
            selected_ids.add(label_id)

    if not selected_ids:
        return None, set()

    selected_mask = np.isin(labels, list(selected_ids))
    return selected_mask.astype(np.uint8), selected_ids


def select_component_by_overlap(
    mask: np.ndarray,
    reference_mask: np.ndarray | None,
) -> np.ndarray | None:
    binary_mask = np.asarray(mask, dtype=bool)
    binary_reference = np.asarray(reference_mask, dtype=bool) if reference_mask is not None else None
    if binary_reference is None or not np.any(binary_mask) or not np.any(binary_reference):
        return None

    labels, num_labels = ndimage.label(binary_mask)
    if num_labels == 0:
        return None

    best_component_id: int | None = None
    best_overlap = 0
    best_area = 0
    for component_id in range(1, num_labels + 1):
        component_mask = labels == component_id
        overlap = int(np.count_nonzero(np.logical_and(component_mask, binary_reference)))
        area = int(np.count_nonzero(component_mask))
        if overlap > best_overlap or (overlap == best_overlap and overlap > 0 and area > best_area):
            best_component_id = component_id
            best_overlap = overlap
            best_area = area

    if best_component_id is None or best_overlap <= 0:
        return None

    selected_mask = labels == best_component_id
    return selected_mask.astype(np.uint8) if np.any(selected_mask) else None


def count_points_matching_mask(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    *,
    max_snap_distance: int = POINT_COMPONENT_SNAP_DISTANCE,
) -> int:
    binary_mask = np.asarray(mask, dtype=bool)
    if not np.any(binary_mask) or not points:
        return 0

    labels, _ = ndimage.label(binary_mask)
    height, width = binary_mask.shape
    matches = 0
    for point in points:
        point_y, point_x = clamp_point(point, width, height)
        label_id = nearest_component_id(labels, binary_mask, point_y, point_x, max_distance=max_snap_distance)
        if label_id is not None:
            matches += 1
    return matches


def choose_best_point_mask_index(
    candidate_masks: list[np.ndarray],
    scores: np.ndarray,
    *,
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
    reference_mask: np.ndarray | None,
) -> int:
    if not candidate_masks:
        return 0

    binary_reference = None
    reference_area = 0
    if reference_mask is not None:
        binary_reference = np.asarray(reference_mask, dtype=bool)
        reference_area = int(np.count_nonzero(binary_reference))
        if reference_area <= 0:
            binary_reference = None

    total_positive = len(positive_points)
    best_index = 0
    best_rank: tuple[int, int, int, int, int, int, float, int] | None = None

    for index, candidate_mask in enumerate(candidate_masks):
        binary_mask = np.asarray(candidate_mask, dtype=bool)
        positive_hits = count_points_matching_mask(binary_mask, positive_points)
        negative_hits = count_points_matching_mask(binary_mask, negative_points)
        overlap = int(np.count_nonzero(np.logical_and(binary_mask, binary_reference))) if binary_reference is not None else 0
        area = int(np.count_nonzero(binary_mask))
        score = float(scores[index]) if index < len(scores) else 0.0

        rank = (
            int(total_positive == 0 or positive_hits == total_positive),
            positive_hits,
            int(negative_hits == 0),
            -negative_hits,
            int(binary_reference is None or overlap > 0),
            overlap,
            score,
            -abs(area - reference_area) if binary_reference is not None else area,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_index = index

    return best_index


def points_to_bbox(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    padding: int = POINT_BOX_PADDING,
) -> tuple[float, float, float, float] | None:
    if not points:
        return None

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_min = max(0.0, min(xs) - padding)
    y_min = max(0.0, min(ys) - padding)
    x_max = min(float(width - 1), max(xs) + padding)
    y_max = min(float(height - 1), max(ys) + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def bbox_to_mask(
    bbox: tuple[float, float, float, float] | None,
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox
    left = max(0, int(np.floor(x_min)))
    top = max(0, int(np.floor(y_min)))
    right = min(width - 1, int(np.ceil(x_max)))
    bottom = min(height - 1, int(np.ceil(y_max)))
    if right < left or bottom < top:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : bottom + 1, left : right + 1] = 1
    return mask


def subtract_point_disks(
    mask: np.ndarray,
    points: list[tuple[float, float]],
    *,
    radius: int = NEGATIVE_POINT_RADIUS,
) -> np.ndarray:
    binary_mask = np.asarray(mask, dtype=bool).copy()
    if not np.any(binary_mask) or not points:
        return binary_mask.astype(np.uint8)

    height, width = binary_mask.shape
    yy, xx = np.ogrid[:height, :width]
    for point in points:
        point_y, point_x = clamp_point(point, width, height)
        disk = (xx - point_x) ** 2 + (yy - point_y) ** 2 <= radius ** 2
        binary_mask[disk] = False

    return binary_mask.astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    *,
    min_area: int = MIN_MASK_COMPONENT_AREA,
) -> np.ndarray:
    binary_mask = np.asarray(mask, dtype=bool)
    if not np.any(binary_mask):
        return binary_mask.astype(np.uint8)

    labels, num_labels = ndimage.label(binary_mask)
    if num_labels == 0:
        return binary_mask.astype(np.uint8)

    component_areas = np.bincount(labels.ravel())
    keep_ids = [label_id for label_id in range(1, len(component_areas)) if component_areas[label_id] >= min_area]
    if not keep_ids:
        return binary_mask.astype(np.uint8)
    return np.isin(labels, keep_ids).astype(np.uint8)


def localize_prior_mask(
    mask: np.ndarray | None,
    focus_mask: np.ndarray | None,
    *,
    area_ratio: float = PRIOR_LOCALIZATION_AREA_RATIO,
    dilation_iterations: int = PRIOR_LOCALIZATION_DILATION,
) -> np.ndarray | None:
    if mask is None or focus_mask is None:
        return mask

    binary_mask = (np.asarray(mask, dtype=np.uint8) > 0)
    binary_focus = (np.asarray(focus_mask, dtype=np.uint8) > 0)
    if not np.any(binary_mask) or not np.any(binary_focus):
        return binary_mask.astype(np.uint8)

    mask_area = int(np.count_nonzero(binary_mask))
    focus_area = int(np.count_nonzero(binary_focus))
    if focus_area <= 0:
        return binary_mask.astype(np.uint8)

    if mask_area <= max(int(focus_area * area_ratio), focus_area + MIN_MASK_COMPONENT_AREA):
        return binary_mask.astype(np.uint8)

    localized_mask = np.logical_and(
        binary_mask,
        ndimage.binary_dilation(binary_focus, iterations=dilation_iterations),
    )
    if np.any(localized_mask):
        return localized_mask.astype(np.uint8)

    return binary_mask.astype(np.uint8)


def refine_mask_from_points(
    current_mask: np.ndarray | None,
    *,
    positive_points: list[tuple[float, float]],
    negative_points: list[tuple[float, float]],
) -> np.ndarray | None:
    if current_mask is None or not np.any(current_mask):
        return None

    refined_mask = (np.asarray(current_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if positive_points:
        selected_mask, _ = select_components_for_points(refined_mask, positive_points)
        if selected_mask is not None and np.any(selected_mask):
            refined_mask = selected_mask.astype(np.uint8)

    if negative_points and np.any(refined_mask):
        labels_before_negative, _ = ndimage.label(refined_mask > 0)
        _, positive_component_ids = select_components_for_points(refined_mask, positive_points)
        _, negative_component_ids = select_components_for_points(refined_mask, negative_points)
        refined_mask = subtract_point_disks(refined_mask, negative_points)
        removable_component_ids = negative_component_ids - positive_component_ids
        if positive_points and removable_component_ids:
            removable_mask = np.isin(labels_before_negative, list(removable_component_ids))
            refined_mask = np.logical_and(refined_mask > 0, ~removable_mask).astype(np.uint8)

    if not negative_points:
        refined_mask = ndimage.binary_fill_holes(refined_mask > 0).astype(np.uint8)
    else:
        refined_mask = (refined_mask > 0).astype(np.uint8)
    refined_mask = remove_small_components(refined_mask)
    return refined_mask if np.any(refined_mask) else None


@dataclass
class SessionEntry:
    session_id: str
    inference_state: dict[str, Any]
    width: int
    height: int
    checkpoint_path: str
    device: str
    confidence_threshold: float
    variant_key: str | None = None
    text_inference_state: dict[str, Any] | None = None
    logits_tokens: set[str] = field(default_factory=set)


@dataclass
class LogitsEntry:
    token: str
    session_id: str
    logits: np.ndarray


@dataclass
class VideoSessionEntry:
    """Tracks a SAM3.1 multiplex video session."""
    session_id: str
    predictor_session_id: str
    num_frames: int
    frame_index_map: dict[int, int]  # absolute CVAT frame → relative 0-based index
    width: int
    height: int
    device: str
    checkpoint_path: str


class SAM3ModelAdapter:
    def __init__(
        self,
        *,
        checkpoint_path: str | None,
        device: str,
        confidence_threshold: float,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor: Sam3Processor | None = None

    def _resolve_base_checkpoint(self) -> str | None:
        env_value = os.environ.get("SAM3_BASE_CHECKPOINT")
        if env_value:
            candidate = Path(env_value).resolve()
            if candidate.exists():
                return str(candidate)
        if DEFAULT_BASE_CHECKPOINT.exists():
            return str(DEFAULT_BASE_CHECKPOINT.resolve())
        return None

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        bpe_path = SAM3_CODE_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        base_checkpoint = self._resolve_base_checkpoint()

        # Build the model the same way the upstream Medical-SAM3 examples do,
        # then load checkpoints with the library's detector/tracker key layout.
        with self._autocast():
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                device=self.device,
                checkpoint_path=None,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )

            # Load base checkpoint first (if available), then overlay
            # fine-tuned weights on top.
            if self.checkpoint_path:
                if base_checkpoint is not None and Path(self.checkpoint_path).resolve() != Path(base_checkpoint).resolve():
                    self._load_custom_checkpoint(base_checkpoint)
                    self._load_custom_checkpoint(self.checkpoint_path)
                else:
                    self._load_custom_checkpoint(self.checkpoint_path)
            elif base_checkpoint is not None:
                self._load_custom_checkpoint(base_checkpoint)

        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def _load_custom_checkpoint(self, checkpoint_path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before loading a checkpoint")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        if any("detector" in key for key in state_dict.keys()):
            clean_state_dict = {
                key.replace("detector.", ""): value
                for key, value in state_dict.items()
                if "detector" in key
            }
        else:
            clean_state_dict = state_dict

        predictor = getattr(self.model, "inst_interactive_predictor", None)
        if predictor is not None:
            clean_state_dict.update(
                {
                    key.replace("tracker.", "inst_interactive_predictor.model."): value
                    for key, value in state_dict.items()
                    if "tracker" in key
                }
            )

        missing_keys, unexpected_keys = self.model.load_state_dict(clean_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"loaded {checkpoint_path} with "
                f"{len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys"
            )

    def unload(self) -> None:
        self.processor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold
        if self.processor is not None:
            self.processor.confidence_threshold = confidence_threshold

    def get_mask_input_size(self) -> tuple[int, int]:
        self.load_model()
        if self.model is None:
            raise RuntimeError("SAM3 model is not available")

        predictor = getattr(self.model, "inst_interactive_predictor", None)
        predictor_model = getattr(predictor, "model", None) if predictor is not None else None
        prompt_encoder = getattr(predictor_model, "sam_prompt_encoder", None)
        mask_input_size = getattr(prompt_encoder, "mask_input_size", None)
        if mask_input_size is None or len(mask_input_size) != 2:
            raise RuntimeError("SAM3 prompt encoder mask input size is not available")

        return int(mask_input_size[0]), int(mask_input_size[1])

    def encode_image(self, image: np.ndarray) -> dict[str, Any]:
        self.load_model()
        if self.processor is None:
            raise RuntimeError("SAM3 processor is not available")

        pil_image = Image.fromarray(image)
        with self._autocast():
            return self.processor.set_image(pil_image)

    def _merge_masks_from_state(
        self,
        state: dict[str, Any],
        *,
        score_threshold: float,
        max_masks: int,
    ) -> np.ndarray | None:
        masks = state.get("masks")
        if masks is None or len(masks) == 0:
            return None

        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]

        scores = state.get("scores")
        if scores is None or len(scores) == 0:
            indices = torch.arange(masks.shape[0])
        else:
            if not isinstance(scores, torch.Tensor):
                scores = torch.as_tensor(scores)
            scores = scores.view(-1)
            keep = scores >= score_threshold
            indices = torch.nonzero(keep).flatten() if keep.any() else torch.tensor([int(torch.argmax(scores))])
            if max_masks > 0 and indices.numel() > max_masks:
                _, top_indices = torch.topk(scores[indices], max_masks)
                indices = indices[top_indices]

        selected_masks = masks[indices]
        if selected_masks.ndim == 2:
            selected_masks = selected_masks.unsqueeze(0)

        union = (selected_masks > 0).any(dim=0)
        if not torch.any(union):
            return None
        return union.detach().cpu().numpy().astype(np.uint8)

    def get_confidence(self, state: dict[str, Any]) -> float:
        scores = state.get("scores")
        if scores is None or len(scores) == 0:
            return 0.0
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores)
        best_idx = int(torch.argmax(scores).item())
        return float(scores[best_idx].item())

    def predict_text_union(
        self,
        inference_state: dict[str, Any],
        *,
        text_prompt: str,
        score_threshold: float,
        max_masks: int,
    ) -> tuple[np.ndarray | None, float]:
        if self.processor is None:
            raise RuntimeError("SAM3 processor is not available")

        self.processor.reset_all_prompts(inference_state)
        with self._autocast():
            state = self.processor.set_text_prompt(prompt=text_prompt, state=inference_state)
        return self._merge_masks_from_state(state, score_threshold=score_threshold, max_masks=max_masks), self.get_confidence(state)

    def predict_box_with_text(
        self,
        inference_state: dict[str, Any],
        *,
        bbox: tuple[float, float, float, float],
        text_prompt: str,
        label: bool,
        score_threshold: float,
        max_masks: int,
    ) -> tuple[np.ndarray | None, float]:
        if self.processor is None:
            raise RuntimeError("SAM3 processor is not available")

        self.processor.reset_all_prompts(inference_state)
        state = inference_state
        with self._autocast():
            if text_prompt:
                state = self.processor.set_text_prompt(prompt=text_prompt, state=state)

            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            box_xywh = torch.tensor([x_min, y_min, width, height], dtype=torch.float32).view(1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_box = normalize_bbox(
                box_cxcywh,
                inference_state["original_width"],
                inference_state["original_height"],
            ).flatten().tolist()
            state = self.processor.add_geometric_prompt(state=state, box=norm_box, label=label)

        return self._merge_masks_from_state(state, score_threshold=score_threshold, max_masks=max_masks), self.get_confidence(state)

    def predict_box_logits(
        self,
        inference_state: dict[str, Any],
        *,
        bbox: tuple[float, float, float, float],
        reference_mask: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        if self.processor is None or self.model is None:
            raise RuntimeError("SAM3 model is not available")

        self.processor.reset_all_prompts(inference_state)
        box_array = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        with self._autocast():
            masks, scores, logits = self.model.predict_inst(
                inference_state,
                point_coords=None,
                point_labels=None,
                box=box_array,
                multimask_output=True,
            )

        if masks is None or len(masks) == 0 or logits is None:
            return None, 0.0, None

        scores_array = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        scores_array = np.asarray(scores_array, dtype=np.float32).reshape(-1)
        candidate_entries: list[tuple[int, np.ndarray]] = []
        for index, candidate in enumerate(masks):
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.detach().cpu().numpy()
            candidate_2d = squeeze_mask_to_2d(np.asarray(candidate))
            if candidate_2d is None:
                continue
            candidate_entries.append((index, (candidate_2d > 0).astype(np.uint8)))
        if not candidate_entries:
            return None, 0.0, None

        best_idx = choose_best_point_mask_index(
            [candidate_mask for _, candidate_mask in candidate_entries],
            scores_array,
            positive_points=[],
            negative_points=[],
            reference_mask=reference_mask,
        )
        selected_index, mask = candidate_entries[best_idx]
        selected_logits = logits[selected_index]
        if isinstance(selected_logits, torch.Tensor):
            selected_logits = selected_logits.detach().cpu().numpy()
        selected_logits = np.asarray(selected_logits, dtype=np.float32)
        if selected_logits.ndim == 2:
            selected_logits = selected_logits[None, ...]
        return np.asarray(mask, dtype=np.uint8), float(scores_array[selected_index]) if scores_array.size else 0.0, selected_logits

    def predict_points(
        self,
        inference_state: dict[str, Any],
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: np.ndarray | None,
        box: np.ndarray | None,
        reference_mask: np.ndarray | None,
        multimask_output: bool,
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        if self.processor is None or self.model is None:
            raise RuntimeError("SAM3 model is not available")

        self.processor.reset_all_prompts(inference_state)
        with self._autocast():
            masks, scores, logits = self.model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                box=box,
                multimask_output=multimask_output,
            )

        if masks is None or len(masks) == 0:
            return None, 0.0, None

        scores_array = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        scores_array = np.asarray(scores_array, dtype=np.float32).reshape(-1)
        candidate_entries: list[tuple[int, np.ndarray]] = []
        for index, candidate in enumerate(masks):
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.detach().cpu().numpy()
            candidate_2d = squeeze_mask_to_2d(np.asarray(candidate))
            if candidate_2d is None:
                continue
            candidate_entries.append((index, (candidate_2d > 0).astype(np.uint8)))
        if not candidate_entries:
            return None, 0.0, None

        positive_points = [
            (float(point[0]), float(point[1]))
            for point, label in zip(point_coords, point_labels)
            if int(label) == 1
        ]
        negative_points = [
            (float(point[0]), float(point[1]))
            for point, label in zip(point_coords, point_labels)
            if int(label) == 0
        ]
        best_idx = choose_best_point_mask_index(
            [candidate_mask for _, candidate_mask in candidate_entries],
            scores_array,
            positive_points=positive_points,
            negative_points=negative_points,
            reference_mask=reference_mask,
        )
        selected_index, mask = candidate_entries[best_idx]
        selected_logits = None
        if logits is not None:
            selected_logits = logits[selected_index]
            if isinstance(selected_logits, torch.Tensor):
                selected_logits = selected_logits.detach().cpu().numpy()
            selected_logits = np.asarray(selected_logits, dtype=np.float32)
            if selected_logits.ndim == 2:
                selected_logits = selected_logits[None, ...]
        return np.asarray(mask, dtype=np.uint8), float(scores_array[selected_index]) if scores_array.size else 0.0, selected_logits


class SAM3TextModelAdapter(SAM3ModelAdapter):
    def load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        if self.checkpoint_path is None:
            raise RuntimeError("A checkpoint path is required for the SAM3 text model")

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        bpe_path = SAM3_CODE_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        base_checkpoint = self._resolve_base_checkpoint()
        with self._autocast():
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                device=self.device,
                eval_mode=True,
                checkpoint_path=None,
                load_from_HF=False,
                enable_segmentation=True,
            )

            if base_checkpoint is not None and Path(self.checkpoint_path).resolve() != Path(base_checkpoint).resolve():
                self._load_custom_checkpoint(base_checkpoint)
                self._load_custom_checkpoint(self.checkpoint_path)
            else:
                self._load_custom_checkpoint(self.checkpoint_path)

        self.model.eval()
        self.processor = Sam3Processor(
            self.model,
            resolution=ATLAS_IMAGE_RESOLUTION,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )


class ModelManager:
    def __init__(self, *, adapter_cls: type[SAM3ModelAdapter] = SAM3ModelAdapter) -> None:
        self._lock = threading.RLock()
        self._adapter_cls = adapter_cls
        self._key: tuple[str | None, str] | None = None
        self._model: SAM3ModelAdapter | None = None

    def ensure_model(
        self,
        *,
        checkpoint_path: str | None,
        device: str,
        confidence_threshold: float,
    ) -> SAM3ModelAdapter:
        key = (checkpoint_path, device)
        with self._lock:
            if self._key != key:
                if self._model is not None:
                    self._model.unload()
                self._model = self._adapter_cls(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )
                self._model.load_model()
                self._key = key
            elif self._model is None:
                self._model = self._adapter_cls(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )
                self._model.load_model()
                self._key = key
            else:
                self._model.set_confidence_threshold(confidence_threshold)
            return self._model

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def loaded_model(self) -> dict[str, Any] | None:
        if self._key is None:
            return None
        checkpoint_path, device = self._key
        return {
            "checkpoint_path": checkpoint_path,
            "device": device,
        }


model_manager = ModelManager()
atlas_text_model_manager = ModelManager(adapter_cls=SAM3TextModelAdapter)
session_store: LruTtlStore[SessionEntry] = LruTtlStore(maxsize=MAX_SESSIONS, ttl_seconds=SESSION_TTL_SECONDS)
logits_store: LruTtlStore[LogitsEntry] = LruTtlStore(maxsize=MAX_LOGITS, ttl_seconds=SESSION_TTL_SECONDS)


class VideoModelManager:
    """Manages a single SAM3.1 multiplex video predictor instance."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._predictor = None
        self._key: tuple[str, str] | None = None

    def ensure_predictor(self, *, device: str, checkpoint_path: str):
        with self._lock:
            key = (checkpoint_path, device)
            if self._predictor is not None and self._key == key:
                return self._predictor

            if self._predictor is not None:
                self._predictor.shutdown()
                self._predictor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            bpe_path = SAM3_CODE_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            self._predictor = build_sam3_predictor(
                checkpoint_path=checkpoint_path,
                version="sam3.1",
                bpe_path=str(bpe_path),
                compile=False,
                warm_up=False,
                use_fa3=False,
                async_loading_frames=True,
            )
            configure_video_predictor_for_interactivity(self._predictor)
            self._key = key
            return self._predictor

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def key(self) -> tuple[str, str] | None:
        return self._key


video_model_manager = VideoModelManager()
video_session_store: LruTtlStore[VideoSessionEntry] = LruTtlStore(
    maxsize=MAX_VIDEO_SESSIONS, ttl_seconds=VIDEO_SESSION_TTL_SECONDS,
)


def configure_video_predictor_for_interactivity(predictor: Any) -> None:
    """Disable delayed video heuristics that hide early interactive results."""
    model = getattr(predictor, "model", None)
    if model is None:
        return

    # CVAT propagation is an interactive labeling workflow, not an offline
    # tracking benchmark. The upstream multiplex defaults buffer the first 15
    # frames and periodically recondition tracks, which makes short sessions
    # look broken because no masks appear in the immediately following frames.
    model.hotstart_delay = 0
    model.hotstart_unmatch_thresh = 0
    model.hotstart_dup_thresh = 0
    model.recondition_every_nth_frame = 0
    model.masklet_confirmation_enable = False
    model.use_iom_recondition = False


@contextmanager
def locked_video_predictor(device: str, checkpoint_path: str):
    lock = video_model_manager.lock
    acquired = lock.acquire(timeout=VIDEO_MODEL_LOCK_TIMEOUT_SECONDS)
    if not acquired:
        raise HTTPException(status_code=503, detail="SAM3.1 video predictor is busy")

    try:
        yield video_model_manager.ensure_predictor(
            device=device,
            checkpoint_path=checkpoint_path,
        )
    finally:
        lock.release()


def purge_session_logits(session: SessionEntry) -> None:
    logits_store.delete_many(session.logits_tokens)


def register_session(entry: SessionEntry) -> None:
    expired, evicted = session_store.set(entry.session_id, entry)
    for session in [*expired, *evicted]:
        purge_session_logits(session)


def register_logits(entry: LogitsEntry) -> None:
    expired, evicted = logits_store.set(entry.token, entry)
    for value in [*expired, *evicted]:
        session = session_store.get(value.session_id)
        if session is not None:
            session.logits_tokens.discard(value.token)


def get_session(session_id: str) -> SessionEntry:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown or expired session_id '{session_id}'")
    return session


def get_logits(token: str, session_id: str) -> np.ndarray:
    entry = logits_store.get(token)
    if entry is None or entry.session_id != session_id:
        raise HTTPException(status_code=400, detail="Invalid logits_token for this session")
    return entry.logits


def session_uses_atlas_text_path(session: SessionEntry) -> bool:
    return (
        normalize_variant_key(session.variant_key) == ATLAS_VARIANT_KEY
        and session.text_inference_state is not None
    )


@contextmanager
def locked_promptable_model(session: SessionEntry):
    use_atlas_text_model = session_uses_atlas_text_path(session)
    manager = atlas_text_model_manager if use_atlas_text_model else model_manager
    inference_state = session.text_inference_state if use_atlas_text_model else session.inference_state
    if inference_state is None:
        raise HTTPException(status_code=500, detail="SAM3 promptable inference state is unavailable")

    with manager.lock:
        model = manager.ensure_model(
            checkpoint_path=session.checkpoint_path,
            device=session.device,
            confidence_threshold=session.confidence_threshold,
        )
        yield model, inference_state


class TextInferRequest(BaseModel):
    session_id: str
    labels: list[str] = Field(default_factory=list)
    score_threshold: float = 0.3
    max_masks: int = 0


class BoxInferRequest(BaseModel):
    session_id: str
    label_name: str
    bbox: tuple[float, float, float, float]
    negative: bool = False
    score_threshold: float = 0.3
    max_masks: int = 0


class PointPayload(BaseModel):
    x: float
    y: float
    label: int


class PointsInferRequest(BaseModel):
    session_id: str
    points: list[PointPayload] = Field(default_factory=list)
    logits_token: str | None = None
    initial_mask_rle: list[float] | None = None
    multimask_output: bool = True
    prompt: str | None = None
    score_threshold: float = 0.3


app = FastAPI(title="SAM3 Sidecar", version="0.1.0")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "request_failed", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": exc.__class__.__name__, "detail": str(exc)},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "devices": available_devices(),
        "loaded_model": model_manager.loaded_model(),
    }


@app.get("/checkpoints")
def checkpoints() -> list[dict[str, str]]:
    return list_checkpoints()


@app.post("/sessions")
async def create_session(
    image: UploadFile = File(...),
    checkpoint_path: str = Form(...),
    device: str = Form(...),
    confidence_threshold: float = Form(0.5),
    variant_key: str | None = Form(None),
) -> dict[str, Any]:
    device = normalize_device(device)
    checkpoint_path = validate_checkpoint_path(checkpoint_path)
    variant_key = normalize_variant_key(variant_key)

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Image payload is empty")
    frame_rgb = decode_image_upload(payload)

    with model_manager.lock:
        model = model_manager.ensure_model(
            checkpoint_path=checkpoint_path,
            device=device,
            confidence_threshold=float(confidence_threshold),
        )
        inference_state = model.encode_image(frame_rgb)

    text_inference_state: dict[str, Any] | None = None
    if variant_key == ATLAS_VARIANT_KEY:
        with atlas_text_model_manager.lock:
            text_model = atlas_text_model_manager.ensure_model(
                checkpoint_path=checkpoint_path,
                device=device,
                confidence_threshold=float(confidence_threshold),
            )
            text_inference_state = text_model.encode_image(frame_rgb)

    session = SessionEntry(
        session_id=str(uuid.uuid4()),
        inference_state=inference_state,
        width=int(frame_rgb.shape[1]),
        height=int(frame_rgb.shape[0]),
        checkpoint_path=checkpoint_path,
        device=device,
        confidence_threshold=float(confidence_threshold),
        variant_key=variant_key,
        text_inference_state=text_inference_state,
    )
    register_session(session)

    return {
        "session_id": session.session_id,
        "width": session.width,
        "height": session.height,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "variant_key": variant_key,
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    session = session_store.pop(session_id)
    if session is not None:
        purge_session_logits(session)
    return {"status": "deleted"}


@app.post("/infer/text")
def infer_text(request: TextInferRequest) -> list[dict[str, Any]]:
    session = get_session(request.session_id)
    if not request.labels:
        return []

    with locked_promptable_model(session) as (model, inference_state):
        results: list[dict[str, Any]] = []
        for label in request.labels:
            mask, confidence = model.predict_text_union(
                inference_state,
                text_prompt=label,
                score_threshold=float(request.score_threshold),
                max_masks=int(request.max_masks),
            )
            if mask is None or not np.any(mask):
                continue
            results.append(
                {
                    "label_name": label,
                    "confidence": confidence,
                    "mask_png_b64": encode_mask_png(mask),
                }
            )
        return results


@app.post("/infer/box")
def infer_box(request: BoxInferRequest) -> dict[str, Any]:
    session = get_session(request.session_id)

    with locked_promptable_model(session) as (model, inference_state):
        mask, confidence = model.predict_box_with_text(
            inference_state,
            bbox=request.bbox,
            text_prompt=request.label_name,
            label=not request.negative,
            score_threshold=float(request.score_threshold),
            max_masks=int(request.max_masks),
        )

    if mask is None or not np.any(mask):
        return {"mask_png_b64": None}

    return {
        "confidence": confidence,
        "mask_png_b64": encode_mask_png(mask),
    }


@app.post("/infer/points")
def infer_points(request: PointsInferRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    if not request.points:
        raise HTTPException(status_code=400, detail="At least one point is required")

    point_coords = np.asarray([[point.x, point.y] for point in request.points], dtype=np.float32)
    point_labels = np.asarray([point.label for point in request.points], dtype=np.int64)
    prompt = str(request.prompt or "").strip()
    previous_mask = get_logits(request.logits_token, session.session_id) if request.logits_token else None
    if previous_mask is None and request.initial_mask_rle:
        previous_mask = decode_cvat_rle_to_full_mask(
            request.initial_mask_rle,
            session.width,
            session.height,
        )
    positive_points = [
        (float(point.x), float(point.y))
        for point in request.points
        if int(point.label) == 1
    ]
    negative_points = [
        (float(point.x), float(point.y))
        for point in request.points
        if int(point.label) == 0
    ]

    with model_manager.lock:
        model = model_manager.ensure_model(
            checkpoint_path=session.checkpoint_path,
            device=session.device,
            confidence_threshold=session.confidence_threshold,
        )

        if prompt and previous_mask is None:
            semantic_mask, semantic_confidence = model.predict_text_union(
                session.inference_state,
                text_prompt=prompt,
                score_threshold=float(request.score_threshold),
                max_masks=0,
            )
            semantic_mask = semantic_mask.astype(np.uint8) if semantic_mask is not None and np.any(semantic_mask) else None

            current_mask = (np.asarray(previous_mask, dtype=np.uint8) > 0).astype(np.uint8) if previous_mask is not None else None
            positive_focus_bbox = points_to_bbox(
                positive_points,
                width=session.width,
                height=session.height,
                padding=POINT_FOCUS_PADDING,
            )
            positive_focus_mask = bbox_to_mask(
                positive_focus_bbox,
                width=session.width,
                height=session.height,
            )
            current_mask = localize_prior_mask(current_mask, positive_focus_mask)
            positive_bbox = positive_focus_bbox or points_to_bbox(
                positive_points,
                width=session.width,
                height=session.height,
            )
            box_mask = None
            box_confidence = 0.0
            if positive_bbox is not None:
                box_mask, box_confidence = model.predict_box_with_text(
                    session.inference_state,
                    bbox=positive_bbox,
                    text_prompt=prompt,
                    label=True,
                    score_threshold=float(request.score_threshold),
                    max_masks=0,
                )
                if box_mask is not None and np.any(box_mask):
                    box_mask = box_mask.astype(np.uint8)
                    if positive_focus_mask is not None:
                        focus_dilated = ndimage.binary_dilation(positive_focus_mask > 0, iterations=8)
                        box_mask = np.logical_and(box_mask > 0, focus_dilated).astype(np.uint8)
                    if not np.any(box_mask):
                        box_mask = None
                else:
                    box_mask = None

            if semantic_mask is not None:
                focused_semantic_mask = semantic_mask
                if positive_focus_mask is not None:
                    focused_semantic_mask = np.logical_and(
                        semantic_mask > 0,
                        positive_focus_mask > 0,
                    ).astype(np.uint8)

                selected_semantic_mask, _ = select_components_for_points(
                    focused_semantic_mask,
                    positive_points,
                )
                if current_mask is None:
                    if selected_semantic_mask is not None and np.any(selected_semantic_mask):
                        current_mask = selected_semantic_mask.astype(np.uint8)
                    elif np.any(focused_semantic_mask):
                        current_mask = focused_semantic_mask.copy()
                elif selected_semantic_mask is not None and np.any(selected_semantic_mask):
                    current_mask = np.logical_or(current_mask > 0, selected_semantic_mask > 0).astype(np.uint8)

                if box_mask is not None:
                    constrained_box_mask = box_mask
                    semantic_dilated = ndimage.binary_dilation(focused_semantic_mask > 0, iterations=16)
                    constrained_box_mask = np.logical_and(constrained_box_mask > 0, semantic_dilated).astype(np.uint8)
                    if np.any(constrained_box_mask):
                        if current_mask is None:
                            current_mask = constrained_box_mask
                        else:
                            current_mask = np.logical_or(current_mask > 0, constrained_box_mask > 0).astype(np.uint8)

                if negative_points and current_mask is not None:
                    labels_before_negative, _ = ndimage.label(current_mask > 0)
                    _, positive_current_component_ids = select_components_for_points(
                        current_mask,
                        positive_points,
                    )
                    _, negative_component_ids = select_components_for_points(
                        current_mask,
                        negative_points,
                    )
                    current_mask = subtract_point_disks(current_mask, negative_points)
                    removable_component_ids = negative_component_ids - positive_current_component_ids
                    if positive_points and removable_component_ids:
                        removable_mask = np.isin(labels_before_negative, list(removable_component_ids))
                        current_mask = np.logical_and(current_mask > 0, ~removable_mask).astype(np.uint8)

                if current_mask is not None and positive_focus_mask is not None:
                    current_mask = localize_prior_mask(current_mask, positive_focus_mask)

                if current_mask is None:
                    return {"confidence": semantic_confidence, "mask_png_b64": None, "logits_token": None}

                if not negative_points:
                    current_mask = ndimage.binary_fill_holes(current_mask > 0).astype(np.uint8)
                else:
                    current_mask = (current_mask > 0).astype(np.uint8)
                current_mask = remove_small_components(current_mask)

                if not np.any(current_mask):
                    return {"confidence": semantic_confidence, "mask_png_b64": None, "logits_token": None}

                return {
                    "confidence": semantic_confidence,
                    "mask_png_b64": encode_mask_png(current_mask),
                    "logits_token": None,
                }

            if box_mask is not None:
                current_mask = box_mask if current_mask is None else np.logical_or(current_mask > 0, box_mask > 0).astype(np.uint8)
                if negative_points and current_mask is not None:
                    current_mask = subtract_point_disks(current_mask, negative_points)
                if current_mask is not None and positive_focus_mask is not None:
                    current_mask = localize_prior_mask(current_mask, positive_focus_mask)
                if not negative_points:
                    current_mask = ndimage.binary_fill_holes(current_mask > 0).astype(np.uint8)
                else:
                    current_mask = (current_mask > 0).astype(np.uint8)
                current_mask = remove_small_components(current_mask)
                if np.any(current_mask):
                    return {
                        "confidence": box_confidence,
                        "mask_png_b64": encode_mask_png(current_mask),
                        "logits_token": None,
                    }

            fallback_mask = refine_mask_from_points(
                current_mask,
                positive_points=positive_points,
                negative_points=negative_points,
            )
            if fallback_mask is not None:
                if positive_focus_mask is not None:
                    fallback_mask = localize_prior_mask(fallback_mask, positive_focus_mask)
                if fallback_mask is None or not np.any(fallback_mask):
                    return {"confidence": max(semantic_confidence, box_confidence), "mask_png_b64": None, "logits_token": None}
                return {
                    "confidence": max(semantic_confidence, box_confidence),
                    "mask_png_b64": encode_mask_png(fallback_mask),
                    "logits_token": None,
                }

            return {"confidence": max(semantic_confidence, box_confidence), "mask_png_b64": None, "logits_token": None}

        mask_input = None
        bootstrap_box = None
        prior_focus_mask = None
        if previous_mask is not None:
            mask_input_size = model.get_mask_input_size()
            previous_mask_2d = squeeze_mask_to_2d(previous_mask)
            if previous_mask_2d is not None:
                if previous_mask_2d.shape == mask_input_size and not is_binary_like_mask(previous_mask_2d):
                    mask_input = previous_mask_2d[None, ...].astype(np.float32, copy=False)
                else:
                    # Only true low-res SAM logits are valid mask_input prompts.
                    # A binary CVAT mask is still useful as a spatial prior, but
                    # should only contribute a box on the first refinement click.
                    bootstrap_mask = previous_mask_2d
                    if positive_points and previous_mask_2d.shape == (session.height, session.width):
                        selected_bootstrap_mask = select_bootstrap_mask(
                            previous_mask_2d,
                            point_coords=point_coords,
                            point_labels=point_labels,
                        )
                        if selected_bootstrap_mask is not None and np.any(selected_bootstrap_mask):
                            bootstrap_mask = selected_bootstrap_mask
                    if bootstrap_mask.shape == (session.height, session.width):
                        prior_focus_mask = (np.asarray(bootstrap_mask, dtype=np.uint8) > 0).astype(np.uint8)
                        bootstrap_bbox = generate_bbox_from_mask(prior_focus_mask)
                        if bootstrap_bbox is not None:
                            bootstrap_box = np.asarray(bootstrap_bbox, dtype=np.float32)
                            _, _, bootstrap_logits = model.predict_box_logits(
                                session.inference_state,
                                bbox=bootstrap_bbox,
                                reference_mask=prior_focus_mask,
                            )
                            if bootstrap_logits is not None:
                                mask_input = np.asarray(bootstrap_logits, dtype=np.float32, copy=False)
                                bootstrap_box = None

            if mask_input is None and bootstrap_box is None:
                previous_mask = None

        multimask_output = bool(request.multimask_output) and mask_input is None and len(request.points) <= 1
        mask, confidence, logits = model.predict_points(
            session.inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            box=bootstrap_box,
            reference_mask=prior_focus_mask,
            multimask_output=multimask_output,
        )

    if mask is None or logits is None or not np.any(mask):
        return {"confidence": confidence, "mask_png_b64": None, "logits_token": None}

    raw_output_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    output_mask = raw_output_mask.copy()
    if positive_points:
        selected_mask, _ = select_components_for_points(output_mask, positive_points)
        if selected_mask is not None and np.any(selected_mask):
            output_mask = selected_mask.astype(np.uint8)
    elif prior_focus_mask is not None:
        overlapped_mask = select_component_by_overlap(output_mask, prior_focus_mask)
        if overlapped_mask is not None and np.any(overlapped_mask):
            output_mask = overlapped_mask.astype(np.uint8)

    if prior_focus_mask is not None:
        localized_mask = localize_prior_mask(output_mask, prior_focus_mask)
        if localized_mask is not None and np.any(localized_mask):
            output_mask = localized_mask.astype(np.uint8)

    if not negative_points:
        output_mask = ndimage.binary_fill_holes(output_mask > 0).astype(np.uint8)
    else:
        output_mask = (output_mask > 0).astype(np.uint8)
    cleaned_mask = remove_small_components(output_mask)
    if np.any(cleaned_mask):
        output_mask = cleaned_mask.astype(np.uint8)

    token: str | None = None
    # SAM mask logits are only reusable when they still correspond to the
    # exact mask we returned. If we localized or filtered the mask, feeding the
    # original logits back into the next click can re-expand to unrelated
    # regions on the next refinement step.
    if np.array_equal(output_mask, raw_output_mask):
        token = str(uuid.uuid4())
        session.logits_tokens.add(token)
        register_logits(
            LogitsEntry(
                token=token,
                session_id=session.session_id,
                logits=np.asarray(logits, dtype=np.float32),
            )
        )

    return {
        "confidence": confidence,
        "mask_png_b64": encode_mask_png(output_mask),
        "logits_token": token,
    }


# ── Video tracking (SAM 3.1 multiplex) ──────────────────────────────


def get_video_session(session_id: str) -> VideoSessionEntry:
    session = video_session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown or expired video session '{session_id}'")
    return session


def _encode_video_masks(outputs: dict) -> list[dict[str, Any]]:
    """Convert propagation outputs to serializable mask list."""
    masks = outputs.get("out_binary_masks")
    obj_ids = outputs.get("out_obj_ids")
    if masks is None or obj_ids is None:
        return []

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.cpu().numpy()

    results = []
    for i, obj_id in enumerate(obj_ids):
        mask_2d = masks[i] if masks.ndim == 3 else masks
        if not np.any(mask_2d):
            continue
        results.append({
            "obj_id": int(obj_id),
            "mask_png_b64": encode_mask_png(mask_2d.astype(np.uint8)),
        })
    return results


class VideoSessionCreateRequest(BaseModel):
    device: str = "cuda"


class VideoPromptRequest(BaseModel):
    session_id: str
    frame_index: int
    text: str | None = None
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    bounding_boxes: list[list[float]] | None = None
    bounding_box_labels: list[int] | None = None
    obj_id: int | None = None
    output_prob_thresh: float = 0.5


class VideoPropagateRequest(BaseModel):
    session_id: str
    direction: str = "both"
    start_frame_index: int | None = None
    max_frames: int | None = None
    output_prob_thresh: float = 0.5


class VideoRemoveObjectRequest(BaseModel):
    session_id: str
    obj_id: int
    frame_index: int = 0


@app.post("/sessions/video")
async def create_video_session(
    images: list[UploadFile] = File(...),
    frame_indices: str = Form(...),
    device: str = Form("cuda"),
    checkpoint_path: str = Form(...),
) -> dict[str, Any]:
    """Create a video tracking session from uploaded frame images.

    ``frame_indices`` is a JSON array of absolute CVAT frame numbers,
    one per uploaded image, in the same order.
    """
    device = normalize_device(device)
    checkpoint_path = validate_checkpoint_path(checkpoint_path)
    indices: list[int] = json.loads(frame_indices)
    if len(indices) != len(images):
        raise HTTPException(
            status_code=400,
            detail=f"frame_indices length ({len(indices)}) must match images ({len(images)})",
        )
    if not indices:
        raise HTTPException(status_code=400, detail="At least one frame is required")

    # Decode uploaded images as PIL
    pil_frames: list[Image.Image] = []
    width = height = 0
    for upload in images:
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty image payload in batch")
        pil_img = Image.open(io.BytesIO(payload)).convert("RGB")
        width, height = pil_img.size
        pil_frames.append(pil_img)

    # Build frame index map: absolute → relative (0-based position)
    frame_index_map = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(indices)}

    # Predictor session state lives inside the predictor instance, so switching
    # checkpoints invalidates existing video sessions.
    if video_model_manager.key not in (None, (checkpoint_path, device)):
        video_session_store.clear()

    with locked_video_predictor(device, checkpoint_path) as predictor:
        with video_autocast(device):
            result = predictor.handle_request({
                "type": "start_session",
                "resource_path": pil_frames,
                "offload_video_to_cpu": True,
            })

    session_id = str(uuid.uuid4())
    entry = VideoSessionEntry(
        session_id=session_id,
        predictor_session_id=result["session_id"],
        num_frames=len(pil_frames),
        frame_index_map=frame_index_map,
        width=width,
        height=height,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    video_session_store.set(session_id, entry)

    return {
        "session_id": session_id,
        "num_frames": entry.num_frames,
        "frame_index_map": frame_index_map,
        "width": width,
        "height": height,
    }


@app.delete("/sessions/video/{session_id}")
def delete_video_session(session_id: str) -> dict[str, str]:
    entry = video_session_store.pop(session_id)
    if entry is not None:
        with locked_video_predictor(entry.device, entry.checkpoint_path) as predictor:
            with video_autocast(entry.device):
                predictor.handle_request({
                    "type": "close_session",
                    "session_id": entry.predictor_session_id,
                })
    return {"status": "deleted"}


@app.post("/video/prompt")
def video_add_prompt(request: VideoPromptRequest) -> dict[str, Any]:
    """Add a prompt (text, points, or boxes) to a specific frame."""
    session = get_video_session(request.session_id)

    # Map absolute CVAT frame index to relative index
    rel_frame = session.frame_index_map.get(request.frame_index)
    if rel_frame is None:
        raise HTTPException(
            status_code=400,
            detail=f"Frame {request.frame_index} not in this video session",
        )

    req = {
        "type": "add_prompt",
        "session_id": session.predictor_session_id,
        "frame_index": rel_frame,
        "output_prob_thresh": request.output_prob_thresh,
    }
    if request.text is not None:
        req["text"] = request.text
    if request.points is not None:
        req["points"] = request.points
    if request.point_labels is not None:
        req["point_labels"] = request.point_labels
    if request.bounding_boxes is not None:
        req["bounding_boxes"] = request.bounding_boxes
    if request.bounding_box_labels is not None:
        req["bounding_box_labels"] = request.bounding_box_labels
    if request.obj_id is not None:
        req["obj_id"] = request.obj_id

    with locked_video_predictor(session.device, session.checkpoint_path) as predictor:
        with video_autocast(session.device):
            result = predictor.handle_request(req)

    # Convert outputs
    outputs = result.get("outputs", {})
    return {
        "frame_index": request.frame_index,
        "masks": _encode_video_masks(outputs),
    }


@app.post("/video/propagate")
def video_propagate(request: VideoPropagateRequest) -> StreamingResponse:
    """Propagate prompts across video frames. Returns NDJSON stream."""
    session = get_video_session(request.session_id)

    # Map absolute start frame to relative if provided
    rel_start = None
    if request.start_frame_index is not None:
        rel_start = session.frame_index_map.get(request.start_frame_index)
        if rel_start is None:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {request.start_frame_index} not in this video session",
            )

    # Build reverse map: relative → absolute
    reverse_map = {v: k for k, v in session.frame_index_map.items()}

    def generate():
        with locked_video_predictor(session.device, session.checkpoint_path) as predictor:
            with video_autocast(session.device):
                stream_req = {
                    "type": "propagate_in_video",
                    "session_id": session.predictor_session_id,
                    "propagation_direction": request.direction,
                    "output_prob_thresh": request.output_prob_thresh,
                }
                if rel_start is not None:
                    stream_req["start_frame_index"] = rel_start
                if request.max_frames is not None:
                    stream_req["max_frame_num_to_track"] = request.max_frames

                for output in predictor.handle_stream_request(stream_req):
                    rel_idx = output["frame_index"]
                    abs_idx = reverse_map.get(rel_idx, rel_idx)
                    masks = _encode_video_masks(output.get("outputs", {}))
                    yield json.dumps({
                        "frame_index": abs_idx,
                        "masks": masks,
                    }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/video/remove-object")
def video_remove_object(request: VideoRemoveObjectRequest) -> dict[str, Any]:
    session = get_video_session(request.session_id)

    rel_frame = session.frame_index_map.get(request.frame_index)
    if rel_frame is None:
        raise HTTPException(
            status_code=400,
            detail=f"Frame {request.frame_index} not in this video session",
        )

    with locked_video_predictor(session.device, session.checkpoint_path) as predictor:
        result = predictor.handle_request({
            "type": "remove_object",
            "session_id": session.predictor_session_id,
            "frame_index": rel_frame,
            "obj_id": request.obj_id,
        })

    outputs = result.get("outputs", {})
    return {
        "frame_index": request.frame_index,
        "masks": _encode_video_masks(outputs),
    }


@app.post("/video/reset")
def video_reset_session(request: dict) -> dict[str, Any]:
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    session = get_video_session(session_id)

    with locked_video_predictor(session.device, session.checkpoint_path) as predictor:
        predictor.handle_request({
            "type": "reset_session",
            "session_id": session.predictor_session_id,
        })

    return {"status": "reset"}
