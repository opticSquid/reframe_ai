"""Subject-aware image reformatting module.

Pipeline:
  1. PLAN: detect subjects (MediaPipe BlazeFace) -> compute optimal crop
     coordinates for a target aspect ratio.  Pure planning, no image I/O
     beyond what MediaPipe needs for inference.

  2. RENDER: apply crop coordinates to the source image and return the
     rendered crop.  Pure pixel-level operation — no detection or
     decision-making.

This separation lets you:
  - Plan a crop once, render it many times (e.g. different output formats)
  - Re-render with modified crop boxes without re-detecting
  - Test planning and rendering independently
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

from .config import FACE_DETECTION_MODEL, TARGET_ASPECT_RATIOS
from .cropper import Subject, CropResult, compute_crop


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class CropPlan:
    """Full plan: where to crop and why."""

    image_path: str
    image_width: int
    image_height: int
    target_ratio: float
    crop: CropResult
    subjects: list[Subject]
    model_used: str = ""
    detection_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        """Serializable metadata for manifests / reports."""
        return {
            "image_path": self.image_path,
            "image_size": {"width": self.image_width, "height": self.image_height},
            "target_ratio": round(self.target_ratio, 6),
            "model_used": self.model_used,
            "subjects_detected": self.detection_count,
            "subjects": [
                {"bbox": s.bbox, "importance": s.importance,
                 "category": s.category, "id": s.id}
                for s in self.subjects
            ],
            "crop": {
                "x": self.crop.x, "y": self.crop.y,
                "width": self.crop.width, "height": self.crop.height,
                "subject_coverage_pct": round(self.crop.subject_coverage_pct, 4),
                "edge_cutoffs": self.crop.edge_cutoffs,
                "primary_subject_center": self.crop.primary_subject_center,
                "scale_used": round(self.crop.scale_used, 4),
                "is_center_fallback": self.crop.is_center_fallback,
            },
        }


# ---------------------------------------------------------------------------
# Aspect-ratio utilities
# ---------------------------------------------------------------------------
def compute_target_dimensions(
    source_w: int, source_h: int, target_ratio: float
) -> tuple[int, int]:
    """Compute (target_w, target_h) for a source image at a given aspect ratio.

    The result is the largest rectangle that fits within the source while
    maintaining the exact target ratio.
    """
    if target_ratio >= 1.0:
        # Landscape: width >= height
        crop_w = source_w
        crop_h = int(round(crop_w / target_ratio))
        if crop_h > source_h:
            crop_h = source_h
            crop_w = int(round(crop_h * target_ratio))
    else:
        # Portrait: height > width
        crop_h = source_h
        crop_w = int(round(crop_h * target_ratio))
        if crop_w > source_w:
            crop_w = source_w
            crop_h = int(round(crop_w / target_ratio))
    return crop_w, crop_h


# ---------------------------------------------------------------------------
# Subject detection (MediaPipe)
# ---------------------------------------------------------------------------
def detect_subjects(
    image_path: str | Path,
    model_path: str | Path = FACE_DETECTION_MODEL,
    min_confidence: float = 0.3,
) -> tuple[list[Subject], str]:
    """Run MediaPipe face detection on an image.

    Returns (subjects, model_name).  If the model file is missing or
    detection fails, returns an empty list so callers can fall back
    gracefully.
    """
    model_path = Path(model_path)
    model_name = model_path.name

    if not model_path.exists():
        warnings.warn(f"Model not found: {model_path}")
        return [], model_name

    img = Image.open(image_path)
    w, h = img.size
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)

    # MediaPipe expects SRGB data
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(img_array),
    )

    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_confidence,
        min_suppression_threshold=0.3,
    )

    subjects: list[Subject] = []
    try:
        with mp_vision.FaceDetector.create_from_options(opts) as detector:
            results = detector.detect(mp_image)
        if results.detections:
            for det in results.detections:
                bbox = det.bounding_box
                x1 = int(bbox.origin_x)
                y1 = int(bbox.origin_y)
                bw = int(bbox.width)
                bh = int(bbox.height)
                score = det.categories[0].score if det.categories else 0.5
                subjects.append(Subject(
                    bbox=(float(x1), float(y1), float(bw), float(bh)),
                    importance=float(score),
                    id=f"face_{len(subjects)}",
                    category="face",
                ))
    except Exception:
        warnings.warn("Face detection failed — returning empty subject list")
        return [], model_name

    return subjects, model_name


# ---------------------------------------------------------------------------
# Crop planning  (detection + arithmetic, no rendering)
# ---------------------------------------------------------------------------
def plan_crop(
    image_path: str | Path,
    target_ratio: float,
    target_coverage: float = 0.5,
    min_margin: float = 0.1,
    min_confidence: float = 0.3,
) -> CropPlan:
    """Plan a subject-aware crop: detect subjects then compute crop coordinates.

    Does NOT read or write any image pixels beyond what MediaPipe needs
    for inference.  Use :func:`render_crop` to apply the result.
    """
    image_path = Path(image_path)
    img = Image.open(image_path)
    w, h = img.size

    subjects, model_name = detect_subjects(
        image_path, min_confidence=min_confidence
    )

    crop_result = compute_crop(
        source_w=w,
        source_h=h,
        subjects=subjects,
        target_ratio=target_ratio,
        target_coverage=target_coverage,
        min_margin=min_margin,
    )

    warnings_list: list[str] = []
    if not subjects:
        warnings_list.append("No subjects detected — using center crop fallback")
    if crop_result.is_center_fallback:
        warnings_list.append("Crop fell back to center (no usable subjects)")

    return CropPlan(
        image_path=str(image_path),
        image_width=w,
        image_height=h,
        target_ratio=target_ratio,
        crop=crop_result,
        subjects=subjects,
        model_used=model_name,
        detection_count=len(subjects),
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Crop rendering  (pure pixel-level operation, no detection)
# ---------------------------------------------------------------------------
def render_crop(
    image_path: str | Path,
    crop: CropResult,
) -> np.ndarray:
    """Apply a crop rectangle to an image and return the cropped pixels.

    This is a pure rendering step — it reads the image, slices the crop
    rectangle, and returns the result.  No detection or planning occurs.
    """
    img = Image.open(image_path)
    arr = np.array(img)
    x, y, w, h = crop.x, crop.y, crop.width, crop.height

    # Safety clamp
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.width - x)
    h = min(h, img.height - y)

    return arr[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# Convenience: end-to-end single call
# ---------------------------------------------------------------------------
def process_image(
    image_path: str | Path,
    target_ratio: float,
    **kwargs,
) -> tuple[CropResult, np.ndarray, dict]:
    """Plan + render in one call.

    Returns (crop_result, rendered_image_array, metadata_dict).
    """
    plan = plan_crop(image_path, target_ratio, **kwargs)
    rendered = render_crop(image_path, plan.crop)
    return plan.crop, rendered, plan.to_metadata()


# ---------------------------------------------------------------------------
# Convenience: all standard aspect ratios
# ---------------------------------------------------------------------------
def process_all_ratios(
    image_path: str | Path,
) -> dict[str, CropPlan]:
    """Plan crops for all standard image aspect ratios (16:9, 1:1, 9:16, 4:5)."""
    plans: dict[str, CropPlan] = {}
    for name, ratio in TARGET_ASPECT_RATIOS.items():
        plans[name] = plan_crop(image_path, target_ratio=ratio)
    return plans
