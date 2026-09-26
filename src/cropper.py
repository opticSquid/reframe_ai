"""Subject-aware crop computation.

Given source image dimensions, detected subjects (with bounding boxes and
importance scores), and a target aspect ratio, computes the optimal crop
rectangle that:

  - Centers on the primary subject(s)
  - Maintains the target aspect ratio
  - Keeps the subject at an appropriate size in the frame (configurable)
  - Clamps to image boundaries
  - Falls back to center crop when no subjects are detected

This is pure arithmetic — no ML, fully deterministic, unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .config import TARGET_ASPECT_RATIOS, INFERENCE_WIDTH


@dataclass
class Subject:
    """A detected subject in the image."""

    bbox: tuple[float, float, float, float]
    """Bounding box as (x, y, width, height) in pixel coordinates."""

    importance: float = 1.0
    """Importance weight (0.0–1.0) for multi-subject crop prioritization."""

    id: str = ""
    """Optional identifier for the subject."""

    category: str = "person"
    """Subject category: 'person', 'object', 'action', etc."""


@dataclass
class CropResult:
    """Result of a crop computation."""

    x: int
    y: int
    width: int
    height: int

    subject_coverage_pct: float = 0.0
    """Percentage of the crop area covered by the weighted subject union."""

    edge_cutoffs: dict[str, float] = field(default_factory=dict)
    """Per-edge fraction of subject bbox cut off: {'top','bottom','left','right'}."""

    primary_subject_center: tuple[float, float] = (0.0, 0.0)
    """Center of the primary (highest-importance) subject, in source coords."""

    scale_used: float = 1.0
    """Scale factor applied to the subject bbox to determine crop size."""

    is_center_fallback: bool = False
    """True if no subjects were detected and center crop was used."""


def _weighted_bbox(subjects: Sequence[Subject]) -> tuple[float, float, float, float]:
    """Compute the union bounding box that encompasses all subjects.

    Returns (x, y, width, height) of the bounding box covering every subject.
    Uses importance weights only to bias the *centroid* used for centering
    (see :func:`_weighted_centroid`), not to shrink the bbox itself.
    """
    if not subjects:
        return (0.0, 0.0, 0.0, 0.0)

    min_x = min(s.bbox[0] for s in subjects)
    min_y = min(s.bbox[1] for s in subjects)
    max_right = max(s.bbox[0] + s.bbox[2] for s in subjects)
    max_bottom = max(s.bbox[1] + s.bbox[3] for s in subjects)

    return (min_x, min_y, max_right - min_x, max_bottom - min_y)


def _weighted_centroid(subjects: Sequence[Subject]) -> tuple[float, float]:
    """Importance-weighted centroid of subject centers.

    Used to center the crop so that more important subjects are
    placed closer to the frame center.
    """
    if not subjects:
        return (0.0, 0.0)

    total_weight = sum(max(s.importance, 0.001) for s in subjects)
    cx = sum((s.bbox[0] + s.bbox[2] / 2) * max(s.importance, 0.001) for s in subjects) / total_weight
    cy = sum((s.bbox[1] + s.bbox[3] / 2) * max(s.importance, 0.001) for s in subjects) / total_weight
    return (cx, cy)


def _subject_area(bbox: tuple[float, float, float, float]) -> float:
    """Area of a bounding box."""
    return bbox[2] * bbox[3]


def compute_crop(
    source_w: int,
    source_h: int,
    subjects: Sequence[Subject],
    target_ratio: float,
    target_coverage: float = 0.5,
    min_margin: float = 0.1,
    max_edge_cutoff: float = 0.25,
) -> CropResult:
    """Compute the optimal subject-aware crop.

    Parameters
    ----------
    source_w, source_h
        Dimensions of the source image in pixels.
    subjects
        List of detected subjects with bounding boxes and importance weights.
    target_ratio
        Target aspect ratio (width / height).
    target_coverage
        Desired fraction of the crop area that the subject union should occupy.
        Higher = tighter crop on the subject. Default 0.5.
    min_margin
        Minimum margin (as fraction of crop dimension) to leave around subjects.
    max_edge_cutoff
        Maximum fraction of a subject that may be cut off at any edge.
        If cutting would exceed this, the crop is adjusted.

    Returns
    -------
    CropResult with crop rectangle and quality metrics.
    """
    # ------------------------------------------------------------------
    # Fallback: center crop if no subjects
    # ------------------------------------------------------------------
    if not subjects or all(s.bbox[2] * s.bbox[3] < 1 for s in subjects):
        return _center_crop(source_w, source_h, target_ratio)

    # ------------------------------------------------------------------
    # Step 1: Weighted subject bounding box
    # ------------------------------------------------------------------
    subj_bbox = _weighted_bbox(subjects)
    subj_area = _subject_area(subj_bbox)

    if subj_area < 1:
        return _center_crop(source_w, source_h, target_ratio)

    # ------------------------------------------------------------------
    # Step 2: Determine crop dimensions from target ratio and subject size
    # ------------------------------------------------------------------
    # We want: crop_area * target_coverage = subject_area
    # So: crop_area = subject_area / target_coverage
    target_crop_area = subj_area / max(target_coverage, 0.01)

    if target_ratio >= 1.0:
        # Landscape: width >= height
        crop_w = (target_crop_area * target_ratio) ** 0.5
        crop_h = crop_w / target_ratio
    else:
        # Portrait: height > width
        crop_h = (target_crop_area / target_ratio) ** 0.5
        crop_w = crop_h * target_ratio

    # Apply minimum margin
    margin_w = min_margin * crop_w
    margin_h = min_margin * crop_h
    crop_w = max(crop_w, subj_bbox[2] + 2 * margin_w)
    crop_h = max(crop_h, subj_bbox[3] + 2 * margin_h)

    # Ensure we don't exceed source dimensions
    crop_w = min(crop_w, float(source_w))
    crop_h = min(crop_h, float(source_h))

    # Adjust to maintain exact ratio
    if target_ratio >= 1.0:
        crop_h = crop_w / target_ratio
    else:
        crop_w = crop_h * target_ratio

    crop_w = min(crop_w, float(source_w))
    crop_h = min(crop_h, float(source_h))

    if target_ratio >= 1.0:
        crop_w = crop_h * target_ratio
    else:
        crop_h = crop_w / target_ratio

    # ------------------------------------------------------------------
    # Step 3: Center on weighted subject centroid
    # ------------------------------------------------------------------
    centroid_x, centroid_y = _weighted_centroid(subjects)

    # Ideal crop position (centered on centroid)
    ideal_x = centroid_x - crop_w / 2
    ideal_y = centroid_y - crop_h / 2

    # ------------------------------------------------------------------
    # Step 4: Clamp to image boundaries
    # ------------------------------------------------------------------
    crop_x = max(0, ideal_x)
    crop_y = max(0, ideal_y)

    # If crop would exceed right/bottom edge, shift left/up
    if crop_x + crop_w > source_w:
        crop_x = source_w - crop_w
    if crop_y + crop_h > source_h:
        crop_y = source_h - crop_h

    # Final clamp
    crop_x = max(0, min(crop_x, float(source_w - crop_w)))
    crop_y = max(0, min(crop_y, float(source_h - crop_h)))

    # Convert to int
    crop_x_int = int(round(crop_x))
    crop_y_int = int(round(crop_y))
    crop_w_int = max(1, int(round(crop_w)))
    crop_h_int = max(1, int(round(crop_h)))

    # ------------------------------------------------------------------
    # Step 5: Compute quality metrics
    # ------------------------------------------------------------------
    edge_cutoffs = _compute_edge_cutoffs(
        subjects, crop_x_int, crop_y_int, crop_w_int, crop_h_int
    )

    coverage = _compute_coverage(subjects, crop_x_int, crop_y_int, crop_w_int, crop_h_int)

    # Primary subject = highest importance
    primary = max(subjects, key=lambda s: s.importance) if subjects else None
    primary_center = (
        primary.bbox[0] + primary.bbox[2] / 2,
        primary.bbox[1] + primary.bbox[3] / 2,
    ) if primary else (source_w / 2, source_h / 2)

    # Compute scale used relative to subject
    scale_used = crop_w_int / subj_bbox[2] if subj_bbox[2] > 0 else 1.0

    return CropResult(
        x=crop_x_int,
        y=crop_y_int,
        width=crop_w_int,
        height=crop_h_int,
        subject_coverage_pct=coverage,
        edge_cutoffs=edge_cutoffs,
        primary_subject_center=primary_center,
        scale_used=scale_used,
        is_center_fallback=False,
    )


def _center_crop(source_w: int, source_h: int, target_ratio: float) -> CropResult:
    """Fallback: center crop maintaining target aspect ratio."""
    if target_ratio >= 1.0:
        crop_w = float(source_w)
        crop_h = crop_w / target_ratio
        if crop_h > source_h:
            crop_h = float(source_h)
            crop_w = crop_h * target_ratio
    else:
        crop_h = float(source_h)
        crop_w = crop_h * target_ratio
        if crop_w > source_w:
            crop_w = float(source_w)
            crop_h = crop_w / target_ratio

    crop_x = (source_w - crop_w) / 2
    crop_y = (source_h - crop_h) / 2

    return CropResult(
        x=int(round(crop_x)),
        y=int(round(crop_y)),
        width=max(1, int(round(crop_w))),
        height=max(1, int(round(crop_h))),
        subject_coverage_pct=0.0,
        edge_cutoffs={"top": 0, "bottom": 0, "left": 0, "right": 0},
        primary_subject_center=(source_w / 2, source_h / 2),
        scale_used=1.0,
        is_center_fallback=True,
    )


def _compute_edge_cutoffs(
    subjects: Sequence[Subject],
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
) -> dict[str, float]:
    """Compute fraction of each subject cut off at each edge."""
    cutoffs = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    if not subjects:
        return cutoffs

    for s in subjects:
        x, y, w, h = s.bbox
        if w * h < 1:
            continue

        # Left cutoff
        if x < crop_x:
            cut = (crop_x - x) / w
            cutoffs["left"] = max(cutoffs["left"], cut)
        # Right cutoff
        if x + w > crop_x + crop_w:
            cut = ((x + w) - (crop_x + crop_w)) / w
            cutoffs["right"] = max(cutoffs["right"], cut)
        # Top cutoff
        if y < crop_y:
            cut = (crop_y - y) / h
            cutoffs["top"] = max(cutoffs["top"], cut)
        # Bottom cutoff
        if y + h > crop_y + crop_h:
            cut = ((y + h) - (crop_y + crop_h)) / h
            cutoffs["bottom"] = max(cutoffs["bottom"], cut)

    return {k: round(v, 4) for k, v in cutoffs.items()}


def _compute_coverage(
    subjects: Sequence[Subject],
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
) -> float:
    """Compute fraction of crop area covered by subjects."""
    crop_area = crop_w * crop_h
    if crop_area < 1:
        return 0.0

    total_subj_area = sum(
        s.bbox[2] * s.bbox[3] * max(s.importance, 0.001)
        for s in subjects
        if s.bbox[2] * s.bbox[3] >= 1
    )

    # Simple coverage (not pixel-perfect intersection, just bbox area sum)
    return min(total_subj_area / crop_area, 1.0)


def compute_all_image_crops(
    source_w: int,
    source_h: int,
    subjects: Sequence[Subject],
    target_coverages: dict[str, float] | None = None,
) -> dict[str, CropResult]:
    """Compute crops for all standard image aspect ratios.

    Returns a dict keyed by ratio name (e.g., '16_9', '1_1', '9_16', '4_5').
    """
    if target_coverages is None:
        target_coverages = {"16_9": 0.5, "1_1": 0.5, "9_16": 0.6, "4_5": 0.55}

    results = {}
    for name, ratio in TARGET_ASPECT_RATIOS.items():
        results[name] = compute_crop(
            source_w=source_w,
            source_h=source_h,
            subjects=subjects,
            target_ratio=ratio,
            target_coverage=target_coverages.get(name, 0.5),
        )

    return results
