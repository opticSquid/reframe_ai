"""Top-level image pipeline orchestrator.

Ties together the five conceptual stages:
  PERCEPTION → SEMANTIC REASONING → CROP PLANNING → RENDERING → VALIDATION/AI REVIEW

This module is the entry point for processing a single image into multiple
platform-ready variants (16:9, 1:1, 9:16, 4:5). Each variant is independently
regeneratable.

Reuses existing building blocks:
  - detect_subjects (MediaPipe) — perception stage
  - plan_crop_with_ai / plan_crop — semantic reasoning + crop planning
  - render_crop — rendering stage
  - validate_asset — deterministic validation stage
  - evaluate_crop_with_ai — AI visual review stage
  - regenerate_crop — full refinement loop (plan→render→validate→review→refine)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    TARGET_ASPECT_RATIOS,
    OUTPUT_DIR,
    gemini_available,
)
from .image_crop import (
    CropPlan,
    detect_subjects,
    render_crop,
)
from .validator import (
    ValidationResult,
    validate_asset,
)
from .regeneration import (
    AIEvaluationResult,
    RegenerationResult,
    build_manifest,
    regenerate_crop,
)


@dataclass
class VariantResult:
    """Result of processing a single image variant (one aspect ratio)."""

    ratio_name: str
    target_ratio: float
    plan: CropPlan
    rendered: np.ndarray
    render_path: str
    manifest: dict[str, Any]
    manifest_path: str
    validation: ValidationResult
    ai_evaluation: AIEvaluationResult | None
    passed: bool
    iterations: list[Any]
    explanation: str
    ai_status: str
    processing_time_sec: float


@dataclass
class ImagePipelineResult:
    """Result of processing an image into all platform variants."""

    image_path: str
    image_width: int
    image_height: int
    variants: dict[str, VariantResult] = field(default_factory=dict)
    all_passed: bool = False
    subjects_detected: int = 0
    ai_detected: bool = False

    def to_summary(self) -> dict[str, Any]:
        """Serializable summary for CLI output and API responses."""
        return {
            "image_path": self.image_path,
            "image_dimensions": {"width": self.image_width, "height": self.image_height},
            "subjects_detected": self.subjects_detected,
            "ai_active": self.ai_detected,
            "all_passed": self.all_passed,
            "variants": {
                name: {
                    "ratio_name": v.ratio_name,
                    "crop": {
                        "x": v.plan.crop.x,
                        "y": v.plan.crop.y,
                        "width": v.plan.crop.width,
                        "height": v.plan.crop.height,
                    },
                    "render_path": v.render_path,
                    "manifest_path": v.manifest_path,
                    "passed": v.passed,
                    "iterations": len(v.iterations),
                    "ai_eval_passed": v.ai_evaluation.passed if v.ai_evaluation else None,
                    "processing_time_sec": round(v.processing_time_sec, 3),
                }
                for name, v in self.variants.items()
            },
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def process_image_to_variants(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    max_retries: int = 3,
    use_ai: bool = True,
    save_intermediates: bool = False,
) -> ImagePipelineResult:
    """Process a single image into all 4 platform-ready variants.

    Steps per variant (via regenerate_crop):
      1. PERCEPTION — MediaPipe detects subjects
      2. SEMANTIC REASONING — Gemini recommends crop params (if available)
      3. CROP PLANNING — deterministic arithmetic from subjects + AI params
      4. RENDERING — PIL/numpy crop slice
      5. VALIDATION — deterministic spec validator + Gemini AI visual review
      6. REFINEMENT — if any check fails, construct feedback → re-plan → retry

    Parameters
    ----------
    image_path
        Path to the input image.
    output_dir
        Base output directory. Subdirs: image_variants/, manifests/.
        Defaults to project output/ dir.
    max_retries
        Maximum refinement attempts per variant.
    use_ai
        Whether to query Gemini for planning and evaluation.
    save_intermediates
        Whether to save intermediate render iterations.

    Returns
    -------
    ImagePipelineResult with all variants, validation status, and paths.
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    image_variants_dir = output_dir / "image_variants"
    manifests_dir = output_dir / "manifests"
    image_variants_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Load image once
    img = Image.open(image_path)
    src_w, src_h = img.size

    # Detect subjects once (shared across all variants)
    subjects, model_used = detect_subjects(image_path)

    # Detect AI availability
    ai_detected = gemini_available() and len(subjects) > 0

    variants: dict[str, VariantResult] = {}
    all_passed = True

    for ratio_name, target_ratio in TARGET_ASPECT_RATIOS.items():
        t0 = time.perf_counter()

        # Run the full regeneration loop for this ratio
        result: RegenerationResult = regenerate_crop(
            image_path=image_path,
            target_ratio=target_ratio,
            max_retries=max_retries,
            use_ai=use_ai,
            save_intermediates=save_intermediates,
            output_dir=output_dir / f"regen_{ratio_name}",
        )

        # Save the final rendered crop
        output_path = image_variants_dir / f"{image_path.stem}_{ratio_name}.png"
        Image.fromarray(result.rendered).save(output_path)

        # Build complete manifest with all required fields
        processing_time = time.perf_counter() - t0
        manifest = build_manifest(
            plan=result.plan,
            output_path=str(output_path),
            asset_name=f"{image_path.stem}_{ratio_name}",
            processing_time_sec=processing_time,
            validation_passed=result.final_validation.passed,
            validation_warnings=result.final_validation.warnings,
        )

        # Save manifest
        manifest_path = manifests_dir / f"{image_path.stem}_{ratio_name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        # Re-validate with file verification on the saved output
        final_validation = validate_asset(manifest, verify_files=True)

        passed = final_validation.passed and (
            result.final_ai_evaluation is None or result.final_ai_evaluation.passed
        )
        if not passed:
            all_passed = False

        variants[ratio_name] = VariantResult(
            ratio_name=ratio_name,
            target_ratio=target_ratio,
            plan=result.plan,
            rendered=result.rendered,
            render_path=str(output_path),
            manifest=manifest,
            manifest_path=str(manifest_path),
            validation=final_validation,
            ai_evaluation=result.final_ai_evaluation,
            passed=passed,
            iterations=result.iterations,
            explanation=result.explanation,
            ai_status=result.ai_status,
            processing_time_sec=processing_time,
        )

    return ImagePipelineResult(
        image_path=str(image_path),
        image_width=src_w,
        image_height=src_h,
        variants=variants,
        all_passed=all_passed,
        subjects_detected=len(subjects),
        ai_detected=ai_detected,
    )


def process_single_variant(
    image_path: str | Path,
    ratio_name: str,
    output_dir: str | Path | None = None,
    max_retries: int = 3,
    use_ai: bool = True,
) -> VariantResult:
    """Process a single image variant (one aspect ratio) independently.

    Convenience wrapper for single-variant regeneration.
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)

    if ratio_name not in TARGET_ASPECT_RATIOS:
        raise ValueError(
            f"Unknown ratio '{ratio_name}'. Valid: {list(TARGET_ASPECT_RATIOS.keys())}"
        )

    target_ratio = TARGET_ASPECT_RATIOS[ratio_name]
    t0 = time.perf_counter()

    # Run regenerate_crop directly
    result: RegenerationResult = regenerate_crop(
        image_path=image_path,
        target_ratio=target_ratio,
        max_retries=max_retries,
        use_ai=use_ai,
        save_intermediates=False,
        output_dir=output_dir / f"regen_{ratio_name}",
    )

    # Save rendered crop
    image_variants_dir = output_dir / "image_variants"
    manifests_dir = output_dir / "manifests"
    image_variants_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    output_path = image_variants_dir / f"{image_path.stem}_{ratio_name}.png"
    Image.fromarray(result.rendered).save(output_path)

    processing_time = time.perf_counter() - t0
    manifest = build_manifest(
        plan=result.plan,
        output_path=str(output_path),
        asset_name=f"{image_path.stem}_{ratio_name}",
        processing_time_sec=processing_time,
        validation_passed=result.final_validation.passed,
        validation_warnings=result.final_validation.warnings,
    )

    manifest_path = manifests_dir / f"{image_path.stem}_{ratio_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    final_validation = validate_asset(manifest, verify_files=True)

    passed = final_validation.passed and (
        result.final_ai_evaluation is None or result.final_ai_evaluation.passed
    )

    return VariantResult(
        ratio_name=ratio_name,
        target_ratio=target_ratio,
        plan=result.plan,
        rendered=result.rendered,
        render_path=str(output_path),
        manifest=manifest,
        manifest_path=str(manifest_path),
        validation=final_validation,
        ai_evaluation=result.final_ai_evaluation,
        passed=passed,
        iterations=result.iterations,
        explanation=result.explanation,
        ai_status=result.ai_status,
        processing_time_sec=processing_time,
    )
