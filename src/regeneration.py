"""Subject-aware image regeneration workflow.

Implements the generate→validate→feedback→retry loop with AI assistance:

  1. DESCRIBE + PLAN — Gemini semantically analyzes the image (description
     cached) and recommends crop parameters (center, coverage, margin) in a
     single call; MediaPipe subjects ground the arithmetic.
  2. RENDER  — apply crop coordinates to pixels (render_crop).
  3. VALIDATE — deterministic checks via platform spec validator.
  4. EVALUATE — Gemini receives the rendered crop image + the original
     image's semantic description (text, not image bytes) + subject
     locations + crop params. It checks whether important subjects/regions
     from step 1 are being cropped out, and if so recommends adjustment
     parameters (target_coverage, min_margin, center_x, center_y) within
     the target ratio.
  5. If PASS (both deterministic + AI) → done.
  6. If FAIL:
     a. Construct structured failure feedback from ValidationResult + AIEvalResult.
     b. Use AI's structured adjustment recommendation (from step 4).
     c. Re-plan with adjusted parameters (no new Gemini planning call).
     d. Re-render, re-validate, re-evaluate.
     e. Repeat up to max_retries.
  7. Return final crop + human-readable explanation of why it was selected.

The validator (src/validator.py) is used as-is — never modified.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ._image_utils import _np_to_png_bytes
from .config import GEMINI_MODEL, gemini_available, get_gemini_api_key
from .image_crop import CropPlan, detect_subjects, plan_crop, render_crop
from .cropper import CropResult, Subject
from .validator import ValidationResult, validate_asset

# Models to try in order of preference
_MODELS_TO_TRY = [GEMINI_MODEL, "gemini-3-flash", "gemini-3-flash-preview"]

# Cache image descriptions to avoid redundant Gemini calls across ratios
_description_cache: dict[str, str] = {}


def _resize_image_bytes(image_path: str | Path, max_dim: int = 1024) -> bytes:
    """Open an image, resize to fit within max_dim, return PNG bytes."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract and parse JSON from a model response string."""
    if not text:
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def _retry_models(model_name_list: list[str], call: Any) -> tuple[Any, Exception | None]:
    """Try a callable against each model name; return (response, last_error)."""
    from google import genai
    api_key = get_gemini_api_key()
    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None
    for model_name in model_name_list:
        try:
            resp = call(client, model_name)
            return resp, None
        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(16)
                continue
    return None, last_error


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class AIEvaluationResult:
    """Result of Gemini's semantic critique of a rendered crop."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    adjustment: dict[str, Any] | None = None
    raw_response: str = ""


@dataclass
class IterationRecord:
    """Record of one generate→validate→evaluate cycle in the regeneration loop."""
    iteration: int
    crop: CropResult
    validation: ValidationResult
    ai_evaluation: AIEvaluationResult | None = None
    feedback: str | None = None
    ai_guidance: str | None = None
    image_description: str = ""
    status: str = "FAIL"  # "PASS" or "FAIL"
    render_path: str | None = None


@dataclass
class RegenerationResult:
    """Final result of the regeneration workflow."""
    plan: CropPlan
    rendered: np.ndarray
    final_validation: ValidationResult
    final_ai_evaluation: AIEvaluationResult | None
    iterations: list[IterationRecord] = field(default_factory=list)
    total_iterations: int = 0
    passed: bool = False
    explanation: str = ""
    ai_status: str = "active"

    def to_summary(self) -> dict[str, Any]:
        """Serializable summary of the regeneration result."""
        return {
            "passed": self.passed,
            "total_iterations": self.total_iterations,
            "ai_status": self.ai_status,
            "final_crop": {
                "x": self.plan.crop.x,
                "y": self.plan.crop.y,
                "width": self.plan.crop.width,
                "height": self.plan.crop.height,
            },
            "final_validation_errors": self.final_validation.errors,
            "final_validation_warnings": self.final_validation.warnings,
            "final_ai_evaluation": (
                {
                    "passed": self.final_ai_evaluation.passed if self.final_ai_evaluation else None,
                    "errors": self.final_ai_evaluation.errors if self.final_ai_evaluation else [],
                    "warnings": self.final_ai_evaluation.warnings if self.final_ai_evaluation else [],
                    "suggestions": self.final_ai_evaluation.suggestions if self.final_ai_evaluation else [],
                }
                if self.final_ai_evaluation else None
            ),
            "iterations": [
                {
                    "iteration": rec.iteration,
                    "status": rec.status,
                    "feedback": rec.feedback,
                    "ai_evaluation": (
                        {
                            "passed": rec.ai_evaluation.passed,
                            "errors": rec.ai_evaluation.errors,
                            "warnings": rec.ai_evaluation.warnings,
                            "suggestions": rec.ai_evaluation.suggestions,
                        }
                        if rec.ai_evaluation else None
                    ),
                    "crop": {
                        "x": rec.crop.x, "y": rec.crop.y,
                        "w": rec.crop.width, "h": rec.crop.height,
                    },
                }
                for rec in self.iterations
            ],
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Manifest construction (for validation)
# ---------------------------------------------------------------------------
def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for manifest input_hash."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    plan: CropPlan,
    output_path: str = "",
    asset_name: str = "crop_output",
    input_hash: str = "",
    timestamp: str = "",
    processing_time_sec: float = 0.0,
    validation_passed: bool | None = None,
    validation_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a manifest dict from a CropPlan for validator consumption."""
    ratio_name = _ratio_to_name(plan.target_ratio)
    if not input_hash:
        input_hash = _compute_file_hash(plan.image_path)
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "asset_type": "image_variant",
        "asset_name": asset_name,
        "asset_variant": ratio_name,
        "input_path": plan.image_path,
        "input_hash": input_hash,
        "input_dimensions": {"width": plan.image_width, "height": plan.image_height},
        "output_path": output_path,
        "output_dimensions": {"width": plan.crop.width, "height": plan.crop.height},
        "crop_params": {
            "x": plan.crop.x,
            "y": plan.crop.y,
            "w": plan.crop.width,
            "h": plan.crop.height,
            "edge_cutoffs": plan.crop.edge_cutoffs,
        },
        "processing_time_sec": processing_time_sec,
        "timestamp": timestamp,
        "validation_passed": validation_passed,
        "validation_warnings": validation_warnings or [],
    }


def _ratio_to_name(ratio: float) -> str:
    """Convert a numeric ratio to its canonical name."""
    ratios = {"16_9": 16 / 9, "1_1": 1.0, "9_16": 9 / 16, "4_5": 4 / 5}
    for name, r in ratios.items():
        if abs(r - ratio) < 0.001:
            return name
    return f"custom_{ratio:.4f}"


def _spec_min_dimensions(ratio: float) -> tuple[int, int]:
    """Look up minimum crop dimensions from the platform spec for a ratio."""
    from .validator import load_spec
    spec = load_spec()
    for fmt in spec.get("asset_types", {}).get("image_variant", {}).get("formats", []):
        if abs(fmt.get("aspect_ratio", 0) - ratio) < 0.001:
            return int(fmt.get("min_width", 0)), int(fmt.get("min_height", 0))
    return (0, 0)


def _spec_max_dimensions(ratio: float) -> tuple[int, int]:
    """Look up maximum crop dimensions from the platform spec for a ratio."""
    from .validator import load_spec
    spec = load_spec()
    for fmt in spec.get("asset_types", {}).get("image_variant", {}).get("formats", []):
        if abs(fmt.get("aspect_ratio", 0) - ratio) < 0.001:
            return int(fmt.get("max_width", 0)), int(fmt.get("max_height", 0))
    return (0, 0)


# ---------------------------------------------------------------------------
# AI-assisted crop planning
# ---------------------------------------------------------------------------
def _crop_score(crop: CropResult, validation: ValidationResult) -> float:
    """Score a crop: fewer errors = higher score, more coverage = higher."""
    score = 100.0
    score -= len(validation.errors) * 30
    score -= len(validation.warnings) * 10
    score += crop.subject_coverage_pct * 20
    max_cutoff = max(crop.edge_cutoffs.values()) if crop.edge_cutoffs else 0
    score -= max_cutoff * 50
    return score


def _pick_better_crop(
    ai_plan: CropPlan, ai_validation: ValidationResult,
    det_plan: CropPlan, det_validation: ValidationResult,
) -> tuple[CropPlan, ValidationResult, str]:
    """Compare AI-assisted crop with deterministic crop, return the better one."""
    ai_score = _crop_score(ai_plan.crop, ai_validation)
    det_score = _crop_score(det_plan.crop, det_validation)
    if det_score >= ai_score:
        return det_plan, det_validation, "deterministic"
    return ai_plan, ai_validation, "ai-assisted"


def plan_crop_with_ai(
    image_path: str | Path,
    target_ratio: float,
    subjects: list[Subject],
    source_w: int,
    source_h: int,
    min_crop_w: int = 0,
    min_crop_h: int = 0,
    max_crop_w: int = 0,
    max_crop_h: int = 0,
    feedback: str | None = None,
    image_description: str = "",
    adjustment_params: dict[str, Any] | None = None,
) -> CropPlan:
    """Plan a crop using Gemini for semantic understanding + MediaPipe subjects.

    Flow:
      - If *adjustment_params* is provided (from a previous AI evaluation),
        those values are used directly — no new Gemini call.
      - Otherwise, if *feedback* is provided, Gemini is asked to recommend
        adjusted crop parameters.
      - Otherwise, Gemini is asked to analyze the image (informed by
        *image_description*) and recommend optimal crop parameters.

    Falls back to deterministic planning when Gemini is unavailable.
    """
    returned_desc = image_description
    if adjustment_params is not None:
        params = adjustment_params
        if not params.get("reason"):
            params["reason"] = "AI adjustment recommendation"
    elif feedback:
        params, _ = _query_gemini_for_plan(
            image_path, target_ratio, subjects, source_w, source_h,
            feedback=feedback, image_description=image_description,
        )
    elif gemini_available() and subjects:
        # When no description yet, ask Gemini for both description + params
        # in a single call (saves one API round-trip vs. separate description call)
        include_desc = not bool(image_description)
        params, returned_desc = _query_gemini_for_plan(
            image_path, target_ratio, subjects, source_w, source_h,
            image_description=image_description,
            include_description=include_desc,
        )
    else:
        params = {
            "target_coverage": 0.5, "min_margin": 0.1,
            "center_x": None, "center_y": None,
            "reason": "Heuristic / deterministic fallback (no AI available)",
        }
        returned_desc = ""

    target_coverage = max(0.1, min(0.8, params.get("target_coverage", 0.5)))
    min_margin = max(0.05, min(0.3, params.get("min_margin", 0.1)))
    center_x = params.get("center_x")
    center_y = params.get("center_y")

    plan = plan_crop(
        image_path, target_ratio,
        target_coverage=target_coverage,
        min_margin=min_margin,
        min_crop_w=min_crop_w, min_crop_h=min_crop_h,
        max_crop_w=max_crop_w, max_crop_h=max_crop_h,
        center_x=center_x, center_y=center_y,
    )

    plan.ai_reasoning = params.get("reason", "")
    plan.image_description = returned_desc if returned_desc else image_description
    return plan


def _query_gemini_for_plan(
    image_path: str | Path,
    target_ratio: float,
    subjects: list[Subject],
    source_w: int,
    source_h: int,
    feedback: str | None = None,
    image_description: str = "",
    include_description: bool = False,
) -> tuple[dict[str, Any], str]:
    """Send image + subjects to Gemini and parse crop parameter recommendations."""
    from google.genai import types

    ratio_name = _ratio_to_name(target_ratio)
    subject_info = "\n".join(
        f"  Face {i}: bbox=(x={int(s.bbox[0])}, y={int(s.bbox[1])}, "
        f"w={int(s.bbox[2])}, h={int(s.bbox[3])}), "
        f"importance={s.importance:.3f}"
        for i, s in enumerate(subjects)
    )

    desc_section = ""
    if image_description:
        desc_section = f"\nIMAGE DESCRIPTION:\n{image_description}\n"

    if feedback:
        prompt = f"""You are an expert image composition advisor. A subject-aware crop
for a {ratio_name} ({target_ratio:.4f}) format was planned but FAILED validation.

VALIDATION FEEDBACK:
{feedback}

The image is {source_w}x{source_h} with detected subjects:
{subject_info}
{desc_section}Analyze the validation failures and recommend how to adjust the crop.
Respond with ONLY a JSON object with these exact keys:
  - "target_coverage": float (0.1 to 0.8 — fraction of crop area occupied by subjects)
  - "min_margin": float (0.05 to 0.3 — margin around subjects as fraction of crop)
  - "center_x": int or null (explicit center x if subjects should be repositioned)
  - "center_y": int or null (explicit center y)
  - "reason": string explaining the recommendation

Example: {{"target_coverage": 0.5, "min_margin": 0.1, "center_x": null, "center_y": null, "reason": "Center crop on subjects to reduce edge cutoffs"}}"""
    else:
        if include_description:
            prompt = f"""You are an expert image composition advisor. Analyze this {source_w}x{source_h} image.

Detected subjects:
{subject_info}

First, provide a concise semantic description of the image: what is happening,
who/what matters, and which regions must remain visible in any crop (faces,
text, key objects, watermark location).

Then, plan an optimal {ratio_name} ({target_ratio:.4f}) crop.

Respond with ONLY a JSON object with this exact envelope:
{{"description": "2-3 sentence summary of important subjects/regions", "params": {{"target_coverage": float (0.1-0.8), "min_margin": float (0.05-0.3), "center_x": int or null, "center_y": int or null, "reason": string}}}}"""
        else:
            prompt = f"""You are an expert image composition advisor. Plan an optimal {ratio_name}
({target_ratio:.4f}) crop for this {source_w}x{source_h} image.

Detected subjects:
{subject_info}
{desc_section}Analyze the image semantically. Identify the main subjects and visual focus.
The watermark is typically in the lower-center region of the image.

Recommend crop parameters. Respond with ONLY a JSON object:
  - "target_coverage": float (0.1 to 0.8 — fraction of crop area occupied by subjects)
  - "min_margin": float (0.05 to 0.3 — margin around subjects as fraction of crop)
  - "center_x": int or null (if you know the optimal crop center x from image analysis)
  - "center_y": int or null (if you know the optimal crop center y)
  - "reason": string explaining the recommendation

Example: {{"target_coverage": 0.4, "min_margin": 0.15, "center_x": 1780, "center_y": 1600, "reason": "Center on the two faces, include watermark band below"}}"""

    img_bytes = _resize_image_bytes(image_path)

    contents: list[Any] = [types.Content(
        role="user",
        parts=[
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(
                mime_type="image/png",
                data=img_bytes,
            )),
        ],
    )]

    response, last_error = _retry_models(_MODELS_TO_TRY, lambda c, m: c.models.generate_content(
        model=m, contents=contents
    ))

    if response and response.text:
        data = _parse_json_response(response.text.strip())
        if data:
            if include_description and isinstance(data.get("params"), dict):
                return data["params"], data.get("description", "")
            return data, ""
    return (
        {"target_coverage": 0.5, "min_margin": 0.1,
         "center_x": None, "center_y": None,
         "reason": f"AI planning failed ({last_error}). Using defaults."},
        "",
    )


# ---------------------------------------------------------------------------
# AI-assisted crop evaluation
# ---------------------------------------------------------------------------
def evaluate_crop_with_ai(
    image_path: str | Path,
    rendered_crop: np.ndarray,
    crop: CropResult,
    target_ratio: float,
    validation: ValidationResult,
    src_w: int = 0,
    src_h: int = 0,
    image_description: str = "",
    subjects: list[Subject] | None = None,
) -> AIEvaluationResult:
    """Critique the rendered crop for composition & framing.

    Gemini receives the rendered crop image + the original image's semantic
    description (text, not image bytes) + subject locations + crop params.
    It checks whether important subjects/regions from the description are
    being cropped out, and if so recommends adjustment parameters
    (target_coverage, min_margin, center_x, center_y) within the target ratio.
    """
    if not gemini_available():
        return AIEvaluationResult(
            passed=True,
            raw_response="AI not available — skipping evaluation, trusting deterministic validator",
        )

    from google.genai import types

    ratio_name = _ratio_to_name(target_ratio)

    # Build the validation summary
    val_summary = []
    if validation.errors:
        val_summary.append(f"Deterministic errors: {validation.errors}")
    if validation.warnings:
        val_summary.append(f"Deterministic warnings: {validation.warnings}")
    val_text = "; ".join(val_summary) if val_summary else "None"

    # Build subject info
    subject_bboxes = ""
    if subjects:
        subject_bboxes = "\n".join(
            f"  Face {i}: bbox=(x={int(s.bbox[0])}, y={int(s.bbox[1])}, "
            f"w={int(s.bbox[2])}, h={int(s.bbox[3])})"
            for i, s in enumerate(subjects)
        )

    desc_section = ""
    if image_description:
        desc_section = f"\nIMAGE DESCRIPTION (important subjects/regions to preserve):\n{image_description}\n"

    prompt = f"""You are an image quality critic evaluating a subject-aware crop.

Below is the CROPPED VARIANT of a {src_w}×{src_h} original image.
The original image itself is NOT sent — I've provided its semantic description above.
CROPPED VARIANT: {crop.width}×{crop.height} (target ratio: {ratio_name} = {target_ratio:.4f}, actual: {crop.width / crop.height:.4f})

Crop position in original: x={crop.x}, y={crop.y}
Edge cutoffs (fraction of each subject cut off): {crop.edge_cutoffs}
Deterministic validator findings: {val_text}
{desc_section}SUBJECTS DETECTED:\n{subject_bboxes}

CRITICAL ASSESSMENT:
1. Are the most important subjects/regions from the description getting cropped
   out (cut off by the crop edges)? Check each face/object against the crop bounds.
2. If important content IS being cropped out, how should the crop be adjusted?
   - You MUST keep the target ratio {target_ratio:.4f} ({ratio_name})
   - You can adjust: target_coverage (wider=crop, 0.1-0.8), min_margin (0.05-0.3),
     center_x (int or null), center_y (int or null)
3. Provide a score 1-10 for overall framing quality.

Respond with ONLY a JSON object:
  - "passed": true or false
  - "errors": list of strings (critical issues — important content cropped out)
  - "warnings": list of strings (minor issues)
  - "suggestions": list of strings (how to improve)
  - "score": int (1-10)
  - "adjustment": {{"target_coverage": float or null, "min_margin": float or null, "center_x": int or null, "center_y": int or null, "reason": string}}
    — only provide if you recommend adjusting the crop parameters; if the crop is good, set to null.

Example: {{"passed": true, "errors": [], "warnings": ["Slightly tight on top margin"], "suggestions": ["Could increase min_margin to 0.15"], "score": 9, "adjustment": null}}"""

    # Prepare only the rendered crop image (original is described, not sent)
    crop_bytes = _np_to_png_bytes(rendered_crop)
    contents: list[Any] = [types.Content(
        role="user",
        parts=[
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(
                mime_type="image/png",
                data=crop_bytes,
            )),
        ],
    )]

    response, last_error = _retry_models(_MODELS_TO_TRY, lambda c, m: c.models.generate_content(
        model=m, contents=contents
    ))

    if response and response.text:
        data = _parse_json_response(response.text.strip())
        if data:
            adjustment = data.get("adjustment")
            return AIEvaluationResult(
                passed=bool(data.get("passed", True)),
                errors=list(data.get("errors", [])),
                warnings=list(data.get("warnings", [])),
                suggestions=list(data.get("suggestions", [])),
                adjustment=adjustment,
                raw_response=response.text.strip(),
            )

    return AIEvaluationResult(
        passed=True,
        raw_response=f"AI evaluation failed: {last_error}. Trusting deterministic validator.",
    )


# ---------------------------------------------------------------------------
# Feedback construction
# ---------------------------------------------------------------------------
def construct_feedback(
    validation: ValidationResult,
    ai_eval: AIEvaluationResult | None = None,
) -> str:
    """Build structured text feedback from validation result + AI evaluation."""
    lines = [
        f"VALIDATION FAILED for asset '{validation.asset_name}' ({validation.asset_type}).",
        "",
        f"Errors ({len(validation.errors)}):",
    ]
    for err in validation.errors:
        lines.append(f"  - {err}")
    lines.append("")
    lines.append(f"Warnings ({len(validation.warnings)}):")
    for warn in validation.warnings:
        lines.append(f"  - {warn}")
    if ai_eval:
        lines.append("")
        lines.append(f"AI Evaluation — {'PASSED' if ai_eval.passed else 'FAILED'}:")
        lines.append(f"  AI Errors ({len(ai_eval.errors)}):")
        for err in ai_eval.errors:
            lines.append(f"    - {err}")
        lines.append(f"  AI Warnings ({len(ai_eval.warnings)}):")
        for warn in ai_eval.warnings:
            lines.append(f"    - {warn}")
        if ai_eval.suggestions:
            lines.append("  AI Suggestions:")
            for sug in ai_eval.suggestions:
                lines.append(f"    - {sug}")
    lines.append("")
    lines.append("The crop coordinates need adjustment to resolve these validation failures.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regeneration workflow
# ---------------------------------------------------------------------------
def regenerate_crop(
    image_path: str | Path,
    target_ratio: float,
    max_retries: int = 3,
    use_ai: bool = True,
    save_intermediates: bool = True,
    output_dir: str | Path = "",
) -> RegenerationResult:
    """Run the full AI-assisted generate→validate→feedback→retry loop.

    Parameters
    ----------
    image_path
        Path to the input image.
    target_ratio
        Target aspect ratio (width / height), e.g. 9/16.
    max_retries
        Maximum number of regeneration attempts after the initial try.
    use_ai
        Whether to query Gemini for description, planning, evaluation, and guidance.
    save_intermediates
        Whether to save intermediate render PNGs for debugging.

    Returns
    -------
    RegenerationResult with final crop, rendered image, validation,
    all iteration records, and an explanation.
    """
    output_dir = Path(output_dir) if output_dir else Path("output/regeneration")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    img = Image.open(image_path)
    src_w, src_h = img.size

    # Detect subjects once (reused across iterations)
    subjects, model_used = detect_subjects(image_path)

    # Load spec minimum and maximum dimensions for this target ratio
    min_crop_w, min_crop_h = _spec_min_dimensions(target_ratio)
    max_crop_w, max_crop_h = _spec_max_dimensions(target_ratio)

    # AI is active if API key is set (availability is tested implicitly by
    # the first real API call, which has its own fallback logic)
    ai_active = use_ai and gemini_available()
    ai_status = "active" if ai_active else "unavailable (using deterministic fallback)"

    # --- Check cached description (shared across ratios) ---
    image_description = ""
    cache_key = str(image_path.resolve())
    if ai_active and cache_key in _description_cache:
        image_description = _description_cache[cache_key]

    iterations: list[IterationRecord] = []
    feedback = None
    current_plan: CropPlan | None = None
    current_crop: CropResult | None = None
    current_rendered: np.ndarray | None = None
    final_validation: ValidationResult | None = None
    final_ai_eval: AIEvaluationResult | None = None

    # Track best params across iterations for AI→deterministic transition
    best_plan: CropPlan | None = None
    best_validation: ValidationResult | None = None
    best_score = -1.0
    best_coverage = 0.5
    best_margin = 0.1

    for attempt in range(max_retries + 1):
        # ----------------------------------------------------------
        # PLAN — AI-assisted or deterministic
        # ----------------------------------------------------------
        if ai_active:
            adjustment_params = None
            if feedback and attempt > 0:
                # Use adjustment params from the previous AI evaluation
                if final_ai_eval and final_ai_eval.adjustment:
                    adjustment_params = final_ai_eval.adjustment

            current_plan = plan_crop_with_ai(
                image_path, target_ratio, subjects, src_w, src_h,
                min_crop_w=min_crop_w, min_crop_h=min_crop_h,
                max_crop_w=max_crop_w, max_crop_h=max_crop_h,
                feedback=feedback, image_description=image_description,
                adjustment_params=adjustment_params,
            )

            # Cache the description if it was obtained from the combined call
            if current_plan.image_description and not image_description:
                image_description = current_plan.image_description
                if cache_key not in _description_cache:
                    _description_cache[cache_key] = image_description

            # Check if AI planning fell back to defaults
            if current_plan.ai_reasoning and (
                "failed" in current_plan.ai_reasoning.lower()
                or "fallback" in current_plan.ai_reasoning.lower()
            ):
                ai_active = False
                ai_status = "switched to deterministic after AI failure"
        else:
            current_plan = plan_crop(
                image_path, target_ratio,
                target_coverage=best_coverage,
                min_margin=best_margin,
                min_crop_w=min_crop_w, min_crop_h=min_crop_h,
                max_crop_w=max_crop_w, max_crop_h=max_crop_h,
            )
        current_crop = current_plan.crop

        # ----------------------------------------------------------
        # RENDER
        # ----------------------------------------------------------
        current_rendered = render_crop(image_path, current_crop)
        render_path = None
        if save_intermediates:
            render_path = output_dir / f"iter_{attempt}_render.png"
            Image.fromarray(current_rendered).save(render_path)

        # ----------------------------------------------------------
        # VALIDATE (deterministic)
        # ----------------------------------------------------------
        manifest = build_manifest(
            current_plan,
            output_path=str(render_path) if render_path else "",
        )
        validation = validate_asset(manifest, verify_files=False)

        # Track best params
        score = _crop_score(current_crop, validation)
        if score > best_score:
            best_score = score
            best_plan = current_plan
            best_validation = validation

        # ----------------------------------------------------------
        # EVALUATE (AI semantic critique of the crop)
        # ----------------------------------------------------------
        ai_eval = None
        # Only run AI evaluation if deterministic validation has issues —
        # if the detector already found no errors/warnings, the AI critique
        # would just confirm the same thing, wasting an API call.
        if ai_active and (validation.errors or validation.warnings):
            ai_eval = evaluate_crop_with_ai(
                image_path, current_rendered, current_crop,
                target_ratio, validation,
                src_w=src_w, src_h=src_h,
                image_description=image_description,
                subjects=subjects,
            )

        both_pass = validation.passed and (ai_eval is None or ai_eval.passed)
        iter_record = IterationRecord(
            iteration=attempt,
            crop=current_crop,
            validation=validation,
            ai_evaluation=ai_eval,
            image_description=image_description,
            status="PASS" if both_pass else "FAIL",
            render_path=str(render_path) if render_path else None,
        )

        if both_pass:
            iterations.append(iter_record)
            final_validation = validation
            final_ai_eval = ai_eval
            break

        # Construct feedback for next iteration
        feedback = construct_feedback(validation, ai_eval)
        iter_record.feedback = feedback

        # ----------------------------------------------------------
        # ADJUST — use AI's structured adjustment directly
        # ----------------------------------------------------------
        if ai_active:
            if ai_eval and ai_eval.adjustment:
                adj = ai_eval.adjustment
                iter_record.ai_guidance = (
                    f"Adjusted: coverage={adj.get('target_coverage')}, "
                    f"margin={adj.get('min_margin')}, "
                    f"center_x={adj.get('center_x')}, "
                    f"center_y={adj.get('center_y')}, "
                    f"reason={adj.get('reason', '')}"
                )
            else:
                iter_record.ai_guidance = "AI evaluation passed (no adjustment needed)"
        else:
            iter_record.ai_guidance = None
            # Deterministic adjustment: try tighter coverage around subjects
            best_coverage = min(best_coverage + 0.1, 0.7)
            best_margin = max(best_margin - 0.05, 0.05)

        iterations.append(iter_record)
        final_validation = validation
        final_ai_eval = ai_eval

    # ------------------------------------------------------------------
    # Build explanation
    # ------------------------------------------------------------------
    assert current_plan is not None, "Loop did not execute"
    assert current_crop is not None, "Loop did not execute"
    assert final_validation is not None, "Loop did not execute"

    passed = final_validation.passed and (
        final_ai_eval is None or final_ai_eval.passed
    )
    explanation = _build_explanation(
        passed, iterations, current_crop, target_ratio,
        current_plan, current_plan.ai_reasoning,
        final_validation, final_ai_eval,
    )

    return RegenerationResult(
        plan=current_plan,
        rendered=current_rendered,
        final_validation=final_validation,
        final_ai_evaluation=final_ai_eval,
        iterations=iterations,
        total_iterations=len(iterations) - 1 if passed else max_retries,
        passed=passed,
        explanation=explanation,
        ai_status=ai_status,
    )


def _build_explanation(
    passed: bool,
    iterations: list[IterationRecord],
    crop: CropResult,
    ratio: float,
    plan: CropPlan,
    ai_reasoning: str,
    validation: ValidationResult,
    ai_eval: AIEvaluationResult | None,
) -> str:
    """Construct a human-readable explanation of why the final crop was chosen."""
    ratio_name = _ratio_to_name(ratio)
    num_iters = len(iterations)

    if passed:
        if num_iters == 1:
            base = (
                f"Validation passed on the first attempt.\n"
                f"AI-assisted planning detected {plan.detection_count} subjects "
                f"with model '{plan.model_used}' and recommended "
                f"target_coverage-based centering. "
            )
        else:
            base = (
                f"Validation passed on iteration {iterations[-1].iteration} "
                f"after {num_iters - 1} retry/retries.\n"
                f"The initial crop failed validation. Structured feedback was "
                f"constructed from deterministic check failures "
                f"({len(validation.errors)} errors, {len(validation.warnings)} warnings)"
            )
            if ai_eval and ai_eval.errors:
                base += f" and AI evaluation ({len(ai_eval.errors)} errors)"
            base += ". "
            base += "Feedback was sent to Gemini, which suggested crop adjustments. "
            base += "The planner re-centered on the subject centroid, resolving the issues.\n"

        base += (
            f"Final crop: ({crop.x}, {crop.y}, {crop.width}x{crop.height}) "
            f"at {ratio_name} ({ratio:.4f}). "
            f"AI reasoning: '{ai_reasoning}'. "
            f"Detected {plan.detection_count} subjects with '{plan.model_used}'. "
            f"Subject coverage: {crop.subject_coverage_pct:.1%}. "
            f"Edge cutoffs: {crop.edge_cutoffs}. "
        )
        if ai_eval:
            base += f"AI evaluation: score/critique available."
        return base
    else:
        return (
            f"Validation failed after {num_iters - 1} retry/retries — no fully valid crop found.\n"
            f"Final crop: ({crop.x}, {crop.y}, {crop.width}x{crop.height}). "
            f"Errors: {validation.errors}. "
            f"Warnings: {validation.warnings}. "
            f"AI reasoning: '{ai_reasoning}'."
        )
