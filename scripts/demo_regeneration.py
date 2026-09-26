"""Demonstrate the regeneration workflow with AI-assisted planning + evaluation.

Flow:
  1. MediaPipe detects subjects in the image.
  2. Gemini semantically analyzes the image + subjects → recommends crop params.
  3. Planner computes crop coordinates using both CV + AI signals.
  4. Crop is rendered.
  5. Deterministic validator checks dimensions, aspect ratio, watermark, framing.
  6. Gemini evaluates the rendered crop for composition/framing quality.
  7. If both PASS → done.  If any FAIL → structured feedback → Gemini suggests
     adjustments → re-plan → re-render → re-validate → re-evaluate.
  8. Repeat until max_retries or pass.

No deliberately bad crop is created — the first crop is the best AI-assisted attempt.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("/home/soumalya/Work/reframe_ai/.env")

import numpy as np
from PIL import Image

from src.image_crop import detect_subjects
from src.regeneration import (
    RegenerationResult,
    regenerate_crop,
)

IMAGE_PATH = "data/input_image.png"
OUTPUT_DIR = Path("output/regeneration_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RATIO_9_16 = 9 / 16

print("=" * 70)
print("  REGENERATION WORKFLOW DEMONSTRATION")
print("  AI-assisted planning + AI evaluation + feedback loop")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Step 1: Load image and detect subjects
# ---------------------------------------------------------------------------
img = Image.open(IMAGE_PATH)
src_w, src_h = img.size
print(f"Image: {src_w}x{src_h}")

subjects, model_used = detect_subjects(IMAGE_PATH)
print(f"Detected {len(subjects)} subjects with '{model_used}'")
for i, s in enumerate(subjects):
    cx = s.bbox[0] + s.bbox[2] / 2
    cy = s.bbox[1] + s.bbox[3] / 2
    print(f"  Subject {i+1}: center=({int(cx)}, {int(cy)}) "
          f"size=({int(s.bbox[2])}, {int(s.bbox[3])}) importance={s.importance:.3f}")
print()

# ---------------------------------------------------------------------------
# Step 2: Run regeneration workflow (best-effort AI planning)
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Running regenerate_crop(max_retries=3, use_ai=True)")
print("=" * 70)

result: RegenerationResult = regenerate_crop(
    image_path=IMAGE_PATH,
    target_ratio=RATIO_9_16,
    max_retries=3,
    use_ai=True,
    save_intermediates=True,
    output_dir=OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Step 3: Report results
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  REGENERATION RESULTS")
print("=" * 70)
print()

print(f"Passed: {result.passed}")
print(f"Total iterations: {len(result.iterations)}")
print()

print("--- Iteration History ---")
for rec in result.iterations:
    status_marker = "✓" if rec.status == "PASS" else "✗"
    print(f"\n  Iteration {rec.iteration} [{rec.status}] {status_marker}")
    print(f"    Crop: ({rec.crop.x}, {rec.crop.y}, {rec.crop.width}x{rec.crop.height})")
    print(f"    Edge cutoffs: {rec.crop.edge_cutoffs}")
    print(f"    Validation errors: {rec.validation.errors}")
    print(f"    Validation warnings: {rec.validation.warnings}")
    if rec.ai_evaluation:
        print(f"    AI eval passed: {rec.ai_evaluation.passed}")
        if rec.ai_evaluation.errors:
            print(f"    AI eval errors: {rec.ai_evaluation.errors}")
        if rec.ai_evaluation.warnings:
            print(f"    AI eval warnings: {rec.ai_evaluation.warnings}")
        if rec.ai_evaluation.suggestions:
            print(f"    AI eval suggestions: {rec.ai_evaluation.suggestions}")
    if rec.feedback:
        print(f"    Feedback: (truncated) {rec.feedback[:300]}...")
    if rec.ai_guidance:
        print(f"    AI guidance: (truncated) {rec.ai_guidance[:300]}...")

print()
print("--- Final Crop ---")
final = result.plan.crop
print(f"  Position: ({final.x}, {final.y})")
print(f"  Size: {final.width}x{final.height}")
aspect = final.width / final.height
print(f"  Aspect ratio: {aspect:.4f} (target: {RATIO_9_16:.4f})")
print(f"  Subject coverage: {final.subject_coverage_pct:.2%}")
print(f"  Edge cutoffs: {final.edge_cutoffs}")
print(f"  Center fallback: {final.is_center_fallback}")
print(f"  AI reasoning: {result.plan.ai_reasoning}")

print()
print("--- Final Validation ---")
fv = result.final_validation
print(f"  Passed: {fv.passed}")
print(f"  Errors: {fv.errors}")
print(f"  Warnings: {fv.warnings}")

print()
print("--- Final AI Evaluation ---")
if result.final_ai_evaluation:
    print(f"  Passed: {result.final_ai_evaluation.passed}")
    print(f"  Errors: {result.final_ai_evaluation.errors}")
    print(f"  Warnings: {result.final_ai_evaluation.warnings}")
    print(f"  Suggestions: {result.final_ai_evaluation.suggestions}")
else:
    print("  (AI evaluation was skipped — no key or not available)")

print()
print("--- Explanation ---")
print(result.explanation)

print()
print("--- Output Files ---")
for f in sorted(OUTPUT_DIR.glob("iter_*_render.png")):
    size = f.stat().st_size
    print(f"  {f.name} ({size:,} bytes)")

# Save summary
summary = result.to_summary()
summary_path = OUTPUT_DIR / "regeneration_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"\n  regeneration_summary.json ({summary_path.stat().st_size:,} bytes)")
print("\nDone.")
