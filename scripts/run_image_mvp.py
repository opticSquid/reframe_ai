"""Run the full image pipeline MVP on a single image.

Usage:
    python scripts/run_image_mvp.py [image_path] [--output-dir DIR] [--no-ai]
    python scripts/run_image_mvp.py data/input_image.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import TARGET_ASPECT_RATIOS
from src.pipeline import (
    ImagePipelineResult,
    VariantResult,
    process_image_to_variants,
    process_single_variant,
)


def main() -> None:
    image_path = "data/input_image.png"
    output_dir = None
    use_ai = True
    ratio_name = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        elif args[i] == "--no-ai":
            use_ai = False
            i += 1
        elif args[i] == "--ratio" and i + 1 < len(args):
            ratio_name = args[i + 1]
            i += 2
        else:
            image_path = args[i]
            i += 1

    print("=" * 70)
    print("  REFRAMEAI — Image Pipeline MVP")
    print("=" * 70)
    print(f"  Input:      {image_path}")
    print(f"  AI enabled: {use_ai}")
    if ratio_name:
        print(f"  Ratio:      {ratio_name}")
    else:
        print(f"  Ratios:     {list(TARGET_ASPECT_RATIOS.keys())}")
    print("=" * 70)
    print()

    t0 = time.perf_counter()

    if ratio_name:
        # Single variant
        result = process_single_variant(
            image_path, ratio_name,
            output_dir=output_dir, use_ai=use_ai,
        )
        print_single(result)
    else:
        # All variants
        result = process_image_to_variants(
            image_path, output_dir=output_dir, use_ai=use_ai,
        )
        print_all(result)

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 70)
    print(f"  Total pipeline time: {elapsed:.1f}s")
    if isinstance(result, ImagePipelineResult) and result.variants:
        print(f"  AI status:           {list(result.variants.values())[0].ai_status}")
    elif isinstance(result, VariantResult):
        print(f"  AI status:           {result.ai_status}")
    print("=" * 70)


def print_single(result) -> None:
    print(f"\n--- Variant: {result.ratio_name} ({result.target_ratio:.4f}) ---")
    print(f"  Subjects detected:  {result.plan.detection_count}")
    print(f"  Model:              {result.plan.model_used}")
    print(f"  AI status:          {result.ai_status}")
    print(f"  AI reasoning:       {result.plan.ai_reasoning}")
    print(f"  Passed:             {result.passed}")
    print(f"  Crop:               ({result.plan.crop.x}, {result.plan.crop.y}, "
          f"{result.plan.crop.width}x{result.plan.crop.height})")
    print(f"  Subject coverage:   {result.plan.crop.subject_coverage_pct:.1%}")
    print(f"  Edge cutoffs:       {result.plan.crop.edge_cutoffs}")
    print(f"  Is center fallback: {result.plan.crop.is_center_fallback}")
    print(f"  Iterations:         {len(result.iterations)}")
    print(f"  Rendered output:    {result.render_path}")
    print(f"  Manifest:           {result.manifest_path}")
    print(f"  Validation errors:  {result.validation.errors}")
    print(f"  Validation warnings:{result.validation.warnings}")
    if result.ai_evaluation:
        print(f"  AI eval passed:     {result.ai_evaluation.passed}")
        if result.ai_evaluation.errors:
            print(f"  AI eval errors:     {result.ai_evaluation.errors}")
    else:
        print(f"  AI eval:            (skipped — AI unavailable)")
    print(f"  Explanation:        {result.explanation}")


def print_all(result) -> None:
    print(f"  Image:             {result.image_width}x{result.image_height}")
    print(f"  Subjects detected: {result.subjects_detected}")
    print(f"  AI active:         {result.ai_detected}")
    print(f"  All passed:        {result.all_passed}")
    print()

    for name, v in result.variants.items():
        print(f"  [{name}] ratio={v.target_ratio:.4f}")
        print(f"    Crop: ({v.plan.crop.x}, {v.plan.crop.y}, "
              f"{v.plan.crop.width}x{v.plan.crop.height})")
        print(f"    Coverage: {v.plan.crop.subject_coverage_pct:.1%}")
        print(f"    Passed: {v.passed}  |  Iterations: {len(v.iterations)}")
        print(f"    Output: {v.render_path}")
        print(f"    Manifest: {v.manifest_path}")
        if v.validation.errors:
            print(f"    Validation errors: {v.validation.errors}")
        print()

    # Save full summary
    summary = result.to_summary()
    summary_path = result.variants[list(result.variants.keys())[0]].manifest_path
    summary_dir = Path(summary_path).parent.parent
    summary_file = summary_dir / "pipeline_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Full summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
