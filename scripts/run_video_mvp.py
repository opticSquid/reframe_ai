#!/usr/bin/env python3
"""End-to-end video pipeline runner.

Usage:
    python scripts/run_video_mvp.py [video_path]

If video_path is omitted, uses data/input_video.mp4.
Produces a vertical 9:16 reel + still in output/video_reels/ + output/stills/.
All debug artifacts saved to output/manifests/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, OUTPUT_DIR
from src.video_pipeline import process_video_to_reel


def main() -> None:
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = DATA_DIR / "input_video.mp4"

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        print(f"Usage: python {sys.argv[0]} <path_to_video.mp4>")
        sys.exit(1)

    print(f"=== ReframeAI Video Pipeline ===")
    print(f"Input:  {video_path}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {DATA_DIR.parent / 'models' / 'face_landmarker_float16.task'}")
    print()

    t0 = time.perf_counter()
    result = process_video_to_reel(video_path=video_path)
    elapsed = time.perf_counter() - t0

    summary = result.to_summary()
    print(f"\n=== Result ===")
    print(f"Total time:    {elapsed:.1f}s")
    print(f"Perception:    {summary['timing']['perception']}s")
    print(f"Audio:         {summary['timing']['audio']}s")
    print(f"Speaker:       {summary['timing']['speaker']}s")
    print(f"Render:        {summary['timing']['render']}s")
    print(f"Tracks:        {summary['num_tracks']}")
    print(f"Speaker segs:  {summary['num_speaker_segments']}")
    print(f"Trajectory pts: {summary['num_trajectory_points']}")
    print(f"Reel:          {result.reel_path}")
    print(f"Still:         {result.still_path}")
    if result.debug_path:
        print(f"Debug:         {result.debug_path}")
    print(f"Validation:    {'PASS' if summary['validation_passed'] else 'FAIL'}")
    for err in summary["validation_errors"]:
        print(f"  ERROR: {err}")
    for warn in summary["validation_warnings"]:
        print(f"  WARN: {warn}")


if __name__ == "__main__":
    main()
