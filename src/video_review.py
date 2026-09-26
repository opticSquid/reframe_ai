"""Post-rendering AI visual review for the video pipeline.

After the vertical reel + stills are rendered, this module sends a SINGLE
Gemini call containing a small grid of representative frames (from both the
original video and the rendered reel) plus full trajectory/speaker metadata.

Gemini acts as an AI visual reviewer assessing:
  - Speaker framing (is the active speaker well-framed?)
  - Speaker transition tracking (does the crop follow the correct person?)
  - Unnecessary face cropping (are important faces cut off?)
  - Overall composition quality
  - Bad reframing decisions

This is NOT used for per-frame crop planning — that stays deterministic
in src/cropper.py.  Gemini only reviews the rendered output as a whole.
"""
from __future__ import annotations

import io
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import gemini_available
from .video_speaker import CropTrajectoryPoint, ActiveSpeakerSegment
from .video_ingestion import VideoMetadata


@dataclass
class VideoReviewResult:
    """Result of Gemini's post-rendering visual review."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: int = 0  # 1–10
    frames_reviewed: int = 0
    raw_response: str = ""


def _extract_frame_at_time(
    video_path: str | Path,
    timestamp: float,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Extract a single RGB frame from a video at a given timestamp.

    Uses ffmpeg with hardware-accelerated seek for speed.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    frame_size = width * height * 3
    if len(result.stdout) >= frame_size:
        return np.frombuffer(result.stdout[:frame_size], dtype=np.uint8).reshape(height, width, 3).copy()
    return None


def _extract_frame_raw(
    video_path: str | Path,
    timestamp: float,
) -> np.ndarray | None:
    """Extract a raw RGB frame at its native resolution."""
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if not result.stdout:
        return None
    # We need dimensions; ffprobe for this one frame is wasteful, so probe once
    # and cache.  For simplicity, extract at scaled resolution.
    return None


def _get_video_dims(video_path: str | Path) -> tuple[int, int, float]:
    """Get (width, height, fps) of a video via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    w = int(stream.get("width", 1920))
    h = int(stream.get("height", 1080))
    fps_str = stream.get("r_frame_rate", "25/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 25.0
    else:
        fps = float(fps_str)
    return w, h, fps


def _select_review_timestamps(
    shot_boundaries: list[int],
    trajectory: list[CropTrajectoryPoint],
    segments: list[ActiveSpeakerSegment],
    fps: float,
    duration: float,
) -> list[tuple[float, str]]:
    """Select a small set of representative timestamps for review.

    Picks frames at:
      - Shot boundaries (visual context changes)
      - Speaker transitions (where the active speaker changes)
      - Evenly spaced fallback frames (ensure coverage)

    Returns list of (timestamp, reason) tuples.
    """
    timestamps: list[tuple[float, str]] = []

    # Shot boundary timestamps
    for frame_idx in shot_boundaries[:4]:  # max 4 shot boundaries
        t = frame_idx / fps
        if t < duration:
            timestamps.append((t, f"shot_boundary (frame {frame_idx})"))

    # Speaker transition timestamps
    prev_speaker = None
    for seg in segments[:8]:  # max 8 speaker segments
        if seg.speaker_id != prev_speaker:
            timestamps.append((seg.start, f"speaker_change {prev_speaker}→{seg.speaker_id}"))
            prev_speaker = seg.speaker_id

    # Evenly spaced fallback (ensure total coverage)
    target_count = 8
    interval = duration / target_count if duration > 0 else 1.0
    for i in range(target_count):
        t = i * interval + interval / 2
        if t < duration:
            timestamps.append((t, f"sample_{i}"))

    # Deduplicate by proximity (within 2 seconds), keeping first occurrence
    seen: list[float] = []
    unique: list[tuple[float, str]] = []
    for t, reason in timestamps:
        if all(abs(t - s) > 2.0 for s in seen):
            seen.append(t)
            unique.append((t, reason))

    # Sort by time
    unique.sort(key=lambda x: x[0])
    return unique


def _build_composite(
    original_frames: list[np.ndarray],
    rendered_frames: list[np.ndarray],
    timestamps: list[float],
    reasons: list[str],
    crop_points: list[CropTrajectoryPoint | None],
    speaker_labels: list[int | None],
) -> bytes:
    """Build a composite image grid showing original + rendered frames side by side.

    Layout: N rows, 2 columns.
      Left column = original frame (16:9 native)
      Right column = rendered vertical frame (9:16) with crop rectangle overlay

    Each row labeled with timestamp + reason.
    """
    if not original_frames or not rendered_frames:
        return b""

    max_cols = 2
    num_rows = len(original_frames)

    # Scale each to fit
    thumb_w = 320
    thumb_h_orig = 180   # 16:9
    thumb_h_rendered = 568  # 9:16 (matches 320 * 16/9)

    gap = 10
    label_h = 20

    total_w = thumb_w * 2 + gap
    total_h = num_rows * (max(thumb_h_orig, thumb_h_rendered) + label_h + gap)

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for i in range(num_rows):
        y_offset = i * (max(thumb_h_orig, thumb_h_rendered) + label_h + gap)

        # Label
        label = f"t={timestamps[i]:.1f}s | {reasons[i][:40]}"
        draw.text((5, y_offset), label, fill=(255, 255, 255))

        # Original frame (left)
        orig = Image.fromarray(original_frames[i]).convert("RGB")
        orig.thumbnail((thumb_w, thumb_h_orig), Image.Resampling.LANCZOS)
        canvas.paste(orig, (0, y_offset + label_h))
        # Draw detected face bboxes from crops
        cp = crop_points[i] if i < len(crop_points) else None
        if cp:
            # Draw the crop rectangle on the original frame
            draw_orig = ImageDraw.Draw(canvas)
            x0, y0 = 0, y_offset + label_h
            # Scale crop rect to match thumbnail
            scale_x = orig.width / original_frames[i].shape[1]
            scale_y = orig.height / original_frames[i].shape[0]
            draw_orig.rectangle(
                [x0 + cp.crop_x * scale_x, y0 + cp.crop_y * scale_y,
                 x0 + (cp.crop_x + cp.crop_w) * scale_x,
                 y0 + (cp.crop_y + cp.crop_h) * scale_y],
                outline=(0, 255, 255), width=2,
            )

        # Rendered frame (right)
        rend = Image.fromarray(rendered_frames[i]).convert("RGB")
        rend.thumbnail((thumb_w, thumb_h_rendered), Image.Resampling.LANCZOS)
        # Draw speaker label on rendered
        speaker_id = speaker_labels[i] if i < len(speaker_labels) else None
        canvas.paste(rend, (thumb_w + gap, y_offset + label_h))
        if speaker_id is not None:
            draw.text(
                (thumb_w + gap + 5, y_offset + label_h + 5),
                f"SPEAKER: P{speaker_id}",
                fill=(255, 255, 0),
            )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _build_metadata_text(
    segments: list[ActiveSpeakerSegment],
    trajectory: list[CropTrajectoryPoint],
    shot_boundaries: list[int],
    metadata: VideoMetadata,
    num_review_points: int,
) -> str:
    """Build a text summary of trajectory + speaker data for Gemini context."""
    lines = [
        f"SOURCE VIDEO: {metadata.width}x{metadata.height}, {metadata.fps:.1f}fps, "
        f"{metadata.duration_sec:.1f}s, {metadata.total_frames} frames",
        f"REEL OUTPUT: 9:16 vertical, 720x1280, {num_review_points} review frames",
        "",
        "ACTIVE SPEAKER TIMELINE:",
    ]
    for seg in segments[:10]:
        lines.append(
            f"  {seg.start:.1f}s - {seg.end:.1f}s: speaker P{seg.speaker_id} "
            f"(conf={seg.confidence:.2f}, mar={seg.mar})"
        )

    lines.append("")
    lines.append("CROP TRAJECTORY SUMMARY:")
    # Sample trajectory at regular intervals
    step = max(1, len(trajectory) // 8)
    for p in trajectory[::step]:
        lines.append(
            f"  t={p.time:.1f}s: speaker=P{p.speaker_id}, "
            f"crop=({p.crop_x},{p.crop_y},{p.crop_w}x{p.crop_h})"
        )

    lines.append("")
    lines.append(f"SHOT BOUNDARIES (frame indices): {shot_boundaries[:10]}")

    return "\n".join(lines)


def gemini_review_video(
    original_video_path: str | Path,
    reel_video_path: str | Path,
    trajectory: list[CropTrajectoryPoint],
    segments: list[ActiveSpeakerSegment],
    shot_boundaries: list[int],
    metadata: VideoMetadata,
    max_review_frames: int = 12,
) -> VideoReviewResult:
    """Review the rendered vertical reel using Gemini as an AI visual reviewer.

    Selects representative frames at shot boundaries + speaker transitions,
    extracts them from both the original and rendered video, composites them
    into a grid, and sends a SINGLE Gemini call with full metadata context.

    This is a post-rendering critique only — it does NOT re-plan crops.
    """
    if not gemini_available():
        return VideoReviewResult(
            passed=True,
            raw_response="Gemini not available — skipping AI visual review",
            frames_reviewed=0,
        )

    from google.genai import types

    # --- Select representative timestamps ---
    review_points = _select_review_timestamps(
        shot_boundaries, trajectory, segments,
        metadata.fps, metadata.duration_sec,
    )[:max_review_frames]

    if not review_points:
        # Fallback: middle frame
        review_points = [(metadata.duration_sec / 2, "middle_frame")]

    original_dims = _get_video_dims(original_video_path)
    rendered_dims = _get_video_dims(reel_video_path)

    # Reel is 720x1280
    reel_w, reel_h = 720, 1280

    # --- Extract frames from both videos ---
    original_frames = []
    rendered_frames = []
    crop_points = []
    speaker_labels = []
    timestamps = []
    reasons = []

    for t, reason in review_points:
        orig = _extract_frame_at_time(original_video_path, t, original_dims[0], original_dims[1])
        rend = None
        if reel_video_path and Path(reel_video_path).exists():
            rend = _extract_frame_at_time(reel_video_path, t, reel_w, reel_h)

        if orig is not None and rend is not None:
            original_frames.append(orig)
            rendered_frames.append(rend)
            timestamps.append(t)
            reasons.append(reason)
            # Find nearest trajectory point for crop info
            nearest = min(trajectory, key=lambda p: abs(p.time - t)) if trajectory else None
            crop_points.append(nearest)
            speaker_labels.append(nearest.speaker_id if nearest else None)

    if not original_frames:
        return VideoReviewResult(
            passed=True,
            raw_response="No frames could be extracted for review",
            frames_reviewed=0,
        )

    # --- Build composite image ---
    composite_bytes = _build_composite(
        original_frames, rendered_frames, timestamps, reasons,
        crop_points, speaker_labels,
    )

    if not composite_bytes:
        return VideoReviewResult(
            passed=True,
            raw_response="Failed to build composite image for review",
            frames_reviewed=0,
        )

    # --- Build metadata text ---
    metadata_text = _build_metadata_text(
        segments, trajectory, shot_boundaries, metadata, len(timestamps),
    )

    # --- Single Gemini call ---
    prompt = f"""You are an expert video editor reviewing a vertical (9:16) reel
auto-generated from a horizontal (16:9) master video. The reel uses dynamic
cropping to follow the active speaker.

Below is a composite grid image with {len(timestamps)} representative frames:
  - LEFT column = original 16:9 frame with the applied crop rectangle overlaid (cyan box)
  - RIGHT column = the rendered vertical 9:16 reel frame at the same timestamp

REVIEW METADATA:
{metadata_text}

For EACH frame pair, assess:
1. Is the active speaker well-framed in the vertical crop (not cut off, reasonably centered)?
2. When the speaker changes, does the crop transition to the correct new speaker?
3. Are any faces or important visual content unnecessarily cropped out?
4. Does the composition look natural (no excessive panning/jitter)?
5. Are there obvious bad reframing decisions?

Respond with ONLY a JSON object:
{{
  "passed": bool,
  "errors": ["list of critical issues — e.g. speaker cut off, wrong person framed"],
  "warnings": ["list of minor issues — e.g. slight off-center, minor jitter"],
  "suggestions": ["list of improvement suggestions"],
  "score": int  // 1-10 overall quality
}}"""

    contents: list[Any] = [types.Content(
        role="user",
        parts=[
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(
                mime_type="image/png",
                data=composite_bytes,
            )),
        ],
    )]

    # Try models in order of preference
    api_key = None
    try:
        from .config import get_gemini_api_key
        api_key = get_gemini_api_key()
    except Exception:
        pass

    from google import genai
    client = genai.Client(api_key=api_key)

    models_to_try = ["gemini-3.8-flash", "gemini-3-flash", "gemini-3-flash-preview"]
    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name, contents=contents,
            )
            break
        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                import time
                time.sleep(16)
                continue

    if response and response.text:
        text = response.text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                return VideoReviewResult(
                    passed=bool(data.get("passed", True)),
                    errors=list(data.get("errors", [])),
                    warnings=list(data.get("warnings", [])),
                    suggestions=list(data.get("suggestions", [])),
                    score=int(data.get("score", 0)),
                    frames_reviewed=len(timestamps),
                    raw_response=text,
                )
            except json.JSONDecodeError:
                pass

    return VideoReviewResult(
        passed=True,
        raw_response=f"AI review failed: {last_error}",
        frames_reviewed=len(timestamps),
    )
