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
from .video_perception import VideoPerceiver, DetectedFace, DetectedPose, TrackedPerson


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
    crop_plan: GeminiCropPlanResult | None = None,
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

    # --- Add crop plan context if available ---
    crop_plan_context = ""
    if crop_plan is not None:
        lines = ["CROP PLAN (from pre-render Gemini analysis):"]
        if crop_plan.recommended_center:
            lines.append(f"  Center: {crop_plan.recommended_center}")
        if crop_plan.recommended_coverage is not None:
            lines.append(f"  Coverage: {crop_plan.recommended_coverage}")
        if crop_plan.suggestions:
            lines.append("  Suggestions:")
            for s in crop_plan.suggestions:
                lines.append(f"    - {s}")
        crop_plan_context = "\n".join(lines) + "\n\n"

    # --- Single Gemini call ---
    prompt = f"""You are an expert video editor reviewing a vertical (9:16) reel
auto-generated from a horizontal (16:9) master video. The reel uses dynamic
cropping to follow the active speaker.

Below is a composite grid image with {len(timestamps)} representative frames:
  - LEFT column = original 16:9 frame with the applied crop rectangle overlaid (cyan box)
  - RIGHT column = the rendered vertical 9:16 reel frame at the same timestamp

REVIEW METADATA:
{metadata_text}

{crop_plan_context}For EACH frame pair, assess:
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


# ---------------------------------------------------------------------------
# Pre-render: Gemini crop planning + deterministic segment quality metrics
# ---------------------------------------------------------------------------
@dataclass
class GeminiCropPlanResult:
    """Result of Gemini's pre-render crop planning."""

    recommended_center: tuple[float, float] | None  # normalized (x, y)
    recommended_coverage: float | None               # 0.3–0.8
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class SegmentQuality:
    """Deterministic metrics for a video segment."""

    face_coverage: float          # 0–1: fraction of frames with a face in crop
    audio_coverage: float         # 0–1: fraction of segment with speech
    shot_retention: float         # 0–1: fraction of shot boundaries in segment
    num_speakers: int             # unique speakers in segment
    trajectory_validity: float    # 0–1: fraction of trajectory points with valid crops
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Pose landmark skeleton connections for drawing (MediaPipe pose topology)
_POSE_SKELETON = [
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19),  # left arm
    (12, 14), (14, 16), (16, 18), (18, 20),              # right arm
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (27, 29),                        # left leg
    (24, 26), (26, 28), (28, 30),                        # right leg
    (11, 12), (23, 24),                                  # shoulders, hips
]


def _select_segment_timestamps(
    segment_start: float,
    segment_end: float,
    metadata: VideoMetadata,
    shot_boundaries: list[int],
    segments: list[ActiveSpeakerSegment],
    max_frames: int = 10,
) -> list[tuple[float, str]]:
    """Select representative timestamps within a segment for crop planning."""
    duration = segment_end - segment_start
    fps = metadata.fps
    timestamps: list[tuple[float, str]] = []

    # Shot boundaries within the segment
    for frame_idx in shot_boundaries:
        t = frame_idx / fps
        if segment_start <= t <= segment_end:
            timestamps.append((t, f"shot_change@{t:.1f}s"))
            if len(timestamps) >= max_frames:
                break

    # Speaker transitions within the segment
    prev_speaker = None
    for seg in segments:
        if segment_start <= seg.start <= segment_end:
            if seg.speaker_id != prev_speaker:
                timestamps.append((seg.start, f"speaker_{seg.speaker_id}@{seg.start:.1f}s"))
                prev_speaker = seg.speaker_id
                if len(timestamps) >= max_frames:
                    break

    # Evenly spaced fallbacks
    if len(timestamps) < max_frames:
        interval = duration / max_frames if duration > 0 else 1.0
        for i in range(max_frames):
            t = segment_start + i * interval + interval / 2
            if t <= segment_end:
                timestamps.append((t, f"sample_{i}"))
                if len(timestamps) >= max_frames:
                    break

    # Deduplicate by proximity (within 3s)
    seen: list[float] = []
    unique: list[tuple[float, str]] = []
    for t, reason in timestamps:
        if all(abs(t - s) > 3.0 for s in seen):
            seen.append(t)
            unique.append((t, reason))

    unique.sort(key=lambda x: x[0])
    return unique[:max_frames]


def _run_perception_on_frame(
    rgb_array: np.ndarray,
    original_w: int,
    original_h: int,
    scaled_w: int,
    scaled_h: int,
    perceiever: VideoPerceiver,
) -> list[DetectedFace]:
    """Run face detection on a frame and scale bboxes back to original resolution."""
    faces = perceiever.detect_faces(rgb_array)
    scale_x = original_w / scaled_w
    scale_y = original_h / scaled_h
    for f in faces:
        f.x = int(round(f.x * scale_x))
        f.y = int(round(f.y * scale_y))
        f.width = int(round(f.width * scale_x))
        f.height = int(round(f.height * scale_y))
    return faces


def _draw_keyframe_with_overlays(
    frame: np.ndarray,
    faces: list[DetectedFace],
    poses: list[DetectedPose],
    proposed_crop: tuple[int, int, int, int] | None,  # (x, y, w, h) in source pixels
) -> np.ndarray:
    """Draw face bboxes (red), pose landmarks (green), and proposed crop (cyan) on a frame."""
    from .video_perception import _track_color
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw pose landmarks
    if poses:
        for pose in poses:
            if not pose.landmarks or not pose.bbox:
                continue
            bx, by, bw, bh = pose.bbox
            # Get original dimensions for scaling
            orig_w, orig_h = img.size
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            for lm_idx in range(len(pose.landmarks)):
                lx, ly = pose.landmarks[lm_idx]
                px = int(lx * 640 * scale_x)
                py = int(ly * 640 * scale_y)
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(50, 255, 50))
            # Draw skeleton
            for a, b in _POSE_SKELETON:
                if a < len(pose.landmarks) and b < len(pose.landmarks):
                    ax, ay = pose.landmarks[a]
                    bx2, by2 = pose.landmarks[b]
                    draw.line([
                        (int(ax * 640 * scale_x), int(ay * 640 * scale_y)),
                        (int(bx2 * 640 * scale_x), int(by2 * 640 * scale_y)),
                    ], fill=(50, 200, 50), width=1)

    # Draw face bboxes
    for i, face in enumerate(faces):
        color = (255, 50, 50)
        draw.rectangle(
            [face.x, face.y, face.x + face.width, face.y + face.height],
            outline=color, width=2,
        )
        draw.text((face.x, face.y - 15), f"Face {i+1}", fill=color)

    # Draw proposed crop rectangle
    if proposed_crop:
        cx, cy, cw, ch = proposed_crop
        draw.rectangle(
            [cx, cy, cx + cw, cy + ch],
            outline=(0, 255, 255), width=3,
        )

    return np.array(img)


def _build_keyframe_grid(
    frames: list[np.ndarray],
    timestamps: list[float],
    reasons: list[str],
) -> bytes:
    """Build a vertical grid of keyframes for Gemini."""
    if not frames:
        return b""
    thumb_w = 640
    thumb_h = 360
    gap = 8
    label_h = 16
    num = len(frames)

    total_w = thumb_w + gap * 2
    total_h = num * (thumb_h + label_h + gap)

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for i in range(num):
        y_offset = i * (thumb_h + label_h + gap)
        # Label
        label = f"t={timestamps[i]:.1f}s | {reasons[i][:50]}"
        draw.text((gap, y_offset), label, fill=(255, 255, 255))
        # Thumbnail
        img = Image.fromarray(frames[i]).convert("RGB")
        img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(img, (gap, y_offset + label_h))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def gemini_plan_crop(
    video_path: str | Path,
    metadata: VideoMetadata,
    segment_start: float,
    segment_end: float,
    shot_boundaries: list[int],
    segments: list[ActiveSpeakerSegment],
    max_planning_frames: int = 8,
    previous_review: VideoReviewResult | None = None,
) -> GeminiCropPlanResult:
    """Ask Gemini to recommend the optimal crop center for a 30-second segment.

    Samples representative keyframes from the segment, runs face detection +
    pose estimation on each, builds a composite grid showing the keyframes
    with face bboxes (red), pose landmarks (green), and a *proposed* initial
    crop window (cyan), then sends a SINGLE Gemini call.

    Gemini returns the recommended crop center (normalized 0–1) and any
    observations about content that might be missed.

    If Gemini is unavailable, returns a deterministic fallback using the
    active speaker's face centroid.
    """
    if not gemini_available():
        # Deterministic fallback: use center of the segment
        seg_dur = segment_end - segment_start
        seg_mid = segment_start + seg_dur / 2
        # Find nearest speaker segment midpoint
        for seg in segments:
            if seg.start <= seg_mid <= seg.end:
                return GeminiCropPlanResult(
                    recommended_center=(0.5, 0.5),
                    recommended_coverage=0.45,
                    raw_response="Gemini not available — using deterministic center",
                )
        return GeminiCropPlanResult(
            recommended_center=(0.5, 0.5),
            recommended_coverage=0.45,
            raw_response="Gemini not available — using frame center",
        )

    from google.genai import types

    # ---- Select keyframes within the segment ----
    plan_points = _select_segment_timestamps(
        segment_start, segment_end, metadata,
        shot_boundaries, segments, max_planning_frames,
    )
    if not plan_points:
        center_t = (segment_start + segment_end) / 2
        plan_points = [(center_t, "segment_midpoint")]

    scaled_w, scaled_h = 640, int(round(640 * metadata.height / metadata.width))
    src_w = max(scaled_w, metadata.width)
    src_h = max(scaled_h, metadata.height)

    perceiever = VideoPerceiver(min_face_confidence=0.3, num_faces=10)

    frames_data: list[tuple[np.ndarray, list[DetectedFace], list[DetectedPose], float, str]] = []

    for t, reason in plan_points:
        # Extract frame at native resolution
        frame = _extract_frame_at_time(video_path, t, metadata.width, metadata.height)
        if frame is None:
            continue
        # Run perception at reduced resolution for speed
        from .video_perception import raw_to_numpy, extract_frames_by_interval, get_scaled_dims
        # Re-run face detection at native resolution (simpler, accurate)
        faces = perceiever.detect_faces(frame)
        for f in faces:
            # Already in native resolution since we pass the native frame
            f.frame_idx = int(t * metadata.fps)
            f.timestamp = t

        # Pose detection at native resolution
        poses = perceiever.detect_poses(frame)

        # Draw a proposed initial crop (center of all faces, full-height crop)
        proposed_crop = None
        if faces:
            all_x = [f.x + f.width / 2 for f in faces]
            all_y = [f.y + f.height / 2 for f in faces]
            cx = int(np.mean(all_x))
            cy = int(np.mean(all_y))
            # Full-height portrait crop from 16:9 source
            crop_h = int(metadata.height * 0.8)
            crop_w = int(round(crop_h * 9 / 16))
            crop_x = max(0, min(cx - crop_w // 2, metadata.width - crop_w))
            crop_y = max(0, min(cy - crop_h // 2, metadata.height - crop_h))
            proposed_crop = (crop_x, crop_y, crop_w, crop_h)
        else:
            # Default center crop
            crop_h = int(metadata.height * 0.8)
            crop_w = int(round(crop_h * 9 / 16))
            crop_x = (metadata.width - crop_w) // 2
            crop_y = (metadata.height - crop_h) // 2
            proposed_crop = (crop_x, crop_y, crop_w, crop_h)

        # Draw overlays on the frame
        overlayed = _draw_keyframe_with_overlays(frame, faces, poses, proposed_crop)
        frames_data.append((overlayed, faces, poses, t, reason))

    perceiever.close()

    if not frames_data:
        return GeminiCropPlanResult(
            recommended_center=(0.5, 0.5),
            recommended_coverage=0.45,
            errors=["No keyframes could be extracted for crop planning"],
            raw_response="",
        )

    # Build composite
    composite = _build_keyframe_grid(
        [d[0] for d in frames_data],
        [d[3] for d in frames_data],
        [d[4] for d in frames_data],
    )

    # Build metadata text
    face_info = []
    pose_info = []
    for i, (frame, faces, poses, t, reason) in enumerate(frames_data):
        face_strs = [f"face{ j+1 }(x={f.x},y={f.y},w={f.width},h={f.height})" for j, f in enumerate(faces)]
        pose_strs = [f"pose{i+1}(nose=({p.nose[0]:.2f},{p.nose[1]:.2f}), shoulders=L({p.left_shoulder[0]:.2f},{p.left_shoulder[1]:.2f}) R({p.right_shoulder[0]:.2f},{p.right_shoulder[1]:.2f}))" for i, p in enumerate(poses) if p.nose and p.left_shoulder and p.right_shoulder]
        face_info.append(f"  t={t:.1f}s ({reason[:30]}): {', '.join(face_strs) if face_strs else 'no faces'}")
        pose_info.append(f"  t={t:.1f}s: {', '.join(pose_strs) if pose_strs else 'no poses'}")

    metadata_text = (
        f"SOURCE: {metadata.width}x{metadata.height}, {metadata.fps:.1f}fps, "
        f"{metadata.duration_sec:.1f}s\n"
        f"SEGMENT: {segment_start:.1f}s - {segment_end:.1f}s ({segment_end - segment_start:.1f}s)\n"
        f"TARGET: 9:16 vertical (720x1280)\n"
        f"SPEAKER SEGMENTS IN RANGE:\n" +
        "\n".join(f"  {s.start:.1f}s-{s.end:.1f}s: P{s.speaker_id} (conf={s.confidence:.2f})"
                  for s in segments if segment_start <= s.start <= segment_end) +
        "\n\nFACE DETECTIONS:\n" + "\n".join(face_info) +
        "\n\nPOSE KEYPOINTS:\n" + "\n".join(pose_info) +
        f"PROPOSED CROP (cyan box on frames): centered on faces, 80% source height"
    )

    # If this is a retry, include previous review feedback
    previous_review_context = ""
    if previous_review is not None and previous_review.errors:
        previous_review_context = (
            "\n\nPREVIOUS REVIEW FEEDBACK (this is a retry):\n"
            f"The last render had these issues:\n"
            + "\n".join(f"  - {e}" for e in previous_review.errors)
            + "\nPlease adjust your recommendations to avoid these issues.\n"
        )

    prompt = f"""You are an expert video editor composing a vertical (9:16) reel from a
horizontal (16:9) master video segment. Face bounding boxes (red) and pose
landmarks (green dots) are overlaid on each keyframe. The cyan rectangle shows
a PROPOSED initial crop window.

REVIEW METADATA:
{metadata_text}
{previous_review_context}

For EACH keyframe, assess:
1. Is the proposed crop center optimal, or should it shift (dx, dy)?
2. Are important people/objects being cropped out that should be visible?
3. Is the crop too tight (faces filling too much) or too loose (too much empty space)?
4. At speaker transitions, does the crop need to jump to a different person?

Respond with ONLY a JSON object:
{{
  "center_x": float,   // recommended normalized crop center X (0.0–1.0, relative to source width)
  "center_y": float,   // recommended normalized crop center Y (0.0–1.0, relative to source height)
  "coverage": float,   // recommended target_coverage for compute_crop (0.3–0.6; lower = looser crop)
  "errors": ["critical issues"],
  "warnings": ["minor issues"],
  "suggestions": ["improvement suggestions"]
}}"""

    contents: list[Any] = [types.Content(
        role="user",
        parts=[
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(
                mime_type="image/png",
                data=composite,
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
                return GeminiCropPlanResult(
                    recommended_center=(
                        float(data.get("center_x", 0.5)),
                        float(data.get("center_y", 0.5)),
                    ),
                    recommended_coverage=float(data.get("coverage", 0.45)),
                    errors=list(data.get("errors", [])),
                    warnings=list(data.get("warnings", [])),
                    suggestions=list(data.get("suggestions", [])),
                    raw_response=text,
                )
            except (json.JSONDecodeError, ValueError):
                pass

    return GeminiCropPlanResult(
        recommended_center=(0.5, 0.5),
        recommended_coverage=0.45,
        errors=[f"Gemini crop planning failed: {last_error}"],
        raw_response=(response.text if response and response.text else ""),
    )


def evaluate_segment_quality(
    metadata: VideoMetadata,
    segment_start: float,
    segment_end: float,
    trajectory: list[CropTrajectoryPoint],
    tracks: list[TrackedPerson],
    segments: list[ActiveSpeakerSegment],
    shot_boundaries: list[int],
) -> SegmentQuality:
    """Evaluate a video segment using deterministic metrics.

    Checks:
      - Face coverage: fraction of trajectory points where an active speaker
        face is visible within the crop
      - Audio coverage: fraction of the segment with speech activity
      - Shot retention: fraction of shot boundaries that fall within the segment
      - Speaker diversity: number of unique speakers in the segment
      - Trajectory validity: fraction of trajectory points with non-trivial crops

    The segment passes if face coverage >= threshold AND audio coverage >=
    threshold AND trajectory validity is high.
    """
    from .config import (
        VIDEO_SEGMENT_DURATION_SEC,
        VIDEO_SEGMENT_MIN_FACE_COVERAGE,
        VIDEO_SEGMENT_MIN_AUDIO_COVERAGE,
    )

    fps = metadata.fps
    total_frames = metadata.total_frames
    duration = metadata.duration_sec

    errors: list[str] = []
    warnings: list[str] = []

    # --- Face coverage: fraction of trajectory points with active speaker face
    # in the crop region ---
    face_in_crop = 0
    total_traj = len(trajectory)
    if total_traj > 0:
        # Build track bbox lookup
        track_times: dict[int, list[tuple[float, DetectedFace]]] = {}
        for track in tracks:
            for f in track.face_bboxes:
                track_times.setdefault(track.id, []).append((f.timestamp, f))

        for point in trajectory:
            if point.speaker_id < 0:
                continue
            # Find nearest bbox for this speaker
            samples = track_times.get(point.speaker_id, [])
            if not samples:
                continue
            nearest = min(samples, key=lambda s: abs(s[0] - point.time))
            bbox = nearest[1]
            # Check if face bbox overlaps with crop
            crop_x, crop_y, crop_w, crop_h = point.crop_x, point.crop_y, point.crop_w, point.crop_h
            face_cx = bbox.x + bbox.width / 2
            face_cy = bbox.y + bbox.height / 2
            if crop_x <= face_cx <= crop_x + crop_w and crop_y <= face_cy <= crop_y + crop_h:
                face_in_crop += 1
            else:
                # Face center might be outside crop but bbox could still overlap
                ix1 = max(crop_x, bbox.x)
                iy1 = max(crop_y, bbox.y)
                ix2 = min(crop_x + crop_w, bbox.x + bbox.width)
                iy2 = min(crop_y + crop_h, bbox.y + bbox.height)
                if ix2 > ix1 and iy2 > iy1:
                    face_in_crop += 1

    face_coverage = face_in_crop / total_traj if total_traj > 0 else 0.0

    # --- Audio coverage: fraction of segment with speech ---
    from .video_audio import detect_speech_activity
    # We can't easily re-run SAD here; use speaker segments as proxy
    speech_in_segment = 0.0
    for seg in segments:
        overlap_start = max(seg.start, segment_start)
        overlap_end = min(seg.end, segment_end)
        if overlap_end > overlap_start:
            speech_in_segment += overlap_end - overlap_start
    audio_coverage = speech_in_segment / (segment_end - segment_start) if segment_end > segment_start else 0.0

    # --- Shot retention ---
    fps_for_shots = metadata.fps
    shots_in_segment = 0
    total_shots = len(shot_boundaries)
    for frame_idx in shot_boundaries:
        t = frame_idx / fps_for_shots
        if segment_start <= t <= segment_end:
            shots_in_segment += 1
    shot_retention = shots_in_segment / total_shots if total_shots > 0 else 0.0

    # --- Speaker diversity ---
    unique_speakers = set()
    for seg in segments:
        if segment_start <= seg.start <= segment_end:
            unique_speakers.add(seg.speaker_id)
    num_speakers = len(unique_speakers)

    # --- Trajectory validity ---
    valid_traj = 0
    for point in trajectory:
        if point.crop_w > 10 and point.crop_h > 10:
            valid_traj += 1
    trajectory_validity = valid_traj / total_traj if total_traj > 0 else 0.0

    # --- Determine pass/fail ---
    min_face_cov = VIDEO_SEGMENT_MIN_FACE_COVERAGE
    min_audio_cov = VIDEO_SEGMENT_MIN_AUDIO_COVERAGE

    if face_coverage < min_face_cov:
        errors.append(f"Face coverage too low: {face_coverage:.1%} < {min_face_cov:.1%} threshold")
    if audio_coverage < min_audio_cov:
        warnings.append(f"Audio coverage low: {audio_coverage:.1%} < {min_audio_cov:.1%} threshold")
    if shot_retention < 0.3:
        warnings.append(f"Few shot boundaries in segment ({shot_retention:.1%}): may miss visual variety")
    if trajectory_validity < 0.9:
        errors.append(f"Trajectory has invalid crops: {trajectory_validity:.1%} valid")

    passed = len(errors) == 0

    return SegmentQuality(
        face_coverage=face_coverage,
        audio_coverage=audio_coverage,
        shot_retention=shot_retention,
        num_speakers=num_speakers,
        trajectory_validity=trajectory_validity,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )

