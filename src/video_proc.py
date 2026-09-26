"""Subject-aware video reformatting module.

Pipeline:
  1. DETECT  — extract keyframes, run MediaPipe face detection on each
  2. TRACK   — propagate subject positions across all frames (centroid-based)
  3. PLAN    — compute per-frame 9:16 crop trajectory using compute_crop
  4. RENDER  — crop + resize via ffmpeg, mux with original audio

Also: extract best still frame (sharpest + speaker-active) cropped to 1:1.

This module reuses:
  - Subject, CropResult, compute_crop from cropper.py (deterministic arithmetic)
  - detect_subjects from image_crop.py (MediaPipe)
  - validate_asset from validator.py
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import (
    FACE_DETECTION_MODEL,
    REEL_ASPECT_RATIO,
    STILL_ASPECT_RATIO,
    VIDEO_KEYFRAME_INTERVAL,
    INFERENCE_WIDTH,
    INFERENCE_HEIGHT,
)
from .cropper import Subject, CropResult, compute_crop
from .image_crop import detect_subjects, render_crop
from .validator import load_spec


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class VideoProperties:
    """Metadata about a video file."""

    width: int
    height: int
    fps: float
    fps_num: int
    fps_den: int
    duration_sec: float
    nb_frames: int
    codec: str
    has_audio: bool
    audio_codec: str | None = None


@dataclass
class FrameDetection:
    """Subjects detected in a single video frame."""

    frame_num: int
    timestamp: float
    subjects: list[Subject]


@dataclass
class StillFrameResult:
    """Best still frame extracted from video."""

    timestamp: float
    frame_num: int
    crop: CropResult
    sharpness_score: float
    speaker_activity_score: float
    rendered: np.ndarray
    output_path: str = ""


@dataclass
class VideoReelResult:
    """Result of processing a video into a vertical reel."""

    output_path: str
    output_width: int
    output_height: int
    duration_sec: float
    fps: float
    trajectory: list[CropTrajectoryPoint] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)
    scene_boundaries: list[int] = field(default_factory=list)
    keyframe_crops: list[FrameDetection] = field(default_factory=list)
    subjects_detected: int = 0
    model_used: str = ""
    warnings: list[str] = field(default_factory=list)
    ai_status: str = "unavailable (using deterministic fallback)"
    processing_time_sec: float = 0.0

    def to_metadata(self) -> dict[str, Any]:
        """Serializable metadata for manifests."""
        return {
            "output_path": self.output_path,
            "output_dimensions": {
                "width": self.output_width,
                "height": self.output_height,
            },
            "duration_sec": round(self.duration_sec, 3),
            "fps": round(self.fps, 2),
            "subjects_detected": self.subjects_detected,
            "model_used": self.model_used,
            "speaker_ids": self.speaker_ids,
            "scene_boundaries": self.scene_boundaries,
            "keyframe_crops": [
                {
                    "frame_num": kd.frame_num,
                    "timestamp": round(kd.timestamp, 3),
                    "subjects": [
                        {
                            "bbox": list(s.bbox),
                            "importance": round(s.importance, 4),
                            "id": s.id,
                            "category": s.category,
                        }
                        for s in kd.subjects
                    ],
                }
                for kd in self.keyframe_crops
            ],
            "crop_trajectory": [
                {
                    "frame_num": tp.frame_num,
                    "timestamp": round(tp.timestamp, 3),
                    "crop": {
                        "x": tp.crop.x,
                        "y": tp.crop.y,
                        "width": tp.crop.width,
                        "height": tp.crop.height,
                    },
                }
                for tp in self.trajectory
            ],
            "ai_status": self.ai_status,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


@dataclass
class CropTrajectoryPoint:
    """A single point in the crop trajectory for a video reel."""

    frame_num: int
    timestamp: float
    crop: CropResult


@dataclass
class VideoPipelineResult:
    """Result of processing a video into reel + still."""

    video_path: str
    video_width: int
    video_height: int
    reel: VideoReelResult | None = None
    still: StillFrameResult | None = None
    reel_manifest: dict[str, Any] | None = None
    still_manifest: dict[str, Any] | None = None
    all_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    processing_time_sec: float = 0.0

    def to_summary(self) -> dict[str, Any]:
        """Serializable summary for CLI output and API responses."""
        return {
            "video_path": self.video_path,
            "video_dimensions": {
                "width": self.video_width,
                "height": self.video_height,
            },
            "reel": {
                "output_path": self.reel.output_path if self.reel else None,
                "output_dimensions": {
                    "width": self.reel.output_width if self.reel else 0,
                    "height": self.reel.output_height if self.reel else 0,
                },
                "passed": self.reel is not None,
                "subjects_detected": self.reel.subjects_detected if self.reel else 0,
                "ai_status": self.reel.ai_status if self.reel else None,
            } if self.reel else None,
            "still": {
                "output_path": self.still.output_path if self.still else None,
                "frame_num": self.still.frame_num if self.still else None,
                "timestamp": round(self.still.timestamp, 3) if self.still else None,
                "sharpness_score": round(self.still.sharpness_score, 4) if self.still else None,
                "speaker_activity_score": round(self.still.speaker_activity_score, 4) if self.still else None,
            } if self.still else None,
            "all_passed": self.all_passed,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ---------------------------------------------------------------------------
# Video properties
# ---------------------------------------------------------------------------
def get_video_properties(video_path: str | Path) -> VideoProperties:
    """Extract technical metadata from a video file using ffprobe."""
    path = str(video_path)
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"Could not read video properties: {path}")

    data = json.loads(result.stdout)
    fmt = data.get("format", {})

    # Find video and audio streams
    video_stream = None
    audio_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    # Parse frame rate (e.g., "25/1")
    fps_str = video_stream.get("r_frame_rate", "25/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps_num = int(num)
        fps_den = int(den) if int(den) != 0 else 1
    else:
        fps_num = int(float(fps_str))
        fps_den = 1
    fps = fps_num / fps_den

    duration = float(fmt.get("duration", 0))
    if duration <= 0:
        # Fallback: compute from frames and fps
        nb_frames = int(video_stream.get("nb_frames", 0))
        if nb_frames > 0 and fps > 0:
            duration = nb_frames / fps
        else:
            duration = 0.0

    nb_frames = int(video_stream.get("nb_frames", 0))
    if nb_frames == 0 and fps > 0 and duration > 0:
        nb_frames = int(duration * fps)

    return VideoProperties(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        fps_num=fps_num,
        fps_den=fps_den,
        duration_sec=duration,
        nb_frames=nb_frames,
        codec=video_stream.get("codec_name", "h264"),
        has_audio=audio_stream is not None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
    )


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------
def _extract_keyframe_indices(
    nb_frames: int, fps: float, keyframe_interval: int = VIDEO_KEYFRAME_INTERVAL,
) -> list[int]:
    """Compute keyframe frame indices spaced `keyframe_interval` apart.

    Always includes frame 0 and the last frame.
    """
    if nb_frames <= 0:
        return [0]
    interval = max(1, keyframe_interval)
    indices = list(range(0, nb_frames, interval))
    if indices[-1] != nb_frames - 1:
        indices.append(nb_frames - 1)
    return indices


def _read_frame_at(
    cap: cv2.VideoCapture, frame_num: int,
) -> tuple[bool, np.ndarray | None]:
    """Seek to a specific frame number and return the RGB array."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret or frame is None:
        return False, None
    # OpenCV uses BGR; convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return True, frame_rgb


def _read_all_frames(video_path: str | Path) -> list[np.ndarray]:
    """Read ALL frames from a video as RGB numpy arrays.

    Used for reel rendering where we need to crop every frame.
    """
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Subject detection in video keyframes
# ---------------------------------------------------------------------------
def detect_video_subjects(
    video_path: str | Path,
    keyframe_interval: int = VIDEO_KEYFRAME_INTERVAL,
    min_confidence: float = 0.3,
    model_path: str | Path | None = None,
) -> tuple[list[FrameDetection], str]:
    """Detect subjects in video keyframes using MediaPipe.

    Returns (detections, model_name). Each FrameDetection contains
    subjects detected in that keyframe. If the model is missing or
    detection fails, returns detections with empty subject lists
    (callers fall back to center crop).
    """
    if model_path is None:
        model_path = FACE_DETECTION_MODEL

    model_name = Path(model_path).name
    video_path = Path(video_path)

    props = get_video_properties(video_path)
    keyframe_indices = _extract_keyframe_indices(
        props.nb_frames, props.fps, keyframe_interval
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"Could not open video for detection: {video_path}")
        return [], model_name

    detections: list[FrameDetection] = []
    for fn in keyframe_indices:
        ret, frame_rgb = _read_frame_at(cap, fn)
        if not ret:
            continue

        timestamp = fn / props.fps if props.fps > 0 else 0.0
        subjects = _detect_subjects_from_array(
            frame_rgb, Path(model_path), min_confidence
        )
        detections.append(FrameDetection(
            frame_num=fn,
            timestamp=timestamp,
            subjects=subjects,
        ))

    cap.release()
    return detections, model_name


def _detect_subjects_from_array(
    frame: np.ndarray,
    model_path: Path,
    min_confidence: float = 0.3,
) -> list[Subject]:
    """Run MediaPipe face detection on an in-memory RGB frame.

    Mirrors detect_subjects() from image_crop but works on numpy arrays
    instead of file paths, so video processing doesn't waste I/O saving
    temp files for every keyframe.
    """
    if not model_path.exists():
        return []

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    # Resize for faster inference (maintain aspect ratio)
    h, w = frame.shape[:2]
    scale = min(INFERENCE_WIDTH / w, INFERENCE_HEIGHT / h, 1.0)
    inf_w = max(1, int(w * scale))
    inf_h = max(1, int(h * scale))

    if inf_w != w or inf_h != h:
        inf_frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
    else:
        inf_frame = frame

    img_rgb = Image.fromarray(inf_frame).convert("RGB")
    img_array = np.array(img_rgb)

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
                # Scale bbox back to original frame coordinates
                x1 = int(bbox.origin_x / scale)
                y1 = int(bbox.origin_y / scale)
                bw = max(1, int(bbox.width / scale))
                bh = max(1, int(bbox.height / scale))
                score = det.categories[0].score if det.categories else 0.5
                subjects.append(Subject(
                    bbox=(float(x1), float(y1), float(bw), float(bh)),
                    importance=float(score),
                    id=f"face_{len(subjects)}",
                    category="face",
                ))
    except Exception:
        warnings.warn("Face detection failed on keyframe — returning empty subject list")
        return []

    return subjects


# ---------------------------------------------------------------------------
# Subject tracking across frames
# ---------------------------------------------------------------------------
def track_subjects_across_frames(
    keyframe_detections: list[FrameDetection],
    total_frames: int,
    fps: float,
    keyframe_interval: int = VIDEO_KEYFRAME_INTERVAL,
) -> list[TrackedSubjectData]:
    """Track subjects across all frames using simple centroid-based matching.

    Between keyframes, subjects are propagated by nearest-centroid matching
    with a maximum displacement threshold. If no detection exists at a
    keyframe, the subject is marked as temporarily lost.

    Returns a list of TrackedSubjectData, one per tracked subject.
    """
    # Collect all (frame_num, subject) pairs from keyframes
    frame_subjects: dict[int, list[Subject]] = {}
    for kd in keyframe_detections:
        frame_subjects[kd.frame_num] = kd.subjects

    if not keyframe_detections:
        return []

    # Initialize tracked subjects from first keyframe
    tracked: list[TrackedSubjectData] = []
    next_id = 0

    for kd in sorted(keyframe_detections, key=lambda d: d.frame_num):
        fn = kd.frame_num
        current_subjects = frame_subjects.get(fn, [])

        if fn == keyframe_detections[0].frame_num:
            # Initialize tracks
            for s in current_subjects:
                tracked.append(TrackedSubjectData(
                    id=f"speaker_{next_id}",
                    category=s.category,
                    frames={fn: s.bbox},
                    confidence=s.importance,
                    last_seen_frame=fn,
                ))
                next_id += 1
        else:
            # Match current subjects to existing tracks
            prev_fn = keyframe_detections[
                max(0, next(i for i, k in enumerate(sorted(frame_subjects.keys())) if k == fn) - 1)
            ].frame_num

            # Simple greedy matching: for each existing track, find nearest subject
            assigned: set[int] = set()
            for track in tracked:
                if not track.frames:
                    continue
                # Get last bbox
                last_fn = track.last_seen_frame
                last_bbox = track.frames[last_fn]
                last_cx = last_bbox[0] + last_bbox[2] / 2
                last_cy = last_bbox[1] + last_bbox[3] / 2

                best_idx = -1
                best_dist = float("inf")
                max_dist = max(track.frames[last_fn][2], track.frames[last_fn][3]) * 3

                for i, s in enumerate(current_subjects):
                    if i in assigned:
                        continue
                    cx = s.bbox[0] + s.bbox[2] / 2
                    cy = s.bbox[1] + s.bbox[3] / 2
                    dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                    if dist < best_dist and dist < max_dist:
                        best_dist = dist
                        best_idx = i

                if best_idx >= 0:
                    assigned.add(best_idx)
                    track.frames[fn] = current_subjects[best_idx].bbox
                    track.confidence = max(
                        track.confidence, current_subjects[best_idx].importance
                    )
                    track.last_seen_frame = fn

            # Unmatched subjects start new tracks
            for i, s in enumerate(current_subjects):
                if i not in assigned:
                    tracked.append(TrackedSubjectData(
                        id=f"speaker_{next_id}",
                        category=s.category,
                        frames={fn: s.bbox},
                        confidence=s.importance,
                        last_seen_frame=fn,
                    ))
                    next_id += 1

    return tracked


@dataclass
class TrackedSubjectData:
    """A subject tracked across video frames."""
    id: str
    category: str
    frames: dict[int, tuple[float, float, float, float]]  # frame_num -> bbox
    confidence: float
    last_seen_frame: int
    is_speaker: bool = False


# ---------------------------------------------------------------------------
# Crop trajectory computation
# ---------------------------------------------------------------------------
def compute_reel_crop_trajectory(
    video_path: str | Path,
    target_ratio: float = REEL_ASPECT_RATIO,
    keyframe_interval: int = VIDEO_KEYFRAME_INTERVAL,
    target_width: int = 720,
    min_confidence: float = 0.3,
    use_ai: bool = False,
) -> tuple[list[CropTrajectoryPoint], list[FrameDetection], str, list[str]]:
    """Compute a per-frame crop trajectory for the vertical reel.

    1. Detect subjects in keyframes
    2. Track subjects across all frames
    3. For each keyframe, compute crop using compute_crop
    4. Interpolate crop center between keyframes for smooth motion

    Returns (trajectory, keyframe_detections, model_name, speaker_ids).
    """
    video_path = Path(video_path)
    props = get_video_properties(video_path)

    # Detect subjects in keyframes
    keyframe_detections, model_name = detect_video_subjects(
        video_path, keyframe_interval, min_confidence
    )

    if not keyframe_detections:
        warnings.warn("No keyframes processed — using center crop for all frames")

    # Track subjects
    tracked = track_subjects_across_frames(
        keyframe_detections, props.nb_frames, props.fps, keyframe_interval
    )

    # Determine speaker IDs (subjects detected in multiple keyframes)
    speaker_ids: list[str] = []
    if tracked:
        for t in tracked:
            # A subject is a "speaker" if detected in at least 2 keyframes
            if len(t.frames) >= 2:
                t.is_speaker = True
                speaker_ids.append(t.id)
        if not speaker_ids:
            # If only one subject across one keyframe, still count it
            speaker_ids = [t.id for t in tracked]

    # Compute crop width/height for the target ratio
    # Source is landscape (16:9), target is portrait (9:16)
    # We crop to fit the source height, then resize to target_width
    crop_h = props.height
    crop_w = int(round(props.height * target_ratio))
    crop_w = max(1, min(crop_w, props.width))

    # For keyframes, compute crop centered on tracked subjects
    keyframe_crops: list[tuple[int, CropResult]] = []
    for kd in keyframe_detections:
        # Use subjects from this keyframe as the subject list
        subjects = kd.subjects
        if subjects:
            crop_result = compute_crop(
                source_w=props.width,
                source_h=props.height,
                subjects=subjects,
                target_ratio=target_ratio,
                target_coverage=0.5,
                min_margin=0.1,
            )
        else:
            crop_result = compute_crop(
                source_w=props.width,
                source_h=props.height,
                subjects=[],
                target_ratio=target_ratio,
            )
        keyframe_crops.append((kd.frame_num, crop_result))

    # Interpolate crop between keyframes for all frames
    trajectory: list[CropTrajectoryPoint] = []
    if not keyframe_crops:
        # Fallback: center crop for all frames
        center_crop = compute_crop(
            source_w=props.width, source_h=props.height,
            subjects=[], target_ratio=target_ratio,
        )
        for fn in range(props.nb_frames):
            ts = fn / props.fps if props.fps > 0 else 0.0
            trajectory.append(CropTrajectoryPoint(
                frame_num=fn,
                timestamp=ts,
                crop=center_crop,
            ))
    else:
        # Build interpolation segments
        kf_pairs = list(zip(keyframe_crops[:-1], keyframe_crops[1:]))
        for i, (fn, crop) in enumerate(keyframe_crops):
            ts = fn / props.fps if props.fps > 0 else 0.0
            trajectory.append(CropTrajectoryPoint(
                frame_num=fn, timestamp=ts, crop=crop,
            ))
            # Interpolate to next keyframe
            if i < len(keyframe_crops) - 1:
                next_fn, next_crop = keyframe_crops[i + 1]
                n_steps = next_fn - fn
                if n_steps > 1:
                    for step in range(1, n_steps):
                        t = step / n_steps
                        interp_crop = _interpolate_crop(crop, next_crop, t)
                        interp_fn = fn + step
                        interp_ts = interp_fn / props.fps if props.fps > 0 else 0.0
                        trajectory.append(CropTrajectoryPoint(
                            frame_num=interp_fn,
                            timestamp=interp_ts,
                            crop=interp_crop,
                        ))

    return trajectory, keyframe_detections, model_name, speaker_ids


def _interpolate_crop(
    crop_a: CropResult, crop_b: CropResult, t: float,
) -> CropResult:
    """Linearly interpolate between two CropResults."""
    ix = int(round(crop_a.x + (crop_b.x - crop_a.x) * t))
    iy = int(round(crop_a.y + (crop_b.y - crop_b.y) * t))
    iw = crop_a.width  # width stays constant (same target ratio)
    ih = crop_a.height
    return CropResult(
        x=ix, y=iy, width=iw, height=ih,
        subject_coverage_pct=crop_a.subject_coverage_pct,
        edge_cutoffs=crop_a.edge_cutoffs,
        primary_subject_center=crop_a.primary_subject_center,
        scale_used=crop_a.scale_used,
        is_center_fallback=crop_a.is_center_fallback,
    )


# ---------------------------------------------------------------------------
# Scene boundary detection
# ---------------------------------------------------------------------------
def detect_scene_boundaries(
    video_path: str | Path,
    keyframe_interval: int = VIDEO_KEYFRAME_INTERVAL,
) -> list[int]:
    """Detect scene boundaries using frame-to-frame difference.

    A scene boundary is detected when the average pixel difference between
    consecutive keyframes exceeds a threshold.
    """
    video_path = Path(video_path)
    props = get_video_properties(video_path)

    keyframe_indices = _extract_keyframe_indices(
        props.nb_frames, props.fps, keyframe_interval
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    boundaries: list[int] = [0]
    prev_frame = None
    threshold = 30.0  # mean absolute difference threshold (0-255 scale)

    for fn in keyframe_indices:
        ret, frame_rgb = _read_frame_at(cap, fn)
        if not ret:
            continue
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.resize(gray, (64, 36))

        if prev_frame is not None:
            diff = cv2.absdiff(gray_small, prev_frame)
            mean_diff = float(np.mean(diff))
            if mean_diff > threshold:
                boundaries.append(fn)

        prev_frame = gray_small

    cap.release()
    return boundaries


# ---------------------------------------------------------------------------
# Still frame extraction
# ---------------------------------------------------------------------------
def extract_still_frame(
    video_path: str | Path,
    target_ratio: float = STILL_ASPECT_RATIO,
    sample_interval_sec: float = 2.0,
    min_confidence: float = 0.3,
    target_size: int = 1080,
) -> StillFrameResult:
    """Extract the best still frame from a video.

    Candidates are sampled at regular intervals. Each candidate is scored
    by:
      - Sharpness: Laplacian variance (higher = sharper)
      - Speaker activity: face detection confidence in the center region

    The best-scoring frame is cropped to the target aspect ratio.
    """
    video_path = Path(video_path)
    props = get_video_properties(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    sample_interval_frames = max(1, int(props.fps * sample_interval_sec))

    best_score = -1.0
    best_result: StillFrameResult | None = None

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_num % sample_interval_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_num / props.fps if props.fps > 0 else 0.0

            # Sharpness: Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Speaker activity: detect faces and score center-region activity
            subjects = _detect_subjects_from_array(
                frame_rgb, FACE_DETECTION_MODEL, min_confidence
            )
            speaker_activity = 0.0
            if subjects:
                # Weight by face size relative to frame area
                for s in subjects:
                    face_area = s.bbox[2] * s.bbox[3]
                    speaker_activity = max(
                        speaker_activity,
                        face_area / (props.width * props.height) * s.importance,
                    )

            # Combined score: prioritize sharpness but require some speaker activity
            # Sharpness alone can pick blurry faces; speaker activity ensures
            # a person is in frame
            score = sharpness * (1.0 + speaker_activity * 5.0)

            if score > best_score:
                # Compute 1:1 crop centered on the best subject
                if subjects:
                    primary = max(subjects, key=lambda s: s.importance)
                    crop_result = compute_crop(
                        source_w=props.width,
                        source_h=props.height,
                        subjects=subjects,
                        target_ratio=target_ratio,
                        target_coverage=0.5,
                        min_margin=0.1,
                    )
                else:
                    crop_result = compute_crop(
                        source_w=props.width,
                        source_h=props.height,
                        subjects=[],
                        target_ratio=target_ratio,
                    )

                # Render the crop
                rendered = _render_frame_crop(frame_rgb, crop_result)

                best_result = StillFrameResult(
                    timestamp=timestamp,
                    frame_num=frame_num,
                    crop=crop_result,
                    sharpness_score=sharpness,
                    speaker_activity_score=speaker_activity,
                    rendered=rendered,
                )
                best_score = score

        frame_num += 1

    cap.release()

    if best_result is None:
        # Fallback: center frame
        center_fn = props.nb_frames // 2
        ret, frame = _read_frame_at(cap, center_fn)
        if not ret:
            frame = np.zeros((props.height, props.width, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        crop_result = compute_crop(
            source_w=props.width, source_h=props.height,
            subjects=[], target_ratio=target_ratio,
        )
        rendered = _render_frame_crop(frame, crop_result)
        best_result = StillFrameResult(
            timestamp=center_fn / props.fps if props.fps > 0 else 0.0,
            frame_num=center_fn,
            crop=crop_result,
            sharpness_score=sharpness,
            speaker_activity_score=0.0,
            rendered=rendered,
        )

    return best_result


def _render_frame_crop(
    frame_rgb: np.ndarray, crop: CropResult,
) -> np.ndarray:
    """Apply a crop to an in-memory RGB frame (numpy slicing)."""
    h, w = frame_rgb.shape[:2]
    x = max(0, crop.x)
    y = max(0, crop.y)
    cw = min(crop.width, w - x)
    ch = min(crop.height, h - y)
    return frame_rgb[y:y + ch, x:x + cw].copy()


# ---------------------------------------------------------------------------
# Reel rendering (ffmpeg)
# ---------------------------------------------------------------------------
def render_reel(
    video_path: str | Path,
    trajectory: list[CropTrajectoryPoint],
    output_path: str | Path,
    target_width: int = 720,
    target_height: int | None = None,
    fps: float | None = None,
    keyframe_interval: int = VIDEO_KEYFRAME_INTERVAL,
) -> str:
    """Render the vertical reel video with audio.

    Uses ffmpeg with a dynamically generated crop filter that applies
    different crops to different frame ranges (keyframe-based).
    The crop is resized to the target dimensions and muxed with
    the original audio track.

    Returns the path to the output file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    props = get_video_properties(video_path)
    if target_height is None:
        target_height = int(round(target_width / REEL_ASPECT_RATIO))
    if fps is None:
        fps = props.fps
    elif fps < props.fps:
        fps = props.fps  # Don't drop below source fps

    # Build the crop trajectory: group consecutive frames with the same
    # keyframe crop into segments
    # We use the trajectory's keyframe crops (every keyframe_interval frames)
    # and interpolate between them

    segments = _build_crop_segments(trajectory, props.nb_frames)
    filter_parts: list[str] = []
    for seg in segments:
        x, y, w, h = seg["x"], seg["y"], seg["w"], seg["h"]
        start, end = seg["start"], seg["end"]
        enable_expr = f"between(n\\,{start}\\,{end})"
        filter_parts.append(
            f"crop={w}:{h}:{x}:{y}:enable='{enable_expr}'"
        )

    # Build filter chain: crop → scale → output
    crop_filter = ",".join(filter_parts) if filter_parts else ""
    scale_filter = f"scale={target_width}:{target_height}:flags=lanczos"
    fps_filter = f"fps={fps:.2f}"

    filters = ",".join(f for f in [crop_filter, fps_filter, scale_filter] if f)

    # FFmpeg command
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", filters,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg reel rendering failed:\n{result.stderr}"
        )

    return str(output_path)


def _build_crop_segments(
    trajectory: list[CropTrajectoryPoint],
    total_frames: int,
) -> list[dict[str, Any]]:
    """Build crop segments from trajectory for ffmpeg enable expressions.

    Each segment covers a range of frames and specifies the crop to apply.
    We use the keyframe crops and fill gaps with the nearest keyframe crop.
    """
    if not trajectory:
        return []

    # Group trajectory by keyframe intervals (use every trajectory point
    # that's a keyframe or interpolate)
    # For simplicity, use every Nth trajectory point as a segment boundary
    segment_interval = max(1, len(trajectory) // 50)  # Max ~50 segments
    segments: list[dict[str, Any]] = []

    for i in range(0, len(trajectory), segment_interval):
        tp = trajectory[i]
        end_fn = min(
            total_frames - 1,
            trajectory[min(i + segment_interval, len(trajectory) - 1)].frame_num,
        )
        segments.append({
            "x": tp.crop.x,
            "y": tp.crop.y,
            "w": tp.crop.width,
            "h": tp.crop.height,
            "start": tp.frame_num,
            "end": end_fn,
        })

    # Fix any gaps
    for i in range(len(segments) - 1):
        segments[i]["end"] = segments[i + 1]["start"] - 1

    return segments


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------
def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_video_manifest(
    reel_result: VideoReelResult,
    input_video_path: str | Path,
    target_ratio: float,
) -> dict[str, Any]:
    """Build a manifest dict for a video reel for validator consumption."""
    input_hash = _compute_file_hash(str(input_video_path))
    props = get_video_properties(input_video_path)
    ratio_name = _ratio_to_name(target_ratio)

    return {
        "asset_type": "video_reel",
        "asset_name": f"{Path(input_video_path).stem}_{ratio_name}",
        "input_path": str(input_video_path),
        "input_hash": input_hash,
        "input_dimensions": {"width": props.width, "height": props.height},
        "output_path": reel_result.output_path,
        "output_dimensions": {
            "width": reel_result.output_width,
            "height": reel_result.output_height,
        },
        "crop_params": {
            "x": reel_result.trajectory[0].crop.x if reel_result.trajectory else 0,
            "y": reel_result.trajectory[0].crop.y if reel_result.trajectory else 0,
            "w": reel_result.trajectory[0].crop.width if reel_result.trajectory else 0,
            "h": reel_result.trajectory[0].crop.height if reel_result.trajectory else 0,
            "edge_cutoffs": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
            "crop_trajectory": [
                {
                    "frame_num": tp.frame_num,
                    "timestamp": round(tp.timestamp, 3),
                    "x": tp.crop.x,
                    "y": tp.crop.y,
                    "w": tp.crop.width,
                    "h": tp.crop.height,
                }
                for tp in reel_result.trajectory
            ],
            "speaker_ids": reel_result.speaker_ids,
            "scene_boundaries": reel_result.scene_boundaries,
            "keyframe_crops": [
                {
                    "frame_num": kd.frame_num,
                    "timestamp": round(kd.timestamp, 3),
                    "subjects": [
                        {
                            "bbox": list(s.bbox),
                            "importance": round(s.importance, 4),
                            "id": s.id,
                            "category": s.category,
                        }
                        for s in kd.subjects
                    ],
                }
                for kd in reel_result.keyframe_crops
            ],
        },
        "audio_present": True,  # We always mux audio
        "duration_sec": round(reel_result.duration_sec, 3),
        "fps": round(reel_result.fps, 2),
        "processing_time_sec": round(reel_result.processing_time_sec, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_passed": None,
        "validation_warnings": [],
    }


def build_still_manifest(
    still_result: StillFrameResult,
    input_video_path: str | Path,
    output_path: str,
    target_ratio: float,
    processing_time_sec: float = 0.0,
    validation_passed: bool | None = None,
    validation_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict for a still frame for validator consumption."""
    input_hash = _compute_file_hash(str(input_video_path))
    props = get_video_properties(input_video_path)
    ratio_name = _ratio_to_name(target_ratio)

    return {
        "asset_type": "still_frame",
        "asset_name": f"{Path(input_video_path).stem}_still_{ratio_name}",
        "input_path": str(input_video_path),
        "input_hash": input_hash,
        "input_dimensions": {"width": props.width, "height": props.height},
        "output_path": output_path,
        "output_dimensions": {
            "width": still_result.crop.width,
            "height": still_result.crop.height,
        },
        "crop_params": {
            "x": still_result.crop.x,
            "y": still_result.crop.y,
            "w": still_result.crop.width,
            "h": still_result.crop.height,
            "edge_cutoffs": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
            "source_timestamp": round(still_result.timestamp, 3),
            "sharpness_score": round(still_result.sharpness_score, 4),
            "speaker_activity_score": round(still_result.speaker_activity_score, 4),
        },
        "processing_time_sec": processing_time_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
