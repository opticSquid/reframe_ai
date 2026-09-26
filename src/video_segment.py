"""Video segmentation: find the most engaging 30-second segment.

Analyzes the full video using four signals computed at 1-second
intervals:

  1. **Motion density** — frame-to-frame pixel difference (RMS), computed
     via ffmpeg at reduced resolution. High motion = action/dialogue.
  2. **Face activity** — number of face detections per second from the
     perception stage (already computed at 4 fps sampling).
  3. **Audio energy** — RMS energy per second from the speech activity
     detector (already computed).
  4. **Shot boundaries** — frame indices where visual content changes
     abruptly (already computed by detection_shot_boundaries).

A sliding 30-second window is scored by a weighted sum of these
signals. The window with the highest score is returned as the
"best segment."

If the video is shorter than 30 seconds, the entire video is returned.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    OUTPUT_DIR,
    VIDEO_SAMPLE_FPS,
    VIDEO_SEGMENT_DURATION_SEC,
    VIDEO_SEGMENT_MOTION_WEIGHT,
    VIDEO_SEGMENT_FACE_WEIGHT,
    VIDEO_SEGMENT_AUDIO_WEIGHT,
    VIDEO_SEGMENT_SHOT_WEIGHT,
)
from .video_ingestion import VideoMetadata
from .video_perception import (
    VideoPerceiver,
    extract_frames_by_interval,
    raw_to_numpy,
    get_scaled_dims,
    detect_shot_boundaries,
    TrackedPerson,
)
from .video_audio import detect_speech_activity
from .video_speaker import ActiveSpeakerSegment


@dataclass
class SegmentScore:
    """Score for a candidate segment."""

    start: float
    end: float
    motion_score: float
    face_score: float
    audio_score: float
    shot_score: float
    total_score: float


@dataclass
class BestSegment:
    """The best segment found in the video."""

    start: float
    end: float
    score: float
    scores: list[SegmentScore] = field(default_factory=list)


def _compute_motion_density(
    video_path: str | Path,
    duration: float,
    fps: float,
    width: int,
    height: int,
) -> list[float]:
    """Compute per-second motion density using frame-to-frame pixel differences.

    Uses ffmpeg to extract frames at 1 fps at reduced resolution, then
    computes the RMS pixel difference between consecutive frames.
    """
    sample_w = min(320, width)
    sample_h = int(round(sample_w * height / width))
    num_samples = int(min(duration, 60)) + 1  # cap at 60 samples
    cmd = [
        "ffmpeg", "-y", "-v", "quiet", "-i", str(video_path),
        "-vf", f"fps=1,scale={sample_w}:{sample_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0 or len(result.stdout) == 0:
        return [0.0] * num_samples

    frame_size = sample_w * sample_h * 3
    num_frames = len(result.stdout) // frame_size
    if num_frames < 2:
        return [0.0] * num_samples

    motion_scores: list[float] = []
    prev_frame: np.ndarray | None = None
    for i in range(num_frames):
        offset = i * frame_size
        raw = result.stdout[offset:offset + frame_size]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(sample_h, sample_w, 3)
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32) / 255.0
        if prev_frame is not None:
            diff = np.abs(gray - prev_frame)
            motion_scores.append(float(np.mean(diff)))
        prev_frame = gray

    # Pad to num_samples
    while len(motion_scores) < num_samples:
        motion_scores.append(0.0)
    return motion_scores[:num_samples]


def _compute_face_density(
    video_path: str | Path,
    metadata: VideoMetadata,
    target_fps: float = 4.0,
) -> list[float]:
    """Compute per-second face detection count using MediaPipe.

    Returns a list of face counts, one per second.
    """
    frames = extract_frames_by_interval(
        video_path, metadata, target_fps=target_fps, max_dim=640,
    )
    scaled_w, scaled_h = get_scaled_dims(metadata, 640)
    perceiever = VideoPerceiver(min_face_confidence=0.3, num_faces=10)

    face_counts: list[float] = []
    for frame_idx, timestamp, raw in frames:
        arr = raw_to_numpy(raw, scaled_w, scaled_h)
        faces = perceiever.detect_faces(arr)
        sec = int(timestamp)
        while len(face_counts) <= sec:
            face_counts.append(0.0)
        face_counts[sec] = max(face_counts[sec], len(faces))

    perceiever.close()

    # Pad to duration
    num_secs = int(metadata.duration_sec) + 1
    while len(face_counts) < num_secs:
        face_counts.append(0.0)
    return face_counts[:num_secs]


def _compute_shot_density(
    shot_boundaries: list[int],
    duration: float,
    fps: float,
) -> list[float]:
    """Compute per-second shot change count."""
    num_secs = int(duration) + 1
    shot_per_sec = [0.0] * num_secs
    for frame_idx in shot_boundaries:
        sec = int(frame_idx / fps)
        if 0 <= sec < num_secs:
            shot_per_sec[sec] += 1.0
    return shot_per_sec


def _normalize(values: list[float]) -> list[float]:
    """Normalize a list of values to 0–1 range."""
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    max_val = arr.max()
    if max_val < 1e-9:
        return [0.0] * len(values)
    return (arr / max_val).tolist()


def find_best_segment(
    video_path: str | Path,
    metadata: VideoMetadata,
    segment_duration: float = VIDEO_SEGMENT_DURATION_SEC,
    motion_weight: float = VIDEO_SEGMENT_MOTION_WEIGHT,
    face_weight: float = VIDEO_SEGMENT_FACE_WEIGHT,
    audio_weight: float = VIDEO_SEGMENT_AUDIO_WEIGHT,
    shot_weight: float = VIDEO_SEGMENT_SHOT_WEIGHT,
) -> BestSegment:
    """Find the best ``segment_duration``-second window in the video.

    Scores every possible window position using a weighted combination of
    motion density, face activity, audio energy, and shot changes.

    If the video is shorter than ``segment_duration``, the entire video
    is returned.
    """
    duration = metadata.duration_sec
    fps = metadata.fps
    width, height = metadata.width, metadata.height

    if duration <= segment_duration:
        return BestSegment(
            start=0.0, end=duration, score=1.0,
            scores=[SegmentScore(
                start=0.0, end=duration,
                motion_score=1.0, face_score=1.0,
                audio_score=1.0, shot_score=1.0,
                total_score=1.0,
            )],
        )

    # Compute per-second signals
    motion_per_sec = _compute_motion_density(video_path, duration, fps, width, height)
    face_per_sec = _compute_face_density(video_path, metadata)

    # Audio from existing SAD
    speech_activity = detect_speech_activity(video_path, metadata)
    audio_per_sec: list[float] = []
    window_sec = speech_activity.window_ms / 1000.0
    num_windows = len(speech_activity.energy_curve)
    num_secs = int(duration) + 1

    # Aggregate audio energy per second
    audio_raw = np.array(speech_activity.energy_curve, dtype=np.float64)
    for s in range(num_secs):
        start_w = int(s / window_sec)
        end_w = int((s + 1) / window_sec)
        end_w = min(end_w, num_windows)
        if start_w < num_windows:
            audio_per_sec.append(float(np.max(audio_raw[start_w:end_w])) if end_w > start_w else 0.0)
        else:
            audio_per_sec.append(0.0)

    # Shot boundaries
    shot_per_sec = _compute_shot_density(
        detect_shot_boundaries(
            extract_frames_by_interval(video_path, metadata, target_fps=2.0, max_dim=640),
            *get_scaled_dims(metadata, 640),
        ),
        duration, fps,
    )

    # Normalize each signal to 0–1
    motion_norm = _normalize(motion_per_sec)
    face_norm = _normalize(face_per_sec)
    audio_norm = _normalize(audio_per_sec)
    shot_norm = _normalize(shot_per_sec)

    num_secs = min(len(motion_norm), len(face_norm), len(audio_norm), len(shot_norm))

    # Slide a window across the video
    window_size = int(segment_duration)
    best_score = -1.0
    best_start = 0.0
    all_scores: list[SegmentScore] = []

    for start_sec in range(0, num_secs - window_size + 1):
        end_sec = start_sec + window_size
        seg_motion = float(np.mean(motion_norm[start_sec:end_sec])) if num_secs > 0 else 0.0
        seg_face = float(np.mean(face_norm[start_sec:end_sec]))
        seg_audio = float(np.mean(audio_norm[start_sec:end_sec]))
        seg_shot = float(np.mean(shot_norm[start_sec:end_sec]))

        total = (
            motion_weight * seg_motion
            + face_weight * seg_face
            + audio_weight * seg_audio
            + shot_weight * seg_shot
        )

        score = SegmentScore(
            start=float(start_sec), end=float(end_sec),
            motion_score=seg_motion, face_score=seg_face,
            audio_score=seg_audio, shot_score=seg_shot,
            total_score=total,
        )
        all_scores.append(score)

        if total > best_score:
            best_score = total
            best_start = float(start_sec)

    best_end = best_start + segment_duration

    # Build a few candidate windows at different offsets for finer granularity
    for offset in [0.25, 0.5, 0.75]:
        start_t = best_start + offset
        end_t = start_t + segment_duration
        if end_t > duration:
            continue
        start_sec = int(start_t)
        end_sec = int(end_t)
        if start_sec >= num_secs or end_sec > num_secs:
            continue
        seg_motion = float(np.mean(motion_norm[start_sec:end_sec]))
        seg_face = float(np.mean(face_norm[start_sec:end_sec]))
        seg_audio = float(np.mean(audio_norm[start_sec:end_sec]))
        seg_shot = float(np.mean(shot_norm[start_sec:end_sec]))
        total = (
            motion_weight * seg_motion
            + face_weight * seg_face
            + audio_weight * seg_audio
            + shot_weight * seg_shot
        )
        if total > best_score:
            best_score = total
            best_start = float(start_t)
            best_end = float(end_t)

    return BestSegment(
        start=best_start,
        end=best_end,
        score=best_score,
        scores=all_scores,
    )
