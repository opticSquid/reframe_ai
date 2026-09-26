"""Active speaker inference for the video pipeline.

Combines audio speech activity detection with visual mouth-open (MAR)
analysis to produce a timeline of who is speaking when.

Algorithm:
  1. Audio analysis: RMS energy per 50ms window → speech activity mask.
  2. Visual analysis: MAR (Mouth Aspect Ratio) per tracked face at sampled frames.
  3. Fusion: When audio is active, the tracked face with the highest MAR
     (at the nearest visual sample) is the active speaker.
  4. Temporal smoothing: hold speaker for 0.3s minimum, interpolate MAR.

Fallback: if face landmarks are unavailable, use audio activity + face
horizontal position (left/right clustering) to assign speaker identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .video_audio import SpeechActivity
from .video_perception import TrackedPerson, DetectedFace


@dataclass
class ActiveSpeakerSegment:
    """A time interval during which a specific speaker is active."""

    start: float
    end: float
    speaker_id: int
    confidence: float
    mar: float | None = None
    audio_energy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
            "mar": self.mar,
            "audio_energy": self.audio_energy,
        }


@dataclass
class CropTrajectoryPoint:
    """A single point on the crop trajectory."""

    time: float
    center_x: float      # normalized (0–1) relative to source width
    center_y: float      # normalized (0–1) relative to source height
    crop_x: int          # absolute crop top-left
    crop_y: int
    crop_w: int
    crop_h: int
    speaker_id: int
    source_width: int
    source_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "center": {"x": self.center_x, "y": self.center_y},
            "crop": {"x": self.crop_x, "y": self.crop_y, "w": self.crop_w, "h": self.crop_h},
            "speaker_id": self.speaker_id,
            "source_dimensions": {"width": self.source_width, "height": self.source_height},
        }


@dataclass
class SpeakerTimelineResult:
    """Full result of active-speaker inference."""

    segments: list[ActiveSpeakerSegment]
    trajectory: list[CropTrajectoryPoint]
    speakers: list[int]

    def to_speaker_dict(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.segments]

    def to_trajectory_dict(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.trajectory]


def infer_active_speaker_timeline(
    tracks: list[TrackedPerson],
    speech_activity: SpeechActivity,
    fps: float,
    width: int,
    height: int,
    track_mars: dict[int, list[tuple[float, float | None]]],
    min_speaker_dur: float = 0.5,
    hold_speaker_dur: float = 0.3,
) -> list[ActiveSpeakerSegment]:
    """Infer the active speaker timeline from tracks + audio + MAR.

    Parameters
    ----------
    tracks
        TrackedPerson objects from the perception stage.
    speech_activity
        SpeechActivity with audio energy curve and speech mask.
    fps
        Video frame rate.
    width, height
        Source video dimensions (used for position-based fallback).
    track_mars
        Dict mapping track_id → list of (timestamp, mar) pairs from sampled frames.
    min_speaker_dur
        Minimum duration (seconds) for a speaker segment to be kept.
    hold_speaker_dur
        Hold current speaker for this long during speech gaps (ms).
    """
    if not tracks or len(speech_activity.segments) == 0:
        # Fallback: if no speech, assign to leftmost track throughout
        if tracks:
            seg = ActiveSpeakerSegment(
                start=0.0, end=speech_activity.duration_sec,
                speaker_id=tracks[0].id, confidence=0.3,
            )
            return [seg]
        return []

    # Build per-track MAR interpolation functions
    track_interp: dict[int, Any] = {}
    for track_id, mar_samples in track_mars.items():
        if len(mar_samples) >= 2:
            times = np.array([s[0] for s in mar_samples])
            mars = np.array([s[1] if s[1] is not None else 0.0 for s in mar_samples])
            # Use np.interp for simple linear interpolation
            track_interp[track_id] = (times, mars)
        elif len(mar_samples) == 1:
            val = mar_samples[0][1] if mar_samples[0][1] is not None else 0.5
            track_interp[track_id] = (None, np.array([val]))

    def get_mar(track_id: int, t: float) -> float:
        """Interpolate MAR for a track at time t."""
        if track_id not in track_interp:
            return 0.5  # neutral default
        times, mars = track_interp[track_id]
        if times is None:
            return float(mars[0])
        # Clamp to known range
        t_clamped = np.clip(t, times[0], times[-1]) if len(times) > 0 else t
        return float(np.interp(t_clamped, times, mars))

    # Build active speaker timeline
    # Iterate in small time steps within speech segments
    window_ms = speech_activity.window_ms
    window_sec = window_ms / 1000.0
    num_windows = len(speech_activity.speech_mask)
    speech_active = np.array(speech_activity.speech_mask)
    energy_curve = np.array(speech_activity.energy_curve)

    # Determine speaker for each time window
    speaker_per_window: list[int | None] = [None] * num_windows
    for wi in range(num_windows):
        t = wi * window_sec
        if not speech_active[wi]:
            continue

        # Find which track has the highest MAR at this time
        best_track = -1
        best_mar = -1.0
        for track in tracks:
            mar = get_mar(track.id, t)
            if mar > best_mar:
                best_mar = mar
                best_track = track.id

        if best_track >= 0:
            speaker_per_window[wi] = best_track

    # Fallback: if no MAR data, use face position (left/right)
    if all(s is None for s in speaker_per_window):
        for wi in range(num_windows):
            if not speech_active[wi]:
                continue
            t = wi * window_sec
            # Assign to leftmost or rightmost face at nearest sample
            best_track = -1
            best_score = -1.0
            for track in tracks:
                # Get nearest bbox to this time
                nearest_bbox = _nearest_bbox_at_time(track, t)
                if nearest_bbox is not None:
                    # Use horizontal position as a discriminator
                    cx = nearest_bbox.x + nearest_bbox.width / 2
                    # Normalize: left = 0, right = 1
                    norm_x = cx / width
                    # Prefer the track that is more centered (closer to 0.5)
                    score = 1.0 - abs(norm_x - 0.5) * 2
                    if score > best_score:
                        best_score = score
                        best_track = track.id
            if best_track >= 0:
                speaker_per_window[wi] = best_track

    # Smooth: hold speaker during short speech gaps
    speaker_with_hold: list[int | None] = list(speaker_per_window)
    last_speaker: int | None = None
    hold_windows = max(1, int(hold_speaker_dur / window_sec))
    for wi in range(num_windows):
        if speaker_with_hold[wi] is not None:
            last_speaker = speaker_with_hold[wi]
        elif last_speaker is not None:
            # Fill gap with last speaker (but don't extend beyond hold limit)
            gap_start = wi - hold_windows
            if gap_start >= 0 and speaker_per_window[gap_start] == last_speaker:
                speaker_with_hold[wi] = last_speaker
            last_speaker = None  # reset after hold window

    # Merge into segments
    segments: list[ActiveSpeakerSegment] = []
    i = 0
    while i < num_windows:
        if speaker_with_hold[i] is not None:
            start_w = i
            speaker_id = speaker_with_hold[i]
            while i < num_windows and speaker_with_hold[i] == speaker_id:
                i += 1
            end_w = i - 1

            start_time = start_w * window_sec
            end_time = (end_w + 1) * window_sec
            if end_time - start_time < min_speaker_dur:
                start_time = max(0, start_time - min_speaker_dur / 2)

            # Compute confidence from MAR values and audio energy
            mar_vals = []
            energy_vals = []
            for w in range(start_w, end_w + 1):
                t = w * window_sec
                for track in tracks:
                    if track.id == speaker_id:
                        mar = get_mar(track.id, t)
                        mar_vals.append(mar)
                energy_vals.append(float(energy_curve[w]))

            avg_mar = float(np.mean(mar_vals)) if mar_vals else None
            avg_energy = float(np.mean(energy_vals)) if energy_vals else None
            confidence = 0.5
            if avg_energy is not None and avg_energy > 0:
                confidence = min(0.5 + avg_energy * 0.5, 0.9)
            if avg_mar is not None:
                # MAR above 0.1 suggests open mouth
                if avg_mar > 0.1:
                    confidence = min(confidence + 0.2, 0.95)
                elif avg_mar > 0.05:
                    confidence = min(confidence + 0.1, 0.9)

            speaker_id_val = int(speaker_id)
            segments.append(ActiveSpeakerSegment(
                start=start_time,
                end=end_time,
                speaker_id=speaker_id_val,
                confidence=confidence,
                mar=avg_mar,
                audio_energy=avg_energy,
            ))
        else:
            i += 1

    # Merge very short adjacent segments of the same speaker
    merged: list[ActiveSpeakerSegment] = []
    for seg in segments:
        if (merged and merged[-1].speaker_id == seg.speaker_id
                and abs(seg.start - merged[-1].end) < 2.0):
            merged[-1].end = seg.end
            if seg.confidence > merged[-1].confidence:
                merged[-1].confidence = seg.confidence
                merged[-1].mar = seg.mar
        else:
            merged.append(seg)

    return merged


def _nearest_bbox_at_time(
    track: TrackedPerson,
    t: float,
) -> Any | None:
    """Find the bbox nearest to time t for a track."""
    best_bbox = None
    best_diff = float("inf")
    for f in track.face_bboxes:
        diff = abs(f.timestamp - t)
        if diff < best_diff:
            best_diff = diff
            best_bbox = f
    return best_bbox


def build_crop_trajectory(
    tracks: list[TrackedPerson],
    segments: list[ActiveSpeakerSegment],
    total_duration: float,
    total_frames: int,
    fps: float,
    source_width: int,
    source_height: int,
    target_ratio: float = 9 / 16,
    target_coverage: float = 0.4,
    smoothing_alpha: float = 0.2,
    initial_center: tuple[float, float] | None = None,
) -> list[CropTrajectoryPoint]:
    """Build a per-frame crop trajectory driven by the active speaker timeline.

    Uses the existing ``compute_crop`` from ``src.cropper`` to compute the
    crop rectangle for the active speaker on each frame.  Smooths the
    crop center with an EMA to avoid jitter.

    Parameters
    ----------
    initial_center
        Optional (normalized_x, normalized_y) crop center to initialize the
        EMA smoothing. When provided (e.g. from Gemini crop planning), the
        trajectory starts from this center instead of the first face bbox.
    """
    from .cropper import Subject, compute_crop

    if not tracks:
        # Center crop fallback for all frames
        from .cropper import _center_crop
        cr = _center_crop(source_width, source_height, target_ratio)
        points: list[CropTrajectoryPoint] = []
        for fi in range(total_frames):
            t = fi / fps
            points.append(CropTrajectoryPoint(
                time=t,
                center_x=0.5, center_y=0.5,
                crop_x=cr.x, crop_y=cr.y, crop_w=cr.width, crop_h=cr.height,
                speaker_id=-1,
                source_width=source_width, source_height=source_height,
            ))
        return points

    # Build a time-indexed map of speaker → track bboxes
    # For each frame, determine the active speaker and their predicted bbox
    trajectory: list[CropTrajectoryPoint] = []

    # Interpolate track bboxes over time
    track_data: dict[int, list[tuple[float, DetectedFace]]] = {}
    for track in tracks:
        for bbox in track.face_bboxes:
            track_data.setdefault(track.id, []).append((bbox.timestamp, bbox))

    def predict_bbox(track_id: int, t: float) -> tuple[int, int, int, int] | None:
        """Predict face bbox for a track at time t via linear interpolation."""
        samples = track_data.get(track_id, [])
        if not samples:
            return None
        if len(samples) == 1:
            b = samples[0][1]
            return (b.x, b.y, b.width, b.height)
        times = [s[0] for s in samples]
        # Find bracketing samples
        if t <= times[0]:
            b = samples[0][1]
        elif t >= times[-1]:
            b = samples[-1][1]
        else:
            for i in range(len(times) - 1):
                if times[i] <= t <= times[i + 1]:
                    ratio = (t - times[i]) / (times[i + 1] - times[i]) if times[i + 1] > times[i] else 0
                    b1 = samples[i][1]
                    b2 = samples[i + 1][1]
                    ix = int(round(b1.x + (b2.x - b1.x) * ratio))
                    iy = int(round(b1.y + (b2.y - b1.y) * ratio))
                    iw = int(round(b1.width + (b2.width - b1.width) * ratio))
                    ih = int(round(b1.height + (b2.height - b1.height) * ratio))
                    return (ix, iy, iw, ih)
            b = samples[-1][1]
        return (b.x, b.y, b.width, b.height)

    def get_active_speaker(t: float) -> int | None:
        """Find the active speaker ID at time t."""
        for seg in segments:
            if seg.start <= t <= seg.end:
                return seg.speaker_id
        return None

    # Smoothed crop center
    smooth_cx: float | None = None
    smooth_cy: float | None = None

    # Initialize from Gemini-recommended center if provided
    if initial_center is not None:
        icx, icy = initial_center
        smooth_cx = float(icx) * source_width
        smooth_cy = float(icy) * source_height

    # Fallback: if no tracks, use initial_center or center
    if not tracks:
        from .cropper import _center_crop
        cr = _center_crop(source_width, source_height, target_ratio)
        points: list[CropTrajectoryPoint] = []
        for fi in range(total_frames):
            t = fi / fps
            cx = cr.x + cr.width / 2
            cy = cr.y + cr.height / 2
            if initial_center is not None:
                icx, icy = initial_center
                cx = float(icx) * source_width
                cy = float(icy) * source_height
                cr.x = max(0, min(int(round(cx - cr.width / 2)), source_width - cr.width))
                cr.y = max(0, min(int(round(cy - cr.height / 2)), source_height - cr.height))
            points.append(CropTrajectoryPoint(
                time=t,
                center_x=cx / source_width, center_y=cy / source_height,
                crop_x=cr.x, crop_y=cr.y, crop_w=cr.width, crop_h=cr.height,
                speaker_id=-1,
                source_width=source_width, source_height=source_height,
            ))
        return points

    for fi in range(total_frames):
        t = fi / fps
        speaker_id = get_active_speaker(t)

        if speaker_id is not None:
            track = next((tr for tr in tracks if tr.id == speaker_id), None)
            if track is not None:
                bbox = predict_bbox(track.id, t)
                if bbox is not None:
                    fx, fy, fw, fh = bbox
                    # Asymmetric padding to include shoulders and headroom.
                    # MediaPipe BlazeFace bboxes are face-only (forehead to chin).
                    pad_x = 0.5       # 25% each side for shoulders
                    pad_y_top = 0.4   # 40% above face for hair/headroom
                    pad_y_bottom = 0.6  # 60% below face for chin/shoulders
                    scaled_w = int(round(fw * (1 + pad_x)))
                    scaled_h = int(round(fh * (1 + pad_y_top + pad_y_bottom)))
                    scaled_x = max(0, fx - int(pad_x * fw / 2))
                    scaled_y = max(0, fy - int(pad_y_top * fh))

                    subject = Subject(
                        bbox=(float(scaled_x), float(scaled_y), float(scaled_w), float(scaled_h)),
                        importance=1.0,
                        id=f"person_{speaker_id}",
                        category="person",
                    )
                    cr = compute_crop(
                        source_w=source_width, source_h=source_height,
                        subjects=[subject],
                        target_ratio=target_ratio,
                        target_coverage=target_coverage,
                    )
                    # Enforce minimum crop size to prevent excessive zoom-in
                    # when the face is far from the camera. For a 9:16 portrait
                    # crop from a 16:9 source, enforce at least 35% of source
                    # height (≈2.8x max zoom from the full-height crop).
                    if target_ratio < 1.0 and cr.height < int(source_height * 0.35):
                        cr.height = int(source_height * 0.35)
                        cr.width = int(round(cr.height * target_ratio))
                        cx = cr.x + cr.width / 2
                        cy = cr.y + cr.height / 2
                        cr.x = max(0, min(int(round(cx - cr.width / 2)), source_width - cr.width))
                        cr.y = max(0, min(int(round(cy - cr.height / 2)), source_height - cr.height))
                        cx = cr.x + cr.width / 2
                        cy = cr.y + cr.height / 2
                    elif target_ratio >= 1.0 and cr.width < int(source_width * 0.35):
                        cr.width = int(source_width * 0.35)
                        cr.height = int(round(cr.width / target_ratio))
                        cx = cr.x + cr.width / 2
                        cy = cr.y + cr.height / 2
                        cr.x = max(0, min(int(round(cx - cr.width / 2)), source_width - cr.width))
                        cr.y = max(0, min(int(round(cy - cr.height / 2)), source_height - cr.height))
                        cx = cr.x + cr.width / 2
                        cy = cr.y + cr.height / 2
                    cx = cr.x + cr.width / 2
                    cy = cr.y + cr.height / 2
                    if smooth_cx is None:
                        smooth_cx = float(cx)
                        smooth_cy = float(cy)
                    else:
                        assert smooth_cx is not None and smooth_cy is not None
                        smooth_cx = (1 - smoothing_alpha) * smooth_cx + smoothing_alpha * float(cx)
                        smooth_cy = (1 - smoothing_alpha) * smooth_cy + smoothing_alpha * float(cy)

                    # Recompute crop from smoothed center
                    cr.x = int(round(smooth_cx - cr.width / 2))
                    cr.y = int(round(smooth_cy - cr.height / 2))
                    cr.x = max(0, min(cr.x, source_width - cr.width))
                    cr.y = max(0, min(cr.y, source_height - cr.height))

                    trajectory.append(CropTrajectoryPoint(
                        time=t,
                        center_x=smooth_cx / source_width,
                        center_y=smooth_cy / source_height,
                        crop_x=cr.x, crop_y=cr.y,
                        crop_w=cr.width, crop_h=cr.height,
                        speaker_id=speaker_id,
                        source_width=source_width, source_height=source_height,
                    ))
                    continue

        # Fallback: center crop or hold last position
        if smooth_cx is not None and trajectory:
            # Hold last crop position
            last = trajectory[-1]
            trajectory.append(CropTrajectoryPoint(
                time=t,
                center_x=last.center_x, center_y=last.center_y,
                crop_x=last.crop_x, crop_y=last.crop_y,
                crop_w=last.crop_w, crop_h=last.crop_h,
                speaker_id=speaker_id if speaker_id is not None else -1,
                source_width=source_width, source_height=source_height,
            ))
        else:
            # No speaker data: use smoothed center (from initial_center) or center crop
            from .cropper import _center_crop
            cr = _center_crop(source_width, source_height, target_ratio)
            if smooth_cx is not None and smooth_cy is not None:
                # Reposition crop to the smoothed center (Gemini-recommended or held)
                cx, cy = smooth_cx, smooth_cy
            else:
                cx = cr.x + cr.width / 2
                cy = cr.y + cr.height / 2
                smooth_cx = float(cx)
                smooth_cy = float(cy)
            cr.x = max(0, min(int(round(cx - cr.width / 2)), source_width - cr.width))
            cr.y = max(0, min(int(round(cy - cr.height / 2)), source_height - cr.height))
            trajectory.append(CropTrajectoryPoint(
                time=t,
                center_x=smooth_cx / source_width,
                center_y=smooth_cy / source_height,
                crop_x=cr.x, crop_y=cr.y,
                crop_w=cr.width, crop_h=cr.height,
                speaker_id=-1,
                source_width=source_width, source_height=source_height,
            ))

    return trajectory
