"""Video pipeline orchestrator for ReframeAI.

Top-level entry point that ties together:
  - Video ingestion (metadata extraction)
  - Perception (face detection, tracking, face landmarks/MAR)
  - Audio analysis (speech activity detection)
  - Active speaker inference (audio + visual fusion)
  - Crop trajectory planning (reuses src.cropper.compute_crop)
  - Rendering (dynamic crop video, still, debug overlay)
  - Validation (reuses src.validator.validate_asset)

Mirrors the structure of ``src/pipeline.py`` (the image pipeline orchestrator).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import (
    OUTPUT_DIR,
    VIDEO_SAMPLE_FPS,
    VIDEO_LANDMARK_FPS,
    REEL_ASPECT_RATIO,
)
from .video_ingestion import VideoMetadata, extract_video_metadata
from .video_perception import (
    VideoPerceiver, SimpleTracker,
    extract_frames_by_interval, raw_to_numpy, get_scaled_dims,
    detect_shot_boundaries,
    TrackedPerson, FrameSample, PerceptionResult,
)
from .video_audio import detect_speech_activity, SpeechActivity
from .video_speaker import (
    ActiveSpeakerSegment, CropTrajectoryPoint,
    infer_active_speaker_timeline, build_crop_trajectory,
)
from .video_rendering import render_vertical_video
from .video_review import gemini_review_video, VideoReviewResult
from .validator import validate_asset


@dataclass
class VideoPipelineResult:
    """Full result of the video pipeline."""

    video_path: str
    metadata: VideoMetadata
    tracks: list[TrackedPerson]
    speech_activity: SpeechActivity
    speaker_segments: list[ActiveSpeakerSegment]
    trajectory: list[CropTrajectoryPoint]
    perception_time_sec: float
    audio_time_sec: float
    speaker_time_sec: float
    render_time_sec: float
    review_time_sec: float
    total_time_sec: float
    reel_path: str
    still_path: str
    debug_path: str | None
    validation: Any  # ValidationResult
    manifest: dict[str, Any] = field(default_factory=dict)
    ai_review: VideoReviewResult | None = None
    still_paths: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "metadata": self.metadata.to_dict(),
            "num_tracks": len(self.tracks),
            "num_speaker_segments": len(self.speaker_segments),
            "num_trajectory_points": len(self.trajectory),
            "speakers": [s.speaker_id for s in self.speaker_segments],
            "timing": {
                "perception": round(self.perception_time_sec, 2),
                "audio": round(self.audio_time_sec, 2),
                "speaker": round(self.speaker_time_sec, 2),
                "render": round(self.render_time_sec, 2),
                "review": round(self.review_time_sec, 2),
                "total": round(self.total_time_sec, 2),
            },
            "outputs": {
                "reel": self.reel_path,
                "still": self.still_path,
                "debug": self.debug_path,
                "stills": self.still_paths,
            },
            "validation_passed": self.validation.passed if self.validation else False,
            "validation_errors": self.validation.errors if self.validation else [],
            "validation_warnings": self.validation.warnings if self.validation else [],
            "ai_review": (
                {
                    "passed": self.ai_review.passed,
                    "score": self.ai_review.score,
                    "frames_reviewed": self.ai_review.frames_reviewed,
                    "errors": self.ai_review.errors,
                    "warnings": self.ai_review.warnings,
                    "suggestions": self.ai_review.suggestions,
                }
                if self.ai_review else None
            ),
        }


def process_video_to_reel(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    target_fps: float = VIDEO_SAMPLE_FPS,
    landmark_fps: float = VIDEO_LANDMARK_FPS,
    generate_debug: bool = True,
    progress_callback: Callable[[str, int], None] | None = None,
) -> VideoPipelineResult:
    """Process a master video into a vertical 9:16 reel with dynamic cropping.

    Full pipeline:
      1. Ingest: extract video metadata via ffprobe
      2. Perception: sample frames → face detection → tracking → face landmarks (MAR)
      3. Audio: extract audio → speech activity detection
      4. Speaker: fuse audio + visual MAR → active speaker timeline
      5. Crop: build per-frame crop trajectory using compute_crop
      6. Render: FFmpeg pipe-based encode with dynamic crop + audio mux
      7. Validate: run against spec/platform_spec.yaml
    """
    t_start = time.perf_counter()
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    video_reels_dir = output_dir / "video_reels"
    stills_dir = output_dir / "stills"
    manifests_dir = output_dir / "manifests"
    for d in (video_reels_dir, stills_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Ingest: extracting video metadata", 10)
    metadata = extract_video_metadata(video_path)
    # Compute output dimensions from target ratio
    if REEL_ASPECT_RATIO < 1.0:
        out_w = int(round(metadata.height * REEL_ASPECT_RATIO))
        out_h = metadata.height
    else:
        out_w = metadata.width
        out_h = int(round(metadata.width / REEL_ASPECT_RATIO))

    # ------------------------------------------------------------------
    # Stage 2: Perception (frame sampling + face detection + tracking)
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Perception: detecting faces and tracking speakers", 20)
    t0 = time.perf_counter()

    # Extract frames at target_fps, scaled to max_dim=640
    frames = extract_frames_by_interval(
        video_path, metadata, target_fps=target_fps, max_dim=640,
    )
    scaled_w, scaled_h = get_scaled_dims(metadata, 640)

    # Shot boundary detection
    shot_bounds = detect_shot_boundaries(frames, scaled_w, scaled_h, threshold=0.25)

    # Face detection + tracking
    # Note: faces are detected at scaled resolution (640x360) for speed.
    # We scale bboxes back to source resolution for trajectory computation.
    scale_x = metadata.width / scaled_w
    scale_y = metadata.height / scaled_h
    perceiever = VideoPerceiver(min_face_confidence=0.3, num_faces=10)
    tracker = SimpleTracker(max_lost=15, iou_threshold=0.15, dist_threshold=0.5)

    sample_results: list[FrameSample] = []
    for frame_idx, timestamp, raw in frames:
        arr = raw_to_numpy(raw, scaled_w, scaled_h)
        faces = perceiever.detect_faces(arr)
        for f in faces:
            f.frame_idx = frame_idx
            f.timestamp = timestamp
            # Scale bbox from detection resolution to source resolution
            f.x = int(round(f.x * scale_x))
            f.y = int(round(f.y * scale_y))
            f.width = int(round(f.width * scale_x))
            f.height = int(round(f.height * scale_y))
        tracker.update(faces, metadata.width, metadata.height)
        sample_results.append(FrameSample(
            frame_idx=frame_idx, timestamp=timestamp, faces=faces,
        ))

    perceiever.close()
    tracks = tracker.tracks

    # Filter out tracks with very few detections (likely noise)
    tracks = [t for t in tracks if len(t.face_bboxes) >= 2]

    # ------------------------------------------------------------------
    # Stage 3: Audio analysis
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Audio: detecting speech activity", 40)
    t_audio_0 = time.perf_counter()
    speech_activity = detect_speech_activity(video_path, metadata)
    t_audio_1 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 4: Face landmarks (MAR) on sampled frames during speech
    # Only compute MAR on a subset of frames to save time
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Speaker: inferring active speaker timeline", 55)
    t_speaker_0 = time.perf_counter()

    # Re-run perception with landmarks at lower fps
    landmark_frames = extract_frames_by_interval(
        video_path, metadata, target_fps=landmark_fps, max_dim=640,
    )

    perceiver_lm = VideoPerceiver(min_face_confidence=0.3, num_faces=10)
    track_mars: dict[int, list[tuple[float, float | None]]] = {t.id: [] for t in tracks}

    for frame_idx, timestamp, raw in landmark_frames:
        arr = raw_to_numpy(raw, scaled_w, scaled_h)
        faces = perceiver_lm.detect_faces(arr)
        mars = perceiver_lm.compute_mars(arr, faces)

        # Associate with tracks
        for face, mar in zip(faces, mars):
            # Find which track this face belongs to (nearest by bbox)
            best_track = -1
            best_dist = float("inf")
            for track in tracks:
                if not track.face_bboxes:
                    continue
                last = track.face_bboxes[-1]
                dist = abs(last.timestamp - timestamp)
                if dist < best_dist:
                    best_dist = dist
                    best_track = track.id
            if best_track >= 0:
                track_mars.setdefault(best_track, []).append((timestamp, mar))

    perceiver_lm.close()

    # ------------------------------------------------------------------
    # Stage 5: Active speaker timeline
    # ------------------------------------------------------------------
    segments = infer_active_speaker_timeline(
        tracks=tracks,
        speech_activity=speech_activity,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        track_mars=track_mars,
    )

    # ------------------------------------------------------------------
    # Stage 6: Crop trajectory
    # ------------------------------------------------------------------
    trajectory = build_crop_trajectory(
        tracks=tracks,
        segments=segments,
        total_duration=metadata.duration_sec,
        total_frames=metadata.total_frames,
        fps=metadata.fps,
        source_width=metadata.width,
        source_height=metadata.height,
        target_ratio=REEL_ASPECT_RATIO,
        target_coverage=0.65,
        smoothing_alpha=0.3,
    )
    t_speaker_1 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 7: Render
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Render: generating vertical reel and stills", 70)
    t_render_0 = time.perf_counter()

    reel_path = str(video_reels_dir / f"{stem}_9_16.mp4")
    debug_path = str(video_reels_dir / f"{stem}_debug.mp4") if generate_debug else None

    render_result = render_vertical_video(
        video_path=video_path,
        trajectory=trajectory,
        tracks=tracks,
        output_path=reel_path,
        target_ratio=REEL_ASPECT_RATIO,
        source_width=metadata.width,
        source_height=metadata.height,
        fps=metadata.fps,
        total_frames=metadata.total_frames,
        generate_debug=generate_debug,
        debug_path=debug_path,
        stills_dir=stills_dir,
    )

    t_render_1 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 8: AI Visual Review (post-rendering Gemini critique)
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Review: Gemini assessing framing quality", 85)
    t_review_0 = time.perf_counter()
    ai_review = gemini_review_video(
        original_video_path=video_path,
        reel_video_path=render_result.reel_path,
        trajectory=trajectory,
        segments=segments,
        shot_boundaries=shot_bounds,
        metadata=metadata,
        max_review_frames=12,
    )
    t_review_1 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 9: Validation
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("Validation: checking against platform spec", 95)
    manifest = _build_video_manifest(
        video_path=video_path,
        stem=stem,
        metadata=metadata,
        tracks=tracks,
        segments=segments,
        trajectory=trajectory,
        reel_path=render_result.reel_path,
        still_path=render_result.still_path,
    )

    validation = validate_asset(manifest, verify_files=True)

    # Save manifest
    manifest_path = manifests_dir / f"{stem}_9_16.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Save all debug artifacts
    _save_artifacts(stem, output_dir, metadata, tracks, shot_bounds,
                    speech_activity, segments, trajectory, render_result, validation,
                    ai_review)

    t_end = time.perf_counter()

    return VideoPipelineResult(
        video_path=str(video_path),
        metadata=metadata,
        tracks=tracks,
        speech_activity=speech_activity,
        speaker_segments=segments,
        trajectory=trajectory,
        perception_time_sec=t_audio_0 - t_start,
        audio_time_sec=t_audio_1 - t_audio_0,
        speaker_time_sec=t_speaker_1 - t_audio_1,
        render_time_sec=t_render_1 - t_render_0,
        review_time_sec=t_review_1 - t_review_0,
        total_time_sec=t_end - t_start,
        reel_path=render_result.reel_path,
        still_path=render_result.still_path,
        debug_path=render_result.debug_path,
        validation=validation,
        manifest=manifest,
        ai_review=ai_review,
        still_paths=render_result.still_paths,
    )


def _build_video_manifest(
    video_path: Path,
    stem: str,
    metadata: VideoMetadata,
    tracks: list[TrackedPerson],
    segments: list[ActiveSpeakerSegment],
    trajectory: list[CropTrajectoryPoint],
    reel_path: str,
    still_path: str,
) -> dict[str, Any]:
    """Build a manifest dict for the video reel, compatible with the validator."""
    import hashlib

    input_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()[:16]

    # Use build_manifest from the image pipeline as a base, override for video
    # The validator expects: asset_type, asset_name, input_hash, input_path,
    # output_path, output_dimensions, crop_params, validation_passed, etc.
    first_point = trajectory[0] if trajectory else None
    crop_params = {
        "x": first_point.crop_x if first_point else 0,
        "y": first_point.crop_y if first_point else 0,
        "w": first_point.crop_w if first_point else 0,
        "h": first_point.crop_h if first_point else 0,
        "edge_cutoffs": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "trajectory": [p.to_dict() for p in trajectory[::max(1, len(trajectory) // 100)]],
        "speaker_timeline": [s.to_dict() for s in segments],
        "num_tracks": len(tracks),
    }

    return {
        "asset_type": "video_reel",
        "asset_name": f"{stem}_9_16",
        "input_hash": input_hash,
        "input_path": str(video_path),
        "input_dimensions": {"width": metadata.width, "height": metadata.height},
        "output_path": reel_path,
        "output_dimensions": {"width": first_point.crop_w if first_point else 1080,
                              "height": first_point.crop_h if first_point else 1920},
        "crop_params": crop_params,
        "audio_present": metadata.audio_codec is not None,
        "processing_time_sec": 0.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_passed": None,
        "validation_warnings": [],
        "speaker_segments": [s.to_dict() for s in segments],
        "tracks": [{"id": t.id, "num_detections": len(t.face_bboxes)} for t in tracks],
    }


def _save_artifacts(
    stem: str,
    output_dir: Path,
    metadata: VideoMetadata,
    tracks: list[TrackedPerson],
    shot_bounds: list[int],
    speech_activity: SpeechActivity,
    segments: list[ActiveSpeakerSegment],
    trajectory: list[CropTrajectoryPoint],
    render_result: Any,
    validation: Any,
    ai_review: VideoReviewResult | None = None,
) -> None:
    """Save all debug artifacts as JSON files."""
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # video_metadata.json
    (manifests_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2),
    )

    # person_tracks.json
    tracks_data = []
    for track in tracks:
        tracks_data.append({
            "id": track.id,
            "num_detections": len(track.face_bboxes),
            "bboxes": [
                {"frame": f.frame_idx, "time": f.timestamp,
                 "x": f.x, "y": f.y, "w": f.width, "h": f.height,
                 "confidence": f.confidence}
                for f in track.face_bboxes
            ],
        })
    (manifests_dir / f"{stem}_person_tracks.json").write_text(
        json.dumps(tracks_data, indent=2),
    )

    # active_speaker_timeline.json
    (manifests_dir / f"{stem}_active_speaker_timeline.json").write_text(
        json.dumps([s.to_dict() for s in segments], indent=2),
    )

    # crop_trajectory.json (save every Nth point to keep file reasonable)
    step = max(1, len(trajectory) // 500)
    (manifests_dir / f"{stem}_crop_trajectory.json").write_text(
        json.dumps([p.to_dict() for p in trajectory[::step]], indent=2),
    )

    # validation.json
    validation_data = {
        "passed": validation.passed,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "checks": render_result.validation,
    }
    (manifests_dir / f"{stem}_validation.json").write_text(
        json.dumps(validation_data, indent=2),
    )

    # shot_boundaries.json
    (manifests_dir / f"{stem}_shot_boundaries.json").write_text(
        json.dumps({"boundaries": shot_bounds, "timestamps": [b / metadata.fps for b in shot_bounds]},
                     indent=2),
    )

    # speech_activity.json (summary)
    (manifests_dir / f"{stem}_speech_activity.json").write_text(
        json.dumps(speech_activity.to_dict(), indent=2),
    )

    # ai_review.json — Gemini post-rendering visual review (if available)
    if ai_review is not None:
        review_data = {
            "passed": ai_review.passed,
            "score": ai_review.score,
            "frames_reviewed": ai_review.frames_reviewed,
            "errors": ai_review.errors,
            "warnings": ai_review.warnings,
            "suggestions": ai_review.suggestions,
        }
        (manifests_dir / f"{stem}_ai_review.json").write_text(
            json.dumps(review_data, indent=2),
        )
