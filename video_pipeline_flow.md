# Video Pipeline Flow

```mermaid
flowchart TD
    USER[User uploads video MP4/MOV/AVI] --> S1

    %% Stage 1: Ingest
    S1[STAGE 1: INGEST<br/>extract_video_metadata() in src/video_ingestion.py<br/>ffprobe → VideoMetadata<br/>(duration, fps, codec, audio tracks, total frames)] --> S2

    %% Stage 2: Perception
    S2[STAGE 2: PERCEPTION<br/>VideoPerceiver in src/video_perception.py<br/>• Sample frames @ 4fps, max_dim=640<br/>• Face detection (BlazeFace, reused detector)<br/>• SimpleTracker (IOU + centroid)<br/>• Shot boundary detection<br/>→ list[TrackedPerson]<br/>[bboxes scaled back to source resolution]] --> S3

    %% Stage 3: Audio
    S3[STAGE 3: AUDIO<br/>detect_speech_activity() in src/video_audio.py<br/>ffmpeg audio extract → mono 16kHz PCM<br/>RMS energy @ 50ms windows → speech mask<br/>→ SpeechActivity] --> S4

    %% Stage 4: Landmarks (MAR)
    S4[STAGE 4: LANDMARKS (MAR)<br/>VideoPerceiver.compute_mars() in src/video_perception.py<br/>• Sample frames @ 2fps<br/>• Face landmarker → MAR (Mouth Aspect Ratio)<br/>• Associate MAR to tracks by centroid proximity<br/>→ dict[track_id, list[(timestamp, mar)]]] --> S5

    %% Stage 5: Active Speaker
    S5[STAGE 5: ACTIVE SPEAKER<br/>infer_active_speaker_timeline() in src/video_speaker.py<br/>Fuse: audio speech + visual MAR<br/>• Highest MAR face during speech = speaker<br/>• Temporal smoothing: hold 0.3s minimum<br/>• Merge adjacent same-speaker segments<br/>→ list[ActiveSpeakerSegment]] --> S6

    %% Stage 6: Best Segment Finding
    S6[STAGE 6: BEST SEGMENT FINDING<br/>find_best_segment() in src/video_segment.py<br/>If video > VIDEO_SEGMENT_DURATION_SEC (30s):<br/>• Sliding window scored by weighted sum:<br/>  - Motion density (weight=0.30)<br/>  - Face activity (weight=0.25)<br/>  - Audio energy (weight=0.25)<br/>  - Shot changes (weight=0.20)<br/>If ≤30s: use entire video<br/>→ BestSegment (start, end, score)] --> S7

    %% Stage 7: Render + Review Feedback Loop
    S7{RENDER + REVIEW LOOP<br/>src/video_pipeline.py<br/>up to VIDEO_SEGMENT_MAX_RETRIES=3 iterations}

    %% 7a: Gemini Crop Planning
    S7 --> GP[Gemini Pre-Render Crop Planning<br/>gemini_plan_crop() in src/video_review.py<br/>• Select keyframes (shot boundaries + speaker transitions)<br/>• Overlay face bboxes (red) + pose landmarks (green) + proposed crop (cyan)<br/>• ONE Gemini call → recommended center + coverage<br/>If unavailable: deterministic fallback (speaker face centroid)]
    GP --> TRAJ

    %% 7b: Build Trajectory
    TRAJ[Build Crop Trajectory<br/>build_crop_trajectory() in src/video_speaker.py<br/>• For each frame: find active speaker → predict interpolated bbox<br/>• Asymmetric padding: pad_x=0.5, pad_y_top=0.5, pad_y_bottom=0.5<br/>• compute_crop() with target_coverage=0.30 (default)<br/>• EMA smooth center (alpha=0.2)<br/>• Min crop height = 35% source (prevents excessive zoom)<br/>• Trajectory built for FULL video, then sliced to segment<br/>→ list[CropTrajectoryPoint]] --> SLICE

    SLICE[Slice trajectory + tracks to segment<br/>start_frame:end_frame+1<br/>tracks filtered to segment bboxes] --> RENDER

    %% 7c: Render
    RENDER[STAGE 7c: RENDER<br/>render_vertical_video() in src/video_rendering.py<br/>FFmpeg pipe: decode → Python crop → ffmpeg encode<br/>• Resize to 720×1280 (9:16)<br/>• libx264 CPU encode (superfast, CRF 28)<br/>• Audio mux (copy AAC)<br/>• Best-frame still extraction @ 4 ratios (1:1, 16:9, 9:16, 4:5)<br/>• Debug overlay (track bboxes + speaker label + crop rect)<br/>→ RenderResult (reel + stills + debug)] --> QUAL

    %% 7d: Segment Quality
    QUAL[Segment Quality Evaluation<br/>evaluate_segment_quality() in src/video_review.py<br/>Deterministic metrics:<br/>• Face coverage (speaker face in crop)<br/>• Audio coverage (speech fraction)<br/>• Shot retention<br/>• Speaker diversity<br/>• Trajectory validity<br/>→ SegmentQuality] --> REVIEW

    %% 7e: Gemini Review
    REVIEW[STAGE 7e: AI VISUAL REVIEW<br/>gemini_review_video() in src/video_review.py<br/>• Select 8-12 representative frames (shot boundaries + speaker transitions)<br/>• Composite grid: original 16:9 + rendered 9:16 side by side<br/>• ONE Gemini call with full trajectory + speaker metadata<br/>→ VideoReviewResult (score, errors, suggestions)] --> CHECK

    %% Decision
    CHECK{Both quality + review pass?}
    CHECK -->|Yes| EXIT_LOOP[Exit loop with best result]
    CHECK -->|No| ADJUST[Retry: shift segment boundaries (±3s)<br/>Widen target_coverage (loosen crop)<br/>Feed previous review errors to gemini_plan_crop]
    ADJUST --> GP

    %% Post-loop
    EXIT_LOOP --> S8[STAGE 8: VALIDATION<br/>validate_asset() in src/validator.py<br/>• min 720×1280 for reel<br/>• audio_required = true (ffprobe check)<br/>• aspect ratio 9:16 (±2%)<br/>• Required metadata fields populated<br/>→ ValidationResult]

    S8 --> SAVE[Save outputs:<br/>• output/video_reels/{stem}_9_16.mp4<br/>• output/video_reels/{stem}_debug.mp4<br/>• output/stills/{stem}_{1_1|16_9|9_16|4_5}.jpg<br/>• output/manifests/{stem}_*.json (8 debug artifacts)]

    SAVE --> OUT[OUTPUT<br/>• Vertical 9:16 reel with audio<br/>• Debug overlay video<br/>• 4 still images (1:1, 16:9, 9:16, 4:5)<br/>• 8+ JSON debug artifacts<br/>  (metadata, tracks, speaker timeline,<br/>  crop trajectory, validation, shots,<br/>  speech activity, AI review)]
```

## Stages Summary

| Stage | Module | Key Function | Output |
|-------|--------|-------------|--------|
| 1. Ingest | `src/video_ingestion.py` | `extract_video_metadata()` | `VideoMetadata` |
| 2. Perception | `src/video_perception.py` | `VideoPerceiver.detect_faces()` + `SimpleTracker` | `list[TrackedPerson]` |
| 3. Audio | `src/video_audio.py` | `detect_speech_activity()` | `SpeechActivity` |
| 4. Landmarks | `src/video_perception.py` | `VideoPerceiver.compute_mars()` | `dict[track_id, list[(t, mar)]]` |
| 5. Speaker | `src/video_speaker.py` | `infer_active_speaker_timeline()` | `list[ActiveSpeakerSegment]` |
| 6. Best Segment | `src/video_segment.py` | `find_best_segment()` | `BestSegment` |
| 7. Render+Review Loop | `src/video_review.py` + `src/video_rendering.py` + `src/video_speaker.py` | `gemini_plan_crop()` → `build_crop_trajectory()` → `render_vertical_video()` → `evaluate_segment_quality()` → `gemini_review_video()` | `RenderResult` + `SegmentQuality` + `VideoReviewResult` |
| 8. Validate | `src/validator.py` | `validate_asset()` | `ValidationResult` |

## Entry Point

`src/video_pipeline.py::process_video_to_reel()` — orchestrates all 8 stages with an optional `progress_callback` for real-time SSE streaming via the `POST /process` backend endpoint.

## Key Design Notes

- **MediaPipe detectors reused**: `VideoPerceiver` creates the face detector, face landmarker, and pose landmarker once and reuses them across all frames (0.02s/frame vs 0.4s/frame for per-frame construction).
- **Face bboxes scaled**: Detected at 640px max resolution for speed, scaled back to source resolution (e.g. 1920×1080) before trajectory computation. Without this, crop coordinates are off by ~3x.
- **Asymmetric padding**: `build_crop_trajectory` applies `pad_x=0.5`, `pad_y_top=0.5`, `pad_y_bottom=0.5` because BlazeFace bboxes are face-only (forehead to chin) — shoulders and headroom must be added.
- **Minimum crop size**: When `target_ratio < 1.0` (portrait), crop height is enforced to at least 35% of source height to prevent excessive zoom-in from a single face bbox.
- **Trajectory for full video, sliced to segment**: `build_crop_trajectory` builds per-frame crops for the entire video, then `video_pipeline.py` slices to the best segment (`trajectory[start_frame:end_frame+1]`). Tracks are also sliced for debug overlay.
- **CPU fallback**: `libx264 -preset superfast -crf 28` instead of VAAPI (AMD GPU incompatibility — radeonsi only exposes decode profiles, no encode).
- **Single Gemini call (review)**: Post-rendering review sends ONE API call with a composite grid of representative frames + full trajectory/speaker/shot metadata.
- **Single Gemini call (pre-plan)**: `gemini_plan_crop` sends ONE call with a keyframe grid showing face bboxes + pose landmarks + proposed crop. Does NOT re-plan crops per-frame — deterministic tracker handles follow-through.
- **Retry loop adjusts two axes**: On failure, the loop shifts segment start time (±3s) and widens `target_coverage` (loosens the crop to keep more context) across up to 3 attempts.
- **30s segment trimming**: `find_best_segment` uses motion density + face activity + audio energy + shot boundaries as a weighted scoring function over a sliding window. If video ≤30s, the entire video is used.

### VideoPipelineResult fields

| Field | Type | Description |
|-------|------|-------------|
| `video_path` | str | Source video path |
| `metadata` | VideoMetadata | ffprobe-extracted metadata |
| `tracks` | list[TrackedPerson] | Face tracks across frames |
| `speech_activity` | SpeechActivity | Audio SAD result |
| `speaker_segments` | list[ActiveSpeakerSegment] | Who spoke when |
| `trajectory` | list[CropTrajectoryPoint] | Per-frame crop rectangles |
| `segment_start` / `segment_end` | float | Best segment time range |
| `best_segment` | BestSegment | Scoring result with per-window scores |
| `crop_plan` | GeminiCropPlanResult | Pre-render Gemini crop recommendation |
| `segment_quality` | SegmentQuality | Deterministic quality metrics |
| `ai_review` | VideoReviewResult | Post-render Gemini critique |
| `pipeline_attempts` | int | Number of render/review iterations (1–3) |
| `perception_time_sec` / `audio_time_sec` / `speaker_time_sec` / `render_time_sec` / `review_time_sec` / `total_time_sec` | float | Per-stage timing |
| `reel_path` / `still_path` / `debug_path` | str | Output file paths |
| `still_paths` | list[str] | 4 still image paths (1:1, 16:9, 9:16, 4:5) |
| `validation` | ValidationResult | Spec validation result |
| `manifest` | dict | Full video manifest |
