# Video Pipeline Flow

```
USER
  │
  ▼
┌──────────────────────────────────┐
│  Upload video  (MP4/MOV/AVI)     │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 1: INGEST                  │
│  VideoIngestion                    │
│  ffprobe → metadata                │
│  (duration, fps, codec, audio)     │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 2: PERCEPTION               │
│  VideoPerceiver (reusable)         │
│  • Sample frames @ 4fps, 640px     │
│  • Face detection (BlazeFace)      │
│  • SimpleTracker (IOU + centroid)  │
│  • Shot boundary detection         │
│  → tracks[] (TrackedPerson)        │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 3: AUDIO                    │
│  VideoAudio                        │
│  ffmpeg audio extract → PCM        │
│  RMS energy threshold → speech     │
│  mask                                │
│  → SpeechActivity                   │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 4: LANDMARKS (MAR)         │
│  VideoPerceiver (face landmarker)  │
│  • Sample frames @ 2fps            │
│  • Face landmarks → MAR            │
│  • Associate MAR to tracks         │
│  → track_mars: {track_id: [(t, mar)]}│
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 5: ACTIVE SPEAKER         │
│  VideoSpeaker                      │
│  Fuse: audio speech + visual MAR   │
│  • If audio active, pick face with │
│    highest MAR at each time window │
│  • Temporal smoothing (hold 0.3s)  │
│  • Merge adjacent same-speaker     │
│  → segments[] (ActiveSpeakerSegment)│
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 6: CROP TRAJECTORY         │
│  VideoSpeaker + Cropper            │
│  For each frame:                    │
│  • Find active speaker → track     │
│  • Predict face bbox (interpolated)│
│  • compute_crop() around speaker   │
│  • EMA smooth center (alpha=0.3)   │
│  → trajectory[] (per-frame CropResult)│
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 7: RENDER                    │
│  VideoRendering                    │
│  FFmpeg pipe: decode → Python crop │
│  • Apply per-frame crop trajectory │
│  • Resize to 720×1280 (9:16)       │
│  • libx264 CPU encode              │
│  • Audio mux (copy AAC)            │
│  • Still extraction (best frame)   │
│  • 4 stills: 1:1, 16:9, 9:16, 4:5  │
│  • Debug overlay (bboxes + labels) │
│  → reel.mp4, stills[], debug.mp4   │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 8: AI VISUAL REVIEW        │
│  VideoReview (optional)            │
│  Gemini post-rendering critique:   │
│  • Select 8-12 representative      │
│    frames (shot boundaries +       │
│    speaker transitions)            │
│  • Composite grid: original + crop │
│  • ONE API call with metadata      │
│  → VideoReviewResult (score, issues)│
│  ┌─── No key? ───→ skip ───────────┐
│  │                                   │
└─┴─────────────────────────────────┘ │
                                      │
          ┌──────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 9: VALIDATION              │
│  Validator (reuses image validator)│
│  • min 720×1280 for reel            │
│  • audio_required = true            │
│  • aspect ratio 9:16 (±2%)          │
│  → ValidationResult                 │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  OUTPUT                           │
│  • output/video_reels/<name>_9_16  │
│    .mp4 (with audio)                │
│  • output/video_reels/<name>_debug │
│    .mp4 (overlay)                   │
│  • output/stills/<name>_1_1.jpg     │
│  • output/stills/<name>_16_9.jpg    │
│  • output/stills/<name>_9_16.jpg    │
│  • output/stills/<name>_4_5.jpg     │
│  • output/manifests/                │
│    (8+ JSON debug artifacts)        │
└──────────────────────────────────┘
```

## Stages Summary

| Stage | Module | Key Function | Output |
|-------|--------|-------------|--------|
| 1. Ingest | `src/video_ingestion.py` | `extract_video_metadata()` | `VideoMetadata` |
| 2. Perception | `src/video_perception.py` | `VideoPerceiver.detect_faces()` + `SimpleTracker` | `list[TrackedPerson]` |
| 3. Audio | `src/video_audio.py` | `detect_speech_activity()` | `SpeechActivity` |
| 4. Landmarks | `src/video_perception.py` | `VideoPerceiver.compute_mars()` | `dict[track_id, (t, mar)]` |
| 5. Speaker | `src/video_speaker.py` | `infer_active_speaker_timeline()` | `list[ActiveSpeakerSegment]` |
| 6. Trajectory | `src/video_speaker.py` | `build_crop_trajectory()` | `list[CropTrajectoryPoint]` |
| 7. Render | `src/video_rendering.py` | `render_vertical_video()` | `RenderResult` (reel + stills) |
| 8. Review | `src/video_review.py` | `gemini_review_video()` | `VideoReviewResult` |
| 9. Validate | `src/validator.py` | `validate_asset()` | `ValidationResult` |

## Entry Point

`src/video_pipeline.py:process_video_to_reel()` — orchestrates all 9 stages with optional
`progress_callback` for real-time SSE streaming via the `POST /process` backend endpoint.

## Key Design Notes

- **MediaPipe detectors reused** — `VideoPerceiver` creates detector/landmarker once, not per-frame (0.02s/frame vs 0.4s/frame)
- **Face bboxes scaled** — detected at 640px max, scaled back to source resolution (1920×1080) before trajectory computation
- **CPU fallback** — `libx264 -preset superfast` instead of VAAPI (AMD GPU incompatibility)
- **Single Gemini call** — review stage sends ONE call with composite grid of representative frames, not per-frame inference
