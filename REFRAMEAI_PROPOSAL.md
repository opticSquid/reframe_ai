# ReframeAI — Architecture Proposal

## Project Overview

Media pipeline that takes one master image and/or video and produces:

- **IMAGE**: 4 subject-aware crops (16:9, 1:1, 9:16, 4:5)
- **VIDEO**: One speaker-aware vertical reel (9:16) with dynamic subject tracking
- **STILL**: One extracted frame from the video
- **VALIDATION**: Every asset validated against a machine-readable platform spec

---

## Asset Inspection Summary

### Repository
- Empty bare repo, single commit (`ecfbbe3`)
- `.gitignore` ignores `data/`
- Files: `data/input_image.png`, `data/input_video.mp4`

### Input Image (`input_image.png`)
- **Format**: PNG, 4000x4000, 1:1, RGB
- **Content**: Two women in a temple/indoor setting investigating blood on the floor
- **Lighting**: Golden hour, directional from upper right
- **Watermark**: Semi-transparent "HOICCHO HACKATHON COPY" spans horizontal midline (center band, ~y=2000-2400), partially obscuring both subjects' faces
- **Subject location**: Bottom-center region of the frame

### Input Video (`input_video.mp4`)
- **Format**: H.264 + AAC, 1920x1080, 16:9, 190 seconds, 25fps, 4750 frames
- **Exported from**: DaVinci Resolve (has timecode/metadata tracks)
- **Audio**: AAC, 48kHz, stereo, ~320kbps
- **Watermark**: Same "HOICCHO HACKATHON COPY", same screen position
- **Scene breakdown** (8+ distinct scenes):
  | Timestamp | Scene | People | Subject/Speaker |
  |-----------|-------|--------|-----------------|
  | 0s | Talking head | 1 | Man (medium close-up) |
  | 25s | Hallway scene | ~6 | Man in white shirt, man in black suit |
  | 50s | Outdoor social | 3 | Man in center, two women flanking |
  | 75s | Courthouse | ~4+ | Young woman in purple sari (focus) |
  | 100s | Gift opening | ~6 | Man opening gift (center) |
  | 125s | Red box scene | 4 | Man opening box, woman watching |
  | 150s | Woman close-up | 1 | Woman in grey kurta (speaking) |
  | 175s | Woman close-up | 1 | Woman, pensive |

### Environment
- **CPU**: AMD Ryzen 5 3250U (integrated GPU, no CUDA)
- **RAM**: 13GB
- **Cores**: 4
- **Python**: 3.14.7
- **ffmpeg**: 8.1.3 (installed)
- **Pillow**: 12.3.0 (installed)
- **No ML frameworks**: numpy, torch, cv2, mediapipe NOT installed
- **Internet**: Works (HuggingFace returns 200)

---

## Critical Constraints from User

1. **Keep the watermark in output** — Legal reasons (same company as hackathon organizer). Watermark removal/inpainting is explicitly NOT to be done.
2. **12-hour time budget** — Must be efficient, no over-engineering
3. **Minimal cost** — No expensive GPU/cloud AI API calls unless justified
4. **Cloud deployment required** — Backend on Railway (single container), frontend on Vercel/Netlify
5. **AI-native** — AI is core to the pipeline, not a bolted-on feature. Must use real subject-aware detection, not center crops with face detection bolted on.
6. **Deterministic where possible** — Reproducible results
7. **User has strong software engineering experience** — Can review technical decisions
8. **`uv` for Python package management** — User prefers uv over pip/venv

---

## 1. Proposed Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              ReframeAI Pipeline                          │
                    │                                                         │
   input_image.png  │                                                         │  validated_assets/
   input_video.mp4  ├─────────┬─────────────────────────────────────────────┤ (manifests + media)
                    │         │                                             │
                    │  AI     │  Image Pipeline                              │
                    │  Layer  │  ─ Face/body detection (MediaPipe)           │
                    │         │  ─ Saliency + subject attention scoring       │
                    │         │  ─ Waterfall crop computation (4 ratios)      │
                    │         │  ─ → 4 validated image variants              │
                    │         │                                             │
                    │  AI     │  Video Pipeline                              │
                    │  Layer  │  ─ Shot boundary detection (scene cuts)       │
                    │         │  ─ Per-scene subject detection               │
                    │         │  ─ Speaker identification (audio+visual)      │
                    │         │  ─ Dynamic per-frame tracking + interpolation │
                    │         │  ─ Vertical reel crop (9:16)                  │
                    │         │  ─ Best frame extraction (still, 1:1)        │
                    │         │  ─ → 1 reel + 1 still, both validated        │
                    │         │                                             │
                    │  Det.   │  Validation Engine                           │
                    │  Layer  │  ─ Machine-readable spec (YAML)              │
                    │         │  ─ Dimension, aspect, framing, integrity     │
                    │         │  ─ Watermark awareness in checks              │
                    └─────────┴─────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────────────────────────────────────────┐
                    │  CLI / API Orchestration (pipeline.py)                  │
                    │  ─ uv-managed Python backend                            │
                    │  ─ FastAPI HTTP endpoint                                │
                    │  ─ Railway deployment (single container)                │
                    │                                                         │
                    │  Next.js Frontend (Vercel)                             │
                    │  ─ Upload master asset                                  │
                    │  ─ Trigger pipeline                                     │
                    │  ─ Preview results                                      │
                    │  ─ Download validated assets                            │
                    └─────────────────────────────────────────────────────────┘
```

### Key Design Decision: Watermark Handling

The watermark (semi-transparent "HOICCHO HACKATHON COPY") spans the center of every frame, obscuring faces and mouths. Since we are **keeping the watermark** (legal requirement), the approach is:

1. **Detect the watermark region** — Same screen position for both assets. Hardcode the ROI mask once.
2. **Mask the watermark band during face detection** — Face detection runs on non-watermark pixels only, with fallback logic.
3. **Use body pose as primary subject detector** — MediaPipe Pose gives full-body keypoints independent of face visibility. This is robust to watermark occlusion.
4. **Use object detection as supplementary** — YOLOv8n detects objects of interest (phones, blood, gifts, red boxes) that may be the true "subject" when faces are obscured.
5. **Speaker ID via audio-visual correlation** — Audio RMS peaks correlated with subject screen positions. Doesn't require clear lip visibility.

The watermark stays in all output assets. Validation checks that the watermark is present (integrity check) and that subjects are not entirely hidden behind it.

### Key Design Decision: Scene-Aware Video Processing

The video has 8+ distinct scenes with completely different characters and locations. A single continuous tracker would fail at scene boundaries. Approach:

1. **Shot boundary detection** first (using PySceneDetect or ffmpeg-based histogram comparison).
2. **Per-scene subject detection** — Each scene is processed independently.
3. **Speaker identification per scene** — Determine the speaker in each scene using audio correlation.
4. **Reel construction** — Concatenate per-scene speaker-aware crops into one vertical reel. Scene transitions get a brief smoothing/interpolation.

### Key Design Decision: Keyframe-Based Inference

Running MediaPipe Pose on all 4750 frames would take ~4-5 minutes per pass. Running face + pose + object detection on all frames would take ~15-20 minutes. Instead:

1. **Detect on keyframes** — Every ~1 second (25 frames apart).
2. **Interpolate crop windows** between keyframes — Linear or quadratic smoothing.
3. **Dense inference only where needed** — Speaker identification may need denser sampling during audio peaks.

Estimated inference time: ~3-5 minutes for the full 190s video (keyframes only).

---

## 2. Proposed Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Package Manager** | uv | User preference. Faster than pip, lockfile support. |
| **Language** | Python 3.14 | Already available |
| **Media I/O** | ffmpeg 8.1.3 (CLI) | Already installed, handles all codec/demux/encode |
| **Image I/O** | Pillow | Already installed |
| **Array ops** | numpy | Required for pixel math, fast to install via uv |
| **Face detection** | MediaPipe Face Detection | CPU-fast (~15-30ms/frame), Google's lightweight model |
| **Pose estimation** | MediaPipe Pose | CPU-fast (~30-50ms/frame), robust to face occlusion |
| **Object detection** | YOLOv8n via ultralytics OR MediaPipe Objects | CPU-runnable (~100ms/frame); choose based on model size |
| **Scene detection** | PySceneDetect | Lightweight shot boundary detection |
| **Speaker ID** | Custom audio-visual correlation (numpy + ffmpeg audio extraction) | No heavy model needed; audio RMS correlation with subject positions |
| **Web API** | FastAPI | Lightweight, async, single-file deployable |
| **Frontend** | Next.js (React) | Familiar, deployable on Vercel/Netlify |
| **Container** | Docker (single container for backend) | Railway deployment |
| **Cloud (Backend)** | Railway | Single-container deployment, user preference |
| **Cloud (Frontend)** | Vercel | User preference |

### Dependencies to install via `uv pip`:
```
numpy
pillow
mediapipe
ultralytics  (or use MediaPipe Objects instead — TBD)
scenedetect
opencv-python  (if needed for frame I/O — may avoid to save time)
fastapi
uvicorn
python-multipart  (for file uploads)
pyyaml  (for spec parsing)
```

### No:
- torch / transformers (too heavy, MediaPipe has its own runtime)
- Cloud AI APIs (Google Vision, AWS Rekognition) — unless local models prove insufficient
- Redis, databases, queues, message brokers

### Note on Gemini APIs
User mentioned Gemini APIs are free. If MediaPipe accuracy proves insufficient for speaker identification or if object detection misses key subjects, Gemini Vision API could be called as a fallback for specific frames. However, this adds network dependency and potential latency. Recommendation: try MediaPipe first, keep Gemini as a reserve option.

---

## 3. Proposed Repository Structure

```
reframe_ai/
├── data/                           # INPUT: sample assets (gitignored)
│   ├── input_image.png
│   └── input_video.mp4
├── src/
│   ├── __init__.py
│   ├── __main__.py                 # CLI entry: python -m src
│   ├── pipeline.py                 # Orchestration: CLI + FastAPI
│   ├── config.py                   # Settings, constants, paths
│   ├── image_processor.py          # Image → 4 crops
│   ├── video_processor.py          # Video → reel + still
│   ├── cropper.py                  # Subject-aware crop computation (shared)
│   ├── speaker.py                  # Audio-visual speaker identification
│   ├── validator.py                # Spec validation engine
│   ├── watermark.py                # Watermark detection/masking utilities
│   └── models/
│       ├── __init__.py
│       ├── media_pipe_detector.py  # Face + pose detection wrapper
│       ├── object_detector.py      # YOLOv8n or MediaPipe objects wrapper
│       └── scene_detector.py       # Shot boundary detection wrapper
├── spec/
│   ├── platform_spec.yaml          # Machine-readable output spec
│   └── validation_rules.json       # Concrete validation rules
├── tests/
│   ├── test_cropper.py
│   ├── test_validator.py
│   ├── test_watermark.py
│   └── conftest.py
├── frontend/                       # Next.js app
│   ├── pages/
│   │   ├── index.tsx              # Upload + preview
│   │   └── results.tsx            # Download results
│   ├── components/
│   ├── styles/
│   ├── package.json
│   └── tsconfig.json
├── output/                         # Runtime output (gitignored)
│   ├── image_variants/
│   ├── video_reels/
│   ├── stills/
│   └── manifests/                 # JSON manifests per asset
├── Dockerfile                      # Backend container
├── docker-compose.yml              # Backend + frontend (optional)
├── uv.lock                         # uv lockfile
├── pyproject.toml                  # uv project config
├── requirements.txt                # Generated from uv
├── .gitignore
└── README.md
```

---

## 4. AI Components

All AI components use **on-device, CPU-compatible models**. No cloud API calls by default.

1. **MediaPipe Face Detection** (`src/models/media_pipe_detector.py`)
   - Input: image frame (PIL/numpy array)
   - Output: list of face bounding boxes + keypoints (eyes, nose, mouth)
   - Used: Image pipeline (one-shot), video keyframes (sparse)
   - Robustness: Waterfall region masked during detection; body pose used as fallback

2. **MediaPipe Pose** (`src/models/media_pipe_detector.py`)
   - Input: image frame
   - Output: 33 body keypoints (shoulders, elbows, hips, knees, ankles, etc.)
   - Used: Image pipeline, video keyframes — **primary subject detector**
   - Robustness: Works when faces are watermarked out; gives body orientation

3. **YOLOv8n Object Detection** (`src/models/object_detector.py`)
   - Input: image frame
   - Output: list of detected objects (class, bbox, confidence)
   - Used: Image pipeline (detect objects of interest: phones, gifts, blood), video keyframes
   - Robustness: Detects non-face subjects; can be MediaPipe Objects as fallback
   - **Decision pending**: YOLOv8n (~100ms/frame on CPU) vs MediaPipe Objects (~50ms). Test both.

4. **Shot Boundary Detection** (`src/models/scene_detector.py`)
   - Input: video file
   - Output: list of scene cut timestamps
   - Used: Video pipeline — segment video into scenes for per-scene processing
   - Implementation: PySceneDetect (content detector + threshold)

5. **Audio-Visual Speaker Correlation** (`src/speaker.py`)
   - Input: video audio (extracted via ffmpeg) + per-frame subject positions
   - Output: per-frame speaker ID (which subject is speaking)
   - Algorithm:
     1. Extract audio RMS energy per frame via ffmpeg
     2. For each frame, find the subject whose screen position correlates with audio peaks
     3. Weight by: audio correlation strength, face orientation (toward camera = speaking), body orientation
   - Note: Doesn't require lip visibility (watermark may cover mouths). Uses head pose + audio.

6. **Saliency/Attention Scoring** (`src/cropper.py`)
   - Input: all AI detection outputs for a frame
   - Output: ranked list of subjects by "importance" for cropping
   - Scoring factors (deterministic, weighted):
     - Face detection confidence (highest weight)
     - Body keypoint completeness (fallback weight)
     - Subject size in frame (larger = more important)
     - Facing camera (head pose angle < 30° from camera axis)
     - Position relative to watermark (subjects behind watermark → lower confidence)
     - Object interaction (holding/dropping objects → higher importance)

---

## 5. Deterministic Components

1. **Aspect ratio crop math** — Given a subject bounding box, target ratio, and source dimensions, compute the crop rectangle. Pure arithmetic. Centers subject with configurable margin. No ML.

2. **Rule of thirds / golden ratio adjustment** — If subject is off-center, shift crop to satisfy composition rules. Deterministic given subject position.

3. **Edge safety margin** — Compute safe crop boundaries that don't cut off subject limbs. Uses body keypoints to determine subject extent. Deterministic.

4. **Temporal smoothing** — Given keyframe crop positions, interpolate intermediate frames using Hermite or linear interpolation. Prevents jitter. Deterministic.

5. **Frame extraction** — Exact timestamp-based frame grab via `ffmpeg -ss`. Deterministic.

6. **Video re-encoding** — `ffmpeg -filter:v crop -filter:v scale -c:v libx264 -b:v`. Fixed bitrate, deterministic encode settings.

7. **Validation engine** — Pure Python (stdlib + PyYAML). Reads JSON manifest, checks against YAML spec. All checks are deterministic, no randomness.

8. **Subject priority ranking** — Deterministic scoring formula given all AI outputs. Fixed weights, fixed formula.

9. **Reproducible processing** — All model inference results cached per-frame. All random seeds fixed. Same input → same output, verified by manifest hash.

10. **Manifest generation** — Every output asset gets a JSON manifest with: input hash, crop parameters, subject metadata, model outputs, validation results, timestamps. Enables regeneration.

---

## 6. External APIs / Models Required

| Model | Source | Download Size | CPU Speed (est) | CPU Notes |
|-------|--------|--------------|-----------------|-----------|
| MediaPipe Face Detection | mediapipe PyPI (auto-download) | ~2MB | ~15-30ms/frame | TF-Lite runtime, very fast |
| MediaPipe Pose | mediapipe PyPI (auto-download) | ~5MB | ~30-50ms/frame | TF-Lite runtime |
| YOLOv8n | ultralytics PyPI (auto-download first use) | ~6MB | ~100-200ms/frame | ONNX, CPU-compatible |
| MediaPipe Objects | mediapipe PyPI (auto-download) | ~10MB | ~50-100ms/frame | Alternative to YOLO |
| PySceneDetect | scenedetect PyPI | ~50MB (with ffmpeg deps) | ~2-3s/frame (video scan) | Uses histogram comparison |

**Total model weights**: ~15-25MB. All auto-download on first import.

No cloud AI APIs by default. If user approves and MediaPipe accuracy is insufficient, Gemini Vision API can be called as fallback for specific frames (see Risks).

---

## 7. Risks and Mitigations

### High Priority

1. **Watermark occlusion of faces/mouths** — Faces and mouths are behind the watermark. Face detection confidence may be low. Speaker ID may struggle.
   - **Mitigation**: Body pose as primary detector. Audio-visual correlation instead of lip-reading. Accept reduced face detection accuracy but don't let it block progress.

2. **Multiple unrelated scenes in video** — 8+ scenes with different characters. A naive tracker follows the wrong person across scenes.
   - **Mitigation**: Shot boundary detection first, then per-scene processing. Clear scene boundaries.

3. **No GPU** — CPU-only inference. 4750 frames at ~50ms/frame = ~4 min per pass. Multiple passes (face + pose + objects) could total 15+ min.
   - **Mitigation**: Keyframe-only inference (every 25 frames = 190 frames total). Interpolation between. Expected total: ~3-5 min for full video.

4. **Speaker ambiguity in group scenes** — 6 people in a hallway. Audio correlation alone may not uniquely identify the speaker.
   - **Mitigation**: Combine audio correlation with visual cues (face orientation, body orientation, gesturing). If ambiguous, choose the largest/center-most subject. Log confidence score.

5. **Gemini API cost/time uncertainty** — User mentioned Gemini is free, but API calls add network latency and potential rate limits. Using Gemini on 190 keyframes could add significant time.
   - **Mitigation**: Try MediaPipe first. Only call Gemini as fallback when local detection confidence is low. Cache results.

### Medium Priority

6. **Image crop for extreme ratios (9:16)** — Source is 1:1. Going to 9:16 crops to 1/3 width. Must choose which subject to keep when two are present.
   - **Mitigation**: Subject priority ranking (face + size + orientation + interaction). For the image, the woman with the phone (investigating) is likely primary. Validate by checking subject framing.

7. **Validation spec ambiguity** — The spec doesn't exist yet. Risk of too-loose or too-strict validation.
   - **Mitigation**: Start simple. Define spec with clear, measurable rules: dimensions, aspect ratio, subject center deviation < X%, watermark present, no black bars, audio present (video). Expand based on results.

8. **Still extraction frame selection** — Which frame is "best"?
   - **Mitigation**: Score each keyframe by: subject sharpness (edge density in face/body region), speaker activity (audio RMS), composition (subject centered, not too close to edge). Pick highest score. **See open question below.**

9. **YOLOv8n accuracy on CPU** — May be slow or miss small objects.
   - **Mitigation**: Test on sample frames first. If too slow, use MediaPipe Objects instead. If accuracy insufficient, use Gemini Vision for object detection on keyframes only.

### Low Priority

10. **DaVinci Resolve metadata** — Video has timecode tracks. May contain scene shot markers.
    - **Note**: Probably not useful for this task. Ignore unless needed.

11. **Audio quality** — AAC at ~320kbps is fine. Speaker correlation will work.

---

## 8. Recommended Implementation Order (MVP)

### Phase 1: Foundation (~2-3 hours)
1. Set up `uv` project (`pyproject.toml`, `uv.lock`, `.python-version`)
2. Write machine-readable platform spec (`spec/platform_spec.yaml`)
3. Write validation engine (`src/validator.py`) — THE source of truth for quality
4. Write cropper utility (`src/cropper.py`) — subject bbox → crop rectangle math. Pure, unit-testable

### Phase 2: Image Pipeline (~3-4 hours)
5. Install numpy + mediapipe + PIL, build detection wrapper
6. Build `image_processor.py`: detect subjects → score → compute 4 crops → validate → write output + manifest
7. Test on `input_image.png`, inspect results
8. Iterate on crop quality

### Phase 3: Video Pipeline (~3-4 hours)
9. Install scenedetect, build `scene_detector.py`
10. Build `speaker.py` (audio extraction + correlation)
11. Build `video_processor.py`: scene detection → per-scene detection → speaker ID → dynamic tracking → vertical reel + best still
12. Test on `input_video.mp4`, inspect results

### Phase 4: Full Pipeline + Frontend (~2-3 hours)
13. Build `pipeline.py` (CLI + FastAPI)
14. Add single-variant regeneration (read manifest → re-run single crop)
15. Build Next.js frontend (upload, trigger, preview, download)
16. Dockerize backend, write deployments configs
17. End-to-end test

**Total estimated**: ~10-13 hours for working MVP.

---

## 9. Recommended Explicit NON-Implementation

1. **Watermark removal/inpainting** — Legal reasons (same company as organizer). Watermark stays in output.
2. **Deep learning video trackers (DeepSORT, StrongSORT, BYTETracker)** — Too heavy for CPU. Per-frame pose + interpolation is sufficient.
3. **Large transformer models (Swin, ViT, DETR-base)** — Too slow on CPU. MediaPipe's distilled models are 10-100x faster.
4. **Cloud AI APIs by default** — Keep local. Gemini only as fallback if MediaPipe insufficient.
5. **Multi-model ensemble voting on every frame** — Only run heavy detection on keyframes (every 25 frames).
6. **Kalman filters / motion prediction** — Subjects are mostly sedentary. Linear interpolation is sufficient.
7. **GPU provisioning** — No NVIDIA GPU available. MediaPipe runs fine on CPU.
8. **Real-time streaming pipeline** — Input is files, not streams.
9. **Database, Redis, queues, microservices** — All state is local file-based (JSON manifests). Single process.
10. **Automatic watermark region detection** — Hardcode the ROI mask once (same position for both assets).
11. **Progressive web app / mobile app** — Next.js basic web UI is sufficient for MVP.
12. **User authentication** — No auth needed for hackathon MVP.
13. **Batch processing orchestration (Celery, RQ, etc.)** — Single-file processing via FastAPI. If parallelism needed, it's embarrassingly parallel.

---

## Open Questions for User Approval

### Q1: Still extraction strategy
Should the still frame be:
- **(a)** Best frame by score (sharpness + speaker activity + composition) — requires per-frame inference on keyframes, ~2-3 min
- **(b)** Fixed timestamp (e.g., middle of the dominant scene) — simplest, fully deterministic, instant

### Q2: Speaker-aware reel scope
The video has completely different characters per scene. Should the reel:
- **(a)** Follow the speaker scene-by-scene (subject switches every ~20s as scenes change) — "speaker-aware" in the truest sense
- **(b)** Follow a single protagonist across scenes where they appear (the man from frame 0, if he recurs) — more coherent narrative, but he only appears in 1 of 8 scenes

### Q3: Object detection model choice
YOLOv8n (~100ms/frame, 6MB) vs MediaPipe Objects (~50ms/frame, 10MB):
- YOLOv8n has better accuracy but is slower
- MediaPipe Objects is faster but may miss small objects
- **Recommendation**: Start with MediaPipe Objects. Switch to YOLOv8n if accuracy is insufficient.

### Q4: Gemini API as fallback
If MediaPipe face/pose detection produces low-confidence results on watermarked frames, should we:
- **(a)** Accept the reduced accuracy (simpler, no network dependency, no cost)
- **(b)** Fall back to Gemini Vision API for keyframes with low confidence (better quality, but adds API calls and latency)

### Q5: Frontend scope
Next.js with basic upload/preview/download is proposed. Is that sufficient, or do you want:
- **(a)** Basic web UI (upload, trigger, show results, download)
- **(b)** Additional features (asset comparison sliders, variant switching, manifest inspection)

### Q6: Single-variant regeneration implementation
When regenerating a single variant, should we:
- **(a)** Re-run only the crop math from the stored manifest parameters (no re-inference) — fastest, truly deterministic
- **(b)** Re-run AI detection + crop math — more flexible but adds inference time

---

## Notes

- The watermark "HOICCHO HACKATHON COPY" is visible in all source and output assets. This is intentional per legal requirements.
- The video's watermark position is consistent across all frames (confirmed by visual inspection of 8 sample frames).
- The image's watermark blocks the faces of both subjects at roughly y=2000-2400 (center band).
- No GPU is available. All models must run on CPU. MediaPipe is the best option for CPU inference speed.
- `uv` will manage all Python dependencies. Frontend (Next.js) will be in a separate `frontend/` directory with its own package management (npm/yarn).
