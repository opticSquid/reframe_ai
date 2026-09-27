# AGENTS.md — ReframeAI

Subject-aware media pipeline: master image → 4 subject-aware crops; master video → vertical reel + still. Every output validated against `spec/platform_spec.yaml`.

## Dev environment

- Python 3.14 (`.python-version` pinned). Use `uv` everywhere:
  ```bash
  uv venv --python 3.14 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
  ```
- `requirements.txt` is output of `uv pip freeze` — regenerate with that command, don't hand-edit.
- `GEMINI_API_KEY` env var required for AI features. Copy `.env.example → .env`, never commit `.env`.
  Other vars from `.env.example`: `BACKEND_HOST`, `BACKEND_PORT` (backend server),
  `NEXT_PUBLIC_BACKEND_URL` (frontend→backend API URL).
- Video config in `src/config.py`: `VIDEO_SAMPLE_FPS` (4.0 — face detection/tracking sampling rate), `VIDEO_LANDMARK_FPS` (2.0 — face landmark/MAR sampling rate), `REEL_ASPECT_RATIO` (9/16 — vertical), `VIDEO_KEYFRAME_INTERVAL` (25 — every N frames for keyframe extraction).
- ffmpeg/ffprobe must be on PATH (validator uses ffprobe to check video audio tracks).

## Build & test

```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 5000        # run FastAPI backend
cd frontend && npm install && npm run dev             # run Next.js frontend (port 3000)
python -m pytest tests/ -v          # 57 tests, ~7s
python -m pytest tests/ -x          # stop on first failure
```

Pytest config is in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`).
Frontend needs Node 20+ (see `frontend/package.json`).

## Conventions

- Source lives in `src/` — import as `from src.module import ...` (package root has `pythonpath=["."]`).
- `from __future__ import annotations` at the top of every `.py` file.
- Dataclasses for data (`Subject`, `CropResult`, `CropPlan`, `ValidationResult`, `AIEvaluationResult`).
- Config centralized in `src/config.py` — all paths, constants, API key helpers live there.
- Tests: fixtures in `tests/conftest.py`, test classes grouped by module (`TestSpecLoading`, `TestCropper`, etc.).
- Manifest format is JSON (see `spec/platform_spec.yaml` → `manifest_format.required_fields`).
- Spec is YAML — edit `spec/platform_spec.yaml` to change validation rules.

## Architecture notes

- **Cropper** (`src/cropper.py`): pure arithmetic, deterministic, no ML. Takes subjects + ratio → crop rect.
- **Image crop** (`src/image_crop.py`): MediaPipe BlazeFace detection + crop planning (`plan_crop`) and rendering (`render_crop`). Separates planning from pixel rendering.
- **Regeneration** (`src/regeneration.py`): AI-assisted generate→validate→feedback→retry loop. Gemini provides semantic crop recommendations and critique; validator runs as-is.
- **Validator** (`src/validator.py`): enforces dimensions, aspect ratio (±2%), subject framing, watermark overlap, audio presence, required metadata fields.
- **Backend** (`src/main.py`): FastAPI app with CORS. Endpoints: `GET /health`, `POST /process-image`, `POST /process-image/variant/{ratio_name}`, `POST /process-video` (upload video → vertical 9:16 reel + stills + debug overlay). Run with `uvicorn src.main:app --host 0.0.0.0 --port 5000`.
- **Pipeline** (`src/pipeline.py`): top-level orchestrator — `process_image_to_variants` (all 4 ratios) and `process_single_variant`. Ties together detection→AI planning→rendering→validation→regeneration.
- **Video pipeline** (`src/video_pipeline.py`): orchestrator — `process_video_to_reel`. Ties together ingestion→perception→audio→speaker inference→crop trajectory→rendering→validation.
- **Video ingestion** (`src/video_ingestion.py`): `VideoMetadata` dataclass + `extract_video_metadata` via ffprobe (duration, FPS, codec, audio tracks, total frames).
- **Video perception** (`src/video_perception.py`): `VideoPerceiver` (reusable MediaPipe BlazeFace detector + face landmarker for MAR), `SimpleTracker` (IOU+centroid), frame extraction at reduced resolution, shot boundary detection.
- **Video audio** (`src/video_audio.py`): `SpeechActivity` + `detect_speech_activity` via ffmpeg audio extraction + RMS energy threshold. No librosa/scipy/torchaudio deps.
- **Video speaker** (`src/video_speaker.py`): active speaker inference combining audio SAD + MAR. `infer_active_speaker_timeline` produces speaker segments; `build_crop_trajectory` produces per-frame crop rectangles using `compute_crop` from `src/cropper`.
- **Video rendering** (`src/video_rendering.py`): `render_vertical_video` — pipe-based ffmpeg decode→Python crop→ffmpeg encode. Generates vertical 9:16 reel, stills in 4 aspect ratios (1:1, 16:9, 9:16, 4:5), and debug overlay with track bboxes + active speaker label + crop rectangle.
- **Video review** (`src/video_review.py`): `gemini_review_video` — post-rendering AI visual review. Selects representative frames at shot boundaries + speaker transitions, composites them into a grid, sends a SINGLE Gemini call with full trajectory/speaker metadata to assess framing quality, speaker tracking, and composition. Does NOT re-plan crops (deterministic only).
- **Image utils** (`src/_image_utils.py`): helpers for converting between numpy arrays and PNG bytes.
- `scripts/`: spike/demo scripts (`spike_crop.py`, `demo_regeneration.py`, `test_gemini.py`, `run_image_mvp.py` — end-to-end image pipeline runner, `run_video_mvp.py` — end-to-end video pipeline runner).
- Output dirs: `output/image_variants/`, `output/video_reels/`, `output/stills/`, `output/manifests/` (all gitignored).

## Pitfalls

`data/` is gitignored (sample assets like `input_image.png`, `input_video.mp4` are local-only). `models/` is committed (21MB total) — MediaPipe model files (`.tflite`, `.task`), so no download is needed. Reference both via `config.DATA_DIR` / `config.MODELS_DIR` in code, not hard paths.
- MediaPipe detection requires model files in `models/` — `detect_subjects()` returns an empty list (not an error) when a model is missing, so `plan_crop` falls back to center crop.
- Validator's file-dimension check (`verify_files=True`) uses PIL on the actual output file — tests create temp files with `PIL.Image.new(...)`.
- Video audio check requires `ffprobe`; tests with `verify_files=False` skip disk checks.
- `.env` is gitignored — if tests or runtime code needs `GEMINI_API_KEY`, provide it or expect graceful degradation.
- Docker: backend port 5000, frontend port 3000. Run `docker compose up --build` for local multi-service. Frontend built via `Dockerfile.frontend` (Node → nginx serving static export).
- Video: face bboxes are detected at reduced resolution (640px max) for speed — scale back to source resolution (1920x1080) before passing to `build_crop_trajectory`, or crop coordinates will be off by 3x.
- Video: VAAPI hardware encoding is not supported on some AMD GPUs (encoder fails to open); use CPU `libx264 -preset superfast` as fallback. Decode at full resolution and pipe raw RGB through Python is the bottleneck for long videos.
- Video: the validator requires `min_width: 720, min_height: 1280` and `audio_required: true` for video reels (see `spec/platform_spec.yaml` → `asset_types.video_reel`). Reel is rendered at 720×1280 to meet spec minimums.
