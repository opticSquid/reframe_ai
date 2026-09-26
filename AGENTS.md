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
- ffmpeg/ffprobe must be on PATH (validator uses ffprobe to check video audio tracks).

## Build & test

```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 5000   # run FastAPI backend
python -m pytest tests/ -v          # 57 tests, ~7s
python -m pytest tests/ -x          # stop on first failure
```

Pytest config is in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`).

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
- **Backend** (`src/main.py`): FastAPI app with CORS; run with `uvicorn src.main:app --host 0.0.0.0 --port 5000`.
- `scripts/`: spike/demo scripts (`spike_crop.py`, `demo_regeneration.py`, `test_gemini.py`).
- Output dirs: `output/image_variants/`, `output/video_reels/`, `output/stills/`, `output/manifests/` (all gitignored).

## Pitfalls

- `data/` and `models/` are gitignored — sample assets (`input_image.png`, `input_video.mp4`) and MediaPipe model files (`.tflite`, `.task`) exist locally only; reference via `config.DATA_DIR` / `config.MODELS_DIR` in code, not hard paths.
- MediaPipe detection requires model files in `models/` — `detect_subjects()` returns an empty list (not an error) when a model is missing, so `plan_crop` falls back to center crop.
- Validator's file-dimension check (`verify_files=True`) uses PIL on the actual output file — tests create temp files with `PIL.Image.new(...)`.
- Video audio check requires `ffprobe`; tests with `verify_files=False` skip disk checks.
- `.env` is gitignored — if tests or runtime code needs `GEMINI_API_KEY`, provide it or expect graceful degradation.
- Docker: backend port 5000, frontend port 3000. Run `docker compose up --build` for local multi-service.
