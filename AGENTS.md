# AGENTS.md — ReframeAI

Subject-aware media pipeline: master image → 4 subject-aware crops; master video → vertical reel + still. Every output validated against `spec/platform_spec.yaml`.

## Dev environment

- Python 3.14 (`.python-version` pinned). Use `uv` everywhere:
  ```bash
  uv venv --python 3.14 .venv && source .venv/bin/activate && uv pip install numpy pillow pyyaml pytest
  ```
- `requirements.txt` is output of `uv pip freeze` — regenerate with that command, don't hand-edit.
- `GEMINI_API_KEY` env var required for AI features. Copy `.env.example → .env`, never commit `.env`.
- ffmpeg/ffprobe must be on PATH (validator uses ffprobe to check video audio tracks).

## Build & test

```bash
source .venv/bin/activate
python -m pytest tests/ -v          # 33 tests, ~0.4s
python -m pytest tests/ -x          # stop on first failure
```

Pytest config is in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`).

## Conventions

- Source lives in `src/` — import as `from src.module import ...` (package root has `pythonpath=["."]`).
- `from __future__ import annotations` at the top of every `.py` file.
- Dataclasses for data (`Subject`, `CropResult`, `ValidationResult`).
- Config centralized in `src/config.py` — all paths, constants, API key helpers live there.
- Tests: fixtures in `tests/conftest.py`, test classes grouped by module (`TestSpecLoading`, `TestCropper`, etc.).
- Manifest format is JSON (see `spec/platform_spec.yaml` → `manifest_format.required_fields`).
- Spec is YAML — edit `spec/platform_spec.yaml` to change validation rules.

## Architecture notes

- Cropper (`src/cropper.py`) is pure arithmetic — deterministic, no ML. Takes subjects + ratio → crop rect.
- Validator (`src/validator.py`) enforces: dimensions, aspect ratio (±2%), subject framing, watermark overlap, audio presence, required metadata fields.
- Output dirs: `output/image_variants/`, `output/video_reels/`, `output/stills/`, `output/manifests/` (all gitignored).

## Pitfalls

- `data/` is gitignored — sample assets (`input_image.png`, `input_video.mp4`) exist locally only; don't reference them in committed code as hard paths unless via `config.DATA_DIR`.
- Validator's file-dimension check (`verify_files=True`) uses PIL on the actual output file — tests create temp files with `PIL.Image.new(...)`.
- Video audio check requires `ffprobe`; tests with `verify_files=False` skip disk checks.
- `.env` is gitignored — if tests or runtime code needs `GEMINI_API_KEY`, provide it or expect graceful degradation.
