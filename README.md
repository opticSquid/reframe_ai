# ReframeAI

Subject-aware media pipeline MVP. Takes a master image and produces 4 subject-aware crops (16:9, 1:1, 9:16, 4:5). Takes a master video and produces a speaker-aware vertical reel + best still. Every output is validated against `spec/platform_spec.yaml`.

## Quick start

```bash
# Backend
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 5000

# Frontend
cd frontend
npm install
npm run dev

# Tests (backend)
cd /home/soumalya/Work/reframe_ai
source .venv/bin/activate
python -m pytest tests/ -v    # 33 tests, ~0.5s
```

Health endpoint: `GET http://localhost:5000/health` → `{"status":"ok"}`

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Services: `backend` (port 5000), `frontend` (port 3000).

## Repository structure

```
reframe_ai/
  src/          # Python backend (FastAPI app, cropper, validator, config)
  frontend/     # Next.js frontend (TypeScript, Tailwind, App Router)
  spec/         # Machine-readable platform spec (YAML)
  tests/        # Pytest test suite
  data/         # Input assets (gitignored)
  output/       # Generated outputs (gitignored)
  Dockerfile    # Backend image
  Dockerfile.frontend  # Frontend image
  docker-compose.yml   # Multi-service orchestration
```

## Environment

- Python 3.14 (pinned in `.python-version`)
- `GEMINI_API_KEY` required for AI features — copy `.env.example` to `.env`
- ffmpeg/ffprobe on PATH (validator uses ffprobe for video audio checks)
- Node 20+ for frontend development

## Architecture

Three layers:
1. **AI layer** (Gemini): semantic reasoning — subject importance scoring, speaker ID, output critique. Sparse calls on key frames only.
2. **Deterministic layer** (MediaPipe, FFmpeg, numpy/Pillow): face/pose detection, scene detection, crop computation, rendering.
3. **Validator**: enforces spec — dimensions, aspect ratio, subject framing, watermark preservation, audio presence.

## Phases

- Phase 1: Foundation (spec, validator, cropper) — complete, 33 tests passing
- Phase 2: Image pipeline (MediaPipe detection, 4 crops) — not started
- Phase 3: Video pipeline (speaker-aware reel + still) — not started
- Phase 4: CLI, regeneration, end-to-end — not started
