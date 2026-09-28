# ReframeAI

Subject-aware media pipeline. Takes a master image and produces 4 subject-aware crops (16:9, 1:1, 9:16, 4:5). Takes a master video and produces a vertical 9:16 speaker-aware reel + 4 stills + debug overlay. Every output is validated against `spec/platform_spec.yaml`.

## Quick start

```bash
# 1. Create .env and add your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your-key-here

# 2. Backend
uv sync --all-extras
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 5000

# 3. Frontend (in a new terminal)
cd frontend
npm install
npm run dev     # serves on http://localhost:3000

# 4. Tests
cd /home/soumalya/Work/reframe_ai
python -m pytest tests/ -v    # 57 tests, ~7s
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
  src/              # Python backend (FastAPI app, pipeline, validator, config)
  src/video_*.py    # Video pipeline modules (segmentation, perception, audio, rendering)
  src/regeneration.py  # AI-assisted regeneration loop
  frontend/         # Next.js frontend (TypeScript, Tailwind, App Router)
  spec/             # Machine-readable platform spec (YAML)
  tests/            # Pytest test suite (57 tests)
  data/             # Input assets (gitignored)
  output/           # Generated outputs (gitignored)
  Dockerfile        # Backend image
  Dockerfile.frontend  # Frontend image
  docker-compose.yml   # Multi-service orchestration
```

## Environment

- Python 3.14 (pinned in `.python-version`) — use `uv` for all package management
- `GEMINI_API_KEY` required for AI features (crop planning, output critique) — copy `.env.example` to `.env`
- ffmpeg/ffprobe on PATH (validator uses ffprobe for video/audio checks)
- Node 20+ for frontend development
- MediaPipe model files are committed in `models/` (21MB total) — no download needed

### MediaPipe models

The following models are included in `models/` (Git LFS not required, files are small):

| File | Size | Used by |
|------|------|---------|
| `blaze_face_full_range_sparse_float16.tflite` | 664K | Face detection |
| `face_landmarker_float16.task` | 3.6M | Face landmarks, MAR (speaking detection) |
| `efficientdet_lite0_float16.tflite` | 7.0M | Person/object detection |
| `pose_landmarker_full_float16.task` | 9.0M | Pose estimation (crop planning) |

### `.env` file

Create `.env` from the template before running:

```bash
cp .env.example .env
```

Required variables:

| Variable              | Description                        |
|-----------------------|------------------------------------|
| `GEMINI_API_KEY`      | Google Gemini API key (AI features)|
| `BACKEND_HOST`        | Backend bind host (default 0.0.0.0)|
| `BACKEND_PORT`        | Backend port (default 5000)        |
| `NEXT_PUBLIC_BACKEND_URL` | Frontend→backend API URL     |

## Architecture

Three layers:
1. **AI layer** (Gemini): semantic reasoning — crop center/coverage recommendations, framing critique, output review. Called once per pipeline (pre-render plan + post-render review).
2. **Deterministic layer** (MediaPipe, FFmpeg, numpy/Pillow): face/pose detection, speaker tracking, shot boundary detection, speech activity detection, crop computation, video rendering.
3. **Validator**: enforces spec — dimensions (min 720×1280 for reels), aspect ratio (±2%), subject framing, audio presence, required metadata fields.

### Video pipeline flow

```
Input video → Ingest (ffprobe) → Perception (MediaPipe faces) → Audio (SAD)
  → Speaker (active timeline) → Segment finding (best 30s window)
  → Gemini crop planning → Build trajectory → Render (ffmpeg)
  → Validate (spec) → Gemini review → Quality metrics → Retry if needed
```

Videos ≤30s use the entire video as the reel segment.

## Phases

- Phase 1: Foundation (spec, validator, cropper) — complete, 33 tests
- Phase 2: Image pipeline (MediaPipe detection, 4 crops) — complete
- Phase 3: Video pipeline (speaker-aware reel + 4 stills + debug) — complete
- Phase 4: CLI, regeneration, end-to-end — complete (57 tests)