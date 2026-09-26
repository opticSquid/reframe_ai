# ReframeAI

Subject-aware media pipeline for hackathon asset generation.

## Overview

Takes a master image and/or video and produces subject-aware variants:

- **Image**: 4 crops (16:9, 1:1, 9:16, 4:5) with dynamic subject framing
- **Video**: Speaker-aware vertical reel (9:16) with dynamic tracking
- **Still**: Best frame extracted from video

Every output is validated against a machine-readable platform spec before entering the asset library.

## Architecture

```
Gemini (AI reasoning)    Deterministic CV/Media
"What matters?"          "Where is it?"     "How to crop?"
        |                    |                  |
        v                    v                  v
  +--------------------------------------------------+
  |              Pipeline                            |
  |  - Spec validation                               |
  |  - Subject-aware crop computation                |
  |  - Scene detection + speaker ID                  |
  |  - FFmpeg rendering + re-encoding               |
  +--------------------------------------------------+
```

## Development

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install numpy pillow pyyaml

# Run tests
python -m pytest tests/ -v

# Run pipeline (Phase 2+)
python -m src.pipeline --help
```

## Dependencies

- Python 3.14+
- ffmpeg 8+
- See `pyproject.toml` for Python packages

## Deployment

Backend: Railway (single Docker container)
Frontend: Vercel (Next.js)
