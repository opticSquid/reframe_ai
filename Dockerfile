FROM python:3.14-slim AS base
WORKDIR /app

# System deps required by MediaPipe (libGL, libgthread, etc.)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv && \
    uv venv --python 3.14 .venv

COPY pyproject.toml requirements.txt ./
RUN source .venv/bin/activate && uv pip install --no-cache-dir -r requirements.txt

FROM base AS runtime
COPY --from=deps /app/.venv .venv
COPY src/ ./src/
COPY spec/ ./spec/
COPY models/ ./models/
COPY data/ ./data/

ENV VIRTUAL_ENV=/app/.venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 5000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]
