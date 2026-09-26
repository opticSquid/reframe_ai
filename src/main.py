"""ReframeAI backend — FastAPI application."""
from __future__ import annotations

import uuid
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import OUTPUT_DIR, PLATFORM_SPEC_PATH
from .pipeline import process_image_to_variants, process_single_variant, ImagePipelineResult, VariantResult
from .video_pipeline import process_video_to_reel

app = FastAPI(title="ReframeAI Backend", version="0.1.0")

# Serve output files statically so the frontend can display them
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    max_retries: int = Form(default=3),
    use_ai: bool = Form(default=True),
) -> JSONResponse:
    """Process an uploaded image into all 4 platform-ready variants.

    Steps:
      1. Save uploaded file to a temp directory under output/
      2. Run the full pipeline (PERCEPTION → REASONING → PLANNING → RENDER → VALIDATE/AI REVIEW)
      3. Return asset paths and validation status

    The temp file and all outputs are under output/uploads/<uuid>/.
    """
    upload_dir = OUTPUT_DIR / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    input_path = upload_dir / f"input_{file.filename}"
    content = await file.read()
    input_path.write_bytes(content)

    # Run pipeline
    result: ImagePipelineResult = process_image_to_variants(
        image_path=input_path,
        output_dir=upload_dir,
        max_retries=max_retries,
        use_ai=use_ai,
    )

    return JSONResponse(content=result.to_summary())


@app.post("/process-image/variant/{ratio_name}")
async def process_image_variant(
    ratio_name: str,
    file: UploadFile = File(...),
    max_retries: int = Form(default=3),
    use_ai: bool = Form(default=True),
) -> JSONResponse:
    """Process an uploaded image into a single platform variant (e.g. '9_16')."""
    upload_dir = OUTPUT_DIR / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    input_path = upload_dir / f"input_{file.filename}"
    content = await file.read()
    input_path.write_bytes(content)

    result: VariantResult = process_single_variant(
        image_path=input_path,
        ratio_name=ratio_name,
        output_dir=upload_dir,
        max_retries=max_retries,
        use_ai=use_ai,
    )

    return JSONResponse(content={
        "ratio_name": result.ratio_name,
        "crop": {
            "x": result.plan.crop.x,
            "y": result.plan.crop.y,
            "width": result.plan.crop.width,
            "height": result.plan.crop.height,
        },
        "render_path": result.render_path,
        "manifest_path": result.manifest_path,
        "passed": result.passed,
        "iterations": len(result.iterations),
        "ai_eval_passed": result.ai_evaluation.passed if result.ai_evaluation else None,
        "validation_errors": result.validation.errors,
        "validation_warnings": result.validation.warnings,
        "explanation": result.explanation,
        "ai_status": result.ai_status,
    })


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    generate_debug: bool = Form(default=True),
) -> JSONResponse:
    """Process an uploaded video into a vertical 9:16 speaker-aware reel."""
    upload_dir = OUTPUT_DIR / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    input_path = upload_dir / f"input_{file.filename}"
    content = await file.read()
    input_path.write_bytes(content)

    result = process_video_to_reel(
        video_path=input_path,
        output_dir=upload_dir,
        generate_debug=generate_debug,
    )

    return JSONResponse(content=result.to_summary())


@app.post("/process")
async def process_upload(
    request: Request,
    file: UploadFile = File(...),
    use_ai: bool = Form(default=True),
    generate_debug: bool = Form(default=True),
) -> StreamingResponse:
    """Unified endpoint: auto-detects image vs video, streams progress via SSE.

    Accepts any image or video file. Runs the appropriate pipeline
    (image pipeline for images, video pipeline for videos). Progress is
    streamed as Server-Sent Events so the frontend can update the user
    in real time instead of a blank screen.

    Events:
      - ``progress``: {"message": "...", "percent": 15}
      - ``complete``: {"complete": true, "summary": {...}, "type": "video"|"image"}
      - ``error``: {"error": "..."}  (on failure)
    """
    upload_dir = OUTPUT_DIR / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"input_{file.filename}"
    content = await file.read()
    input_path.write_bytes(content)

    # Detect type by extension
    filename = file.filename or "input"
    ext = Path(filename).suffix.lower()
    is_video = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv")

    async def event_stream():
        import asyncio
        import threading
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done_event = asyncio.Event()
        result_box: list = []

        def threaded_callback(msg: str, pct: int):
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"message": msg, "percent": pct}))

        def run_pipeline():
            try:
                if is_video:
                    from .video_pipeline import process_video_to_reel
                    r = process_video_to_reel(
                        video_path=input_path,
                        output_dir=upload_dir,
                        generate_debug=generate_debug,
                        progress_callback=threaded_callback,
                    )
                else:
                    from .pipeline import process_image_to_variants
                    r = process_image_to_variants(
                        image_path=input_path,
                        output_dir=upload_dir,
                        use_ai=use_ai,
                        progress_callback=threaded_callback,
                    )
                loop.call_soon_threadsafe(result_box.append, r)
            except Exception as e:
                loop.call_soon_threadsafe(result_box.append, e)
            finally:
                loop.call_soon_threadsafe(done_event.set)

        t = threading.Thread(target=run_pipeline, daemon=True)
        t.start()

        # Stream progress + drain remaining
        while not (done_event.is_set() and queue.empty()):
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {event_data}\n\n"
            except asyncio.TimeoutError:
                continue
        while not queue.empty():
            yield f"data: {await queue.get()}\n\n"

        await asyncio.wait_for(done_event.wait(), timeout=5.0)
        if result_box:
            result = result_box[0]
            if isinstance(result, Exception):
                yield f"data: {json.dumps({'error': str(result)})}\n\n"
            else:
                asset_type = "video" if is_video else "image"
                yield f"data: {json.dumps({'complete': True, 'summary': result.to_summary(), 'type': asset_type})}\n\n"
        else:
            yield f"data: {json.dumps({'error': 'Pipeline produced no result'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
