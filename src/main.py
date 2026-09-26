"""ReframeAI backend — FastAPI application."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import OUTPUT_DIR
from .pipeline import process_image_to_variants, process_single_variant, ImagePipelineResult, VariantResult
from .video_pipeline import process_video_to_reel

app = FastAPI(title="ReframeAI Backend", version="0.1.0")

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
