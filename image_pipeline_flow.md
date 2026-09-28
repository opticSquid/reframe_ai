# Image Pipeline Flow

```mermaid
flowchart TD
    USER[User uploads image PNG/JPG] --> INGEST[Load source image<br/>src.image_crop.detect_subjects]

    %% Stage 1: Perception (shared across all ratios)
    INGEST --> S1[STAGE 1: PERCEPTION<br/>MediaPipe BlazeFace detection<br/>detect_subjects() in src/image_crop.py<br/>→ tuple[list[Subject], str]<br/>model name returned alongside subjects]

    %% Stage 2 + 3 + 4 + 5 + 6 happen per-ratio inside regenerate_crop
    S1 --> FOR[Begin per-ratio loop<br/>Ratios: 16_9 → 1_1 → 9_16 → 4_5<br/>src.pipeline.process_image_to_variants]

    %% Regeneration loop (encapsulates stages 2-6 internally)
    FOR --> LOOP{REGENERATION LOOP<br/>src.regeneration.regenerate_crop<br/>max_retries iterations}

    LOOP --> S2[STAGE 2: AI PLANNING<br/>plan_crop_with_ai() in src/regeneration.py<br/>Gemini semantic analysis of image + subjects<br/>→ crop params dict]
    S2 -->|No GEMINI_API_KEY| DET[Fallback: deterministic params<br/>target_coverage=0.5, min_margin=0.1]
    DET --> S3

    S2 -->|Key available| GEM[Gemini recommends<br/>coverage, margin, center_x, center_y]
    GEM --> S3

    %% Stage 3: Crop Planning
    S3[STAGE 3: CROP PLANNING<br/>plan_crop() in src/image_crop.py<br/>Deterministic arithmetic from subjects + AI params<br/>→ CropPlan] --> S4

    %% Stage 4: Rendering
    S4[STAGE 4: RENDERING<br/>render_crop() in src/image_crop.py<br/>PIL/numpy slice of crop rectangle<br/>→ np.ndarray rendered variant] --> S5

    %% Stage 5: Validation
    S5[STAGE 5: VALIDATION<br/>validate_asset() in src/validator.py<br/>• Dimensions check (min/max)<br/>• Aspect ratio ±2%<br/>• Subject framing (edge cutoffs)<br/>• Watermark overlap<br/>→ ValidationResult] --> S6

    %% Stage 6: AI Evaluation (optional)
    S6[STAGE 6: AI EVALUATION<br/>evaluate_crop_with_ai() in src/regeneration.py<br/>Gemini critiques rendered crop for framing quality<br/>→ AIEvaluationResult] --> DEC

    %% Decision
    DEC{Decision}
    DEC -->|Both pass| PASS[PASS<br/>RegenerationResult returned]
    DEC -->|Either fails| FAIL[FAIL<br/>Construct structured feedback<br/>Use AI adjustment params<br/>→ Re-plan with adjusted params]
    FAIL --> LOOP

    %% Post-regeneration
    PASS --> SAVE[Save outputs:<br/>• output/image_variants/{stem}_{ratio}.png<br/>• output/manifests/{stem}_{ratio}.json]
    SAVE --> NEXT{More ratios?}
    NEXT -->|Yes| LOOP
    NEXT -->|16_9 → 1_1 → 9_16 → 4_5 → done| OUT

    OUT[OUTPUT<br/>• 4 variant PNGs (one per aspect ratio)<br/>• 4 manifest JSONs<br/>• validation JSONs]
```

## Stages Summary

| Stage | Module | Key Function | Output |
|-------|--------|-------------|--------|
| 1. Perception | `src/image_crop.py` | `detect_subjects()` | `tuple[list[Subject], str]` (subjects + model name) |
| 2. AI Planning | `src/regeneration.py` | `plan_crop_with_ai()` | Crop params dict (coverage, margin, center_x, center_y) |
| 3. Crop Planning | `src/image_crop.py` | `plan_crop()` | `CropPlan` |
| 4. Rendering | `src/image_crop.py` | `render_crop()` | `np.ndarray` |
| 5. Validation | `src/validator.py` | `validate_asset()` | `ValidationResult` |
| 6. AI Evaluation | `src/regeneration.py` | `evaluate_crop_with_ai()` | `AIEvaluationResult` |
| 7. Refinement Loop | `src/regeneration.py` | `regenerate_crop()` | `RegenerationResult` |

## Entry Point

`src/pipeline.py::process_image_to_variants()` — loads image once, detects subjects once (shared across all 4 variants), then calls `regenerate_crop()` per ratio. Supports `progress_callback` for SSE streaming via the `POST /process` backend endpoint.

`src/pipeline.py::process_single_variant()` — processes a single aspect ratio independently.

## Key Design Notes

- **Perception shared**: Subjects are detected once and reused across all 4 ratios.
- **Regeneration loop**: `regenerate_crop()` encapsulates stages 2–6 in a retry loop (up to `max_retries`). Feedback from validation + AI evaluation is used to adjust crop params on the next iteration — no new Gemini plan call, just a re-plan with adjusted parameters.
- **Description cache**: `_description_cache` in `src/regeneration.py` caches Gemini's image description across ratios to avoid redundant API calls.
- **Spec-driven min/max**: `plan_crop` uses min/max dimensions from `spec/platform_spec.yaml` (via `_spec_min_dimensions` / `_spec_max_dimensions`) to enforce bounds.
- **AI degrades gracefully**: `gemini_available()` checks `GEMINI_API_KEY` env var; if absent, the pipeline runs deterministic-only with `ai_status="unavailable"`.
