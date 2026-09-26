# Image Pipeline Flow

```
USER
  │
  ▼
┌──────────────────────────────────┐
│  Upload image  (PNG/JPG)         │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 1: PERCEPTION              │
│  MediaPipe BlazeFace detection    │
│  → subjects (bbox + importance)   │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 2: AI PLANNING (optional)   │
│  Gemini analyzes image semantically│
│  → crop params (coverage, margin,  │
│    center_x, center_y)            │
│  ┌─── No key? ───→ deterministic ───┐
│  │                                   │
└─┴─────────────────────────────────┘ │
                                       │
          ┌───────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 3: CROP PLANNING            │
│  plan_crop(): deterministic        │
│  subjects + ratio + AI params      │
│  → CropResult (x, y, w, h)         │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 4: RENDERING                │
│  render_crop(): PIL/numpy slice    │
│  → rendered variant image          │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 5: VALIDATION               │
│  validate_asset():                 │
│  • dimensions check                │
│  • aspect ratio (±2%)              │
│  • subject framing                 │
│  • watermark overlap               │
│  → ValidationResult (passed/errors)│
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  STAGE 6: AI EVALUATION (optional) │
│  Gemini critiquing rendered crop: │
│  • faces cropped out?              │
│  • framing natural?                │
│  → AIEvaluationResult              │
│  ┌─── No key? ───→ skip ───────────┐
│  │                                   │
└─┴─────────────────────────────────┘ │
                                      │
          ┌──────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│  DECISION                         │
│  ┌─ All checks PASS ──────────────┐│
└─┬──────────────────────────────────┘│
  │ PASS                              │
  ▼                                   │
┌────────────────────────┐            │
│  Save variant PNG      │            │
│  Save manifest JSON    │            │
└─────────┬──────────────┘            │
  │                                   │
  ▼                                   │
┌────────────────────────┐            │
│  Next ratio?           │◄────────────┘
│  (16_9 → 1_1 → 9_16 →   │
│   4_5 → done)            │
└─────────┬──────────────┘
  │ YES
  ▼
┌──────────────────────────────────┐
│  REPEAT Stages 3–6 for each ratio  │
└─────────┬────────────────────────┘
  │ NO
  ▼
┌──────────────────────────────────┐
│  OUTPUT                              │
│  • output/image_variants/<name>_*   │
│    .png (4 aspect ratios)            │
│  • output/manifests/<name>_<ratio>  │
│    .json                              │
│  • output/manifests/<name>_validation│
│    .json                              │
└──────────────────────────────────┘
```

## Stages Summary

| Stage | Module | Key Function | Output |
|-------|--------|-------------|--------|
| 1. Perception | `src/image_crop.py` | `detect_subjects()` | `list[Subject]` |
| 2. AI Planning | `src/regeneration.py` | `plan_crop_with_ai()` | Crop params dict |
| 3. Crop Planning | `src/image_crop.py` | `plan_crop()` | `CropPlan` |
| 4. Rendering | `src/image_crop.py` | `render_crop()` | `np.ndarray` |
| 5. Validation | `src/validator.py` | `validate_asset()` | `ValidationResult` |
| 6. AI Evaluation | `src/regeneration.py` | `evaluate_crop_with_ai()` | `AIEvaluationResult` |
| 7. Refinement Loop | `src/regeneration.py` | `regenerate_crop()` | `RegenerationResult` |

## Entry Point

`src/pipeline.py:process_image_to_variants()` — processes all 4 ratios in sequence, reuses subjects across variants.
