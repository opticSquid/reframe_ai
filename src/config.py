"""Central configuration: paths, constants, and API key management.

Gemini API key is read from environment variable GEMINI_API_KEY.
If not set, Gemini-dependent features degrade gracefully with a user-friendly error.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPEC_DIR = PROJECT_ROOT / "spec"

IMAGE_VARIANTS_DIR = OUTPUT_DIR / "image_variants"
VIDEO_REELS_DIR = OUTPUT_DIR / "video_reels"
STILLS_DIR = OUTPUT_DIR / "stills"
MANIFESTS_DIR = OUTPUT_DIR / "manifests"

# MediaPipe models
MODELS_DIR = PROJECT_ROOT / "models"
FACE_DETECTION_MODEL = MODELS_DIR / "blaze_face_full_range_sparse_float16.tflite"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker_float16.task"
PERSON_DETECTION_MODEL = MODELS_DIR / "efficientdet_lite0_float16.tflite"
POSE_LANDMARKER_MODEL = MODELS_DIR / "pose_landmarker_full_float16.task"

# ---------------------------------------------------------------------------
# Platform spec
# ---------------------------------------------------------------------------
PLATFORM_SPEC_PATH = SPEC_DIR / "platform_spec.yaml"

# ---------------------------------------------------------------------------
# Target aspect ratios for image variants
# ---------------------------------------------------------------------------
TARGET_ASPECT_RATIOS = {
    "16_9": 16 / 9,    # 1.778
    "1_1": 1.0,        # 1.0
    "9_16": 9 / 16,    # 0.5625
    "4_5": 4 / 5,      # 0.8
}

# Video reel target
REEL_ASPECT_RATIO = 9 / 16   # 0.5625 (vertical)

# Still extraction target
STILL_ASPECT_RATIO = 1.0     # 1:1 square

# ---------------------------------------------------------------------------
# AI / Model defaults
# ---------------------------------------------------------------------------
# Keyframe interval for video processing (every N frames)
VIDEO_KEYFRAME_INTERVAL = 25  # 25fps → ~1 keyframe per second

# Frame sampling rate for perception (frames per second to analyze)
VIDEO_SAMPLE_FPS = 4.0          # detection/tracking sampling
VIDEO_LANDMARK_FPS = 2.0       # face landmark sampling (MAR computation)

# Frame resize for AI model inference (keeps CPU inference fast)
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 480

# Minimum confidence for detections to be considered
MIN_FACE_CONFIDENCE = 0.5
MIN_POSE_CONFIDENCE = 0.5
MIN_OBJECT_CONFIDENCE = 0.4

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3.8-flash"


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None if not set."""
    return os.environ.get(GEMINI_API_KEY_ENV)


def gemini_available() -> bool:
    """Check whether the Gemini API key is configured."""
    return get_gemini_api_key() is not None


def require_gemini_api_key() -> str:
    """Return the key or raise a user-friendly error."""
    key = get_gemini_api_key()
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Set the GEMINI_API_KEY environment "
            "variable to enable AI-semantic features (subject importance "
            "scoring, speaker identification, output critique). "
            "Without it, only deterministic fallback behavior is available.\n"
            "Example: export GEMINI_API_KEY=your-key-here"
        )
    return key
