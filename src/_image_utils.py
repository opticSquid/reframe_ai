"""Helper functions for image conversion between numpy and PNG bytes.

Used internally by the regeneration workflow to prepare images for
Gemini API transmission.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _np_to_png_bytes(arr: np.ndarray, max_dim: int = 1024) -> bytes:
    """Convert an ndarray (H×W×3 RGB) to PNG bytes, resized to max_dim."""
    img = Image.fromarray(arr)
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
