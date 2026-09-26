"""Tests for the image reformatting module (src/image_crop.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.image_crop import (
    CropPlan,
    compute_target_dimensions,
    detect_subjects,
    plan_crop,
    render_crop,
    process_image,
    process_all_ratios,
)
from src.cropper import CropResult, Subject, compute_crop
from src.config import TARGET_ASPECT_RATIOS


# ---------------------------------------------------------------------------
# Aspect ratio calculation tests
# ---------------------------------------------------------------------------
class TestAspectRatioCalculation:
    """Test compute_target_dimensions for all supported ratios."""

    def test_landscape_16_9(self):
        """16:9 on 4000x4000 should yield 4000x2250."""
        w, h = compute_target_dimensions(4000, 4000, 16 / 9)
        assert w == 4000
        # 4000 / (16/9) = 4000 * 9/16 = 2250
        assert h == 2250

    def test_square_1_1(self):
        """1:1 on 4000x4000 should yield 4000x4000."""
        w, h = compute_target_dimensions(4000, 4000, 1.0)
        assert w == 4000
        assert h == 4000

    def test_portrait_9_16(self):
        """9:16 on 4000x4000 should yield 2250x4000."""
        w, h = compute_target_dimensions(4000, 4000, 9 / 16)
        # 4000 * (9/16) = 2250
        assert w == 2250
        assert h == 4000

    def test_portrait_4_5(self):
        """4:5 on 4000x4000 should yield 3200x4000."""
        w, h = compute_target_dimensions(4000, 4000, 4 / 5)
        # 4000 * (4/5) = 3200
        assert w == 3200
        assert h == 4000

    def test_landscape_on_wide_image(self):
        """16:9 on 1920x1080 should yield 1920x1080 (fits exactly)."""
        w, h = compute_target_dimensions(1920, 1080, 16 / 9)
        assert w == 1920
        assert h == 1080

    def test_portrait_on_wide_image(self):
        """9:16 on 1920x1080 should yield 607x1080 (height-constrained)."""
        w, h = compute_target_dimensions(1920, 1080, 9 / 16)
        assert h == 1080
        # 1080 * (9/16) = 607.5 → rounds to 607 or 608
        assert w in (607, 608)

    def test_all_standard_ratios(self):
        """All four standard ratios should produce valid dimensions."""
        for name, ratio in TARGET_ASPECT_RATIOS.items():
            w, h = compute_target_dimensions(4000, 4000, ratio)
            assert w > 0 and h > 0
            assert w <= 4000 and h <= 4000
            # Verify the ratio matches (within rounding)
            assert abs(w / h - ratio) < 0.01


# ---------------------------------------------------------------------------
# Crop bounds tests
# ---------------------------------------------------------------------------
class TestCropBounds:
    """Test that crop coordinates are valid and within source image."""

    def test_crop_within_source(self, sample_image_dims):
        """Crop must not exceed source image boundaries."""
        w, h = sample_image_dims  # 4000, 4000
        subjects = [
            Subject(bbox=(1000, 1000, 500, 500), importance=1.0, id="s1"),
        ]
        result = compute_crop(w, h, subjects, target_ratio=9 / 16, target_coverage=0.4)
        assert result.x >= 0
        assert result.y >= 0
        assert result.x + result.width <= w
        assert result.y + result.height <= h
        assert result.width > 0
        assert result.height > 0

    def test_crop_aspect_ratio_preserved(self, sample_image_dims):
        """Crop aspect ratio must match target ratio (within rounding)."""
        w, h = sample_image_dims  # 4000, 4000
        subjects = [
            Subject(bbox=(1000, 1000, 500, 500), importance=1.0, id="s1"),
        ]
        for name, ratio in TARGET_ASPECT_RATIOS.items():
            result = compute_crop(w, h, subjects, target_ratio=ratio, target_coverage=0.4)
            actual_ratio = result.width / result.height
            assert abs(actual_ratio - ratio) < 0.02, (
                f"Ratio mismatch for {name}: expected {ratio}, got {actual_ratio}"
            )

    def test_crop_center_fallback_within_bounds(self, sample_image_dims):
        """Center fallback crop must also stay within bounds."""
        w, h = sample_image_dims
        result = compute_crop(w, h, subjects=[], target_ratio=16 / 9)
        assert result.is_center_fallback is True
        assert result.x >= 0
        assert result.y >= 0
        assert result.x + result.width <= w
        assert result.y + result.height <= h

    def test_crop_near_edge_clamped(self, sample_image_dims, subject_near_edge):
        """Crop centered on an edge subject must not go negative."""
        w, h = sample_image_dims
        result = compute_crop(w, h, subject_near_edge, target_ratio=16 / 9, target_coverage=0.3)
        assert result.x >= 0
        assert result.y >= 0
        assert result.x + result.width <= w
        assert result.y + result.height <= h


# ---------------------------------------------------------------------------
# Target dimensions tests
# ---------------------------------------------------------------------------
class TestTargetDimensions:
    """Test that target dimensions are correct for various source images."""

    def test_4000_square(self):
        """4000x4000 square source — all ratios should be sub-square."""
        for ratio in [16 / 9, 1.0, 9 / 16, 4 / 5]:
            w, h = compute_target_dimensions(4000, 4000, ratio)
            assert w <= 4000 and h <= 4000

    def test_1920x1080_wide(self):
        """1920x1080 source — landscape crops fit, portrait crops width-constrained."""
        for ratio in [16 / 9, 1.0, 9 / 16, 4 / 5]:
            w, h = compute_target_dimensions(1920, 1080, ratio)
            assert w <= 1920 and h <= 1080

    def test_portrait_source(self):
        """1080x1920 source — portrait crops fit, landscape crops height-constrained."""
        for ratio in [16 / 9, 1.0, 9 / 16, 4 / 5]:
            w, h = compute_target_dimensions(1080, 1920, ratio)
            assert w <= 1080 and h <= 1920

    def test_dimension_ratios_are_exact(self):
        """After computing dimensions, w/h must equal target ratio ±1px."""
        for sx, sy in [(4000, 4000), (1920, 1080), (1080, 1920), (3000, 2000), (2000, 3000)]:
            for ratio in [16 / 9, 1.0, 9 / 16, 4 / 5]:
                w, h = compute_target_dimensions(sx, sy, ratio)
                if w > 1 and h > 1:
                    assert abs(w / h - ratio) < 0.01, (
                        f"Failed: {sx}x{sy} ratio={ratio} -> {w}x{h}, "
                        f"actual={w/h}"
                    )


# ---------------------------------------------------------------------------
# Render crop tests
# ---------------------------------------------------------------------------
class TestRenderCrop:
    """Test the render_crop function."""

    def test_render_correct_size(self, tmp_path):
        """Rendered crop should match the CropResult dimensions."""
        img = Image.new("RGB", (4000, 4000), color=(128, 100, 50))
        path = tmp_path / "test.png"
        img.save(path)

        crop = CropResult(x=100, y=200, width=576, height=1024)
        result = render_crop(path, crop)

        assert result.shape[0] == 1024  # height
        assert result.shape[1] == 576   # width
        assert result.dtype == np.uint8

    def test_render_clamps_out_of_bounds(self, tmp_path):
        """Render should safely clamp if crop exceeds image bounds."""
        img = Image.new("RGB", (1000, 1000), color=(255, 0, 0))
        path = tmp_path / "test.png"
        img.save(path)

        # Crop that extends past the right edge
        crop = CropResult(x=800, y=0, width=500, height=500)
        result = render_crop(path, crop)

        # Should only return the valid portion
        assert result.shape[1] == 200  # 1000 - 800 = 200

    def test_render_preserves_color(self, tmp_path):
        """Rendered crop should preserve pixel values."""
        img = Image.new("RGB", (100, 100), color=(50, 100, 150))
        path = tmp_path / "test.png"
        img.save(path)

        crop = CropResult(x=10, y=10, width=20, height=20)
        result = render_crop(path, crop)

        assert result.shape[0] == 20
        assert result.shape[1] == 20
        # All pixels should be the same color
        assert (result[:, :] == [50, 100, 150]).all()


# ---------------------------------------------------------------------------
# Plan crop integration tests
# ---------------------------------------------------------------------------
class TestPlanCrop:
    """Test the plan_crop function end-to-end."""

    def test_plan_returns_crop_plan(self, tmp_path):
        """plan_crop should return a CropPlan with all fields populated."""
        img = Image.new("RGB", (1000, 1000), color=(200, 200, 200))
        path = tmp_path / "test.png"
        img.save(path)

        plan = plan_crop(path, target_ratio=9 / 16)

        assert isinstance(plan, CropPlan)
        assert plan.image_width == 1000
        assert plan.image_height == 1000
        assert plan.target_ratio == 9 / 16
        assert isinstance(plan.crop, CropResult)
        assert plan.image_path == str(path)

    def test_plan_crop_result_is_valid(self, tmp_path):
        """The CropResult from plan_crop must have valid bounds."""
        img = Image.new("RGB", (2000, 2000), color=(100, 100, 100))
        path = tmp_path / "test.png"
        img.save(path)

        plan = plan_crop(path, target_ratio=16 / 9, target_coverage=0.4)
        result = plan.crop

        assert result.x >= 0
        assert result.y >= 0
        assert result.x + result.width <= 2000
        assert result.y + result.height <= 2000
        assert result.width > 0
        assert result.height > 0

    def test_plan_metadata_serializable(self, tmp_path):
        """CropPlan.to_metadata() must produce valid JSON."""
        img = Image.new("RGB", (1000, 1000), color=(50, 50, 50))
        path = tmp_path / "test.png"
        img.save(path)

        plan = plan_crop(path, target_ratio=1.0)
        metadata = plan.to_metadata()

        # Should be JSON-serializable
        json.dumps(metadata)
        assert "image_path" in metadata
        assert "crop" in metadata
        assert "subjects" in metadata


# ---------------------------------------------------------------------------
# Process image (end-to-end) tests
# ---------------------------------------------------------------------------
class TestProcessImage:
    """Test the convenience process_image function."""

    def test_process_returns_three_items(self, tmp_path):
        """process_image returns (crop_result, image_array, metadata)."""
        img = Image.new("RGB", (2000, 2000), color=(100, 200, 50))
        path = tmp_path / "test.png"
        img.save(path)

        crop_result, rendered, metadata = process_image(path, target_ratio=9 / 16)

        assert isinstance(crop_result, CropResult)
        assert isinstance(rendered, np.ndarray)
        assert isinstance(metadata, dict)
        assert "crop" in metadata
        assert "subjects_detected" in metadata

    def test_rendered_matches_crop_dimensions(self, tmp_path):
        """Rendered image dimensions must match the crop result."""
        img = Image.new("RGB", (2000, 2000), color=(100, 200, 50))
        path = tmp_path / "test.png"
        img.save(path)

        crop_result, rendered, _ = process_image(path, target_ratio=16 / 9, target_coverage=0.3)

        assert rendered.shape[0] == crop_result.height
        assert rendered.shape[1] == crop_result.width

    def test_process_all_ratios(self, tmp_path):
        """process_all_ratios should return plans for all standard ratios."""
        img = Image.new("RGB", (2000, 2000), color=(100, 200, 50))
        path = tmp_path / "test.png"
        img.save(path)

        plans = process_all_ratios(path)

        assert len(plans) == 4
        assert "16_9" in plans
        assert "1_1" in plans
        assert "9_16" in plans
        assert "4_5" in plans

        for name, plan in plans.items():
            assert isinstance(plan, CropPlan)
            assert plan.crop.width > 0
            assert plan.crop.height > 0
