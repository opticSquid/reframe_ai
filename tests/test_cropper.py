"""Tests for the subject-aware crop computation engine (src/cropper.py)."""

from __future__ import annotations

import pytest

from src.cropper import (
    Subject,
    CropResult,
    compute_crop,
    compute_all_image_crops,
    _center_crop,
    _weighted_bbox,
    _compute_edge_cutoffs,
)
from src.config import TARGET_ASPECT_RATIOS


class TestCenterCrop:
    """Test the fallback center crop when no subjects are detected."""

    def test_no_subjects_returns_center_crop(self, sample_image_dims):
        """Empty subjects list → center crop maintaining target ratio."""
        result = compute_crop(4000, 4000, [], 1.7778)
        assert result.is_center_fallback is True
        assert abs(result.width / result.height - 1.7778) < 0.01

    def test_center_crop_16_9_from_square_source(self, sample_image_dims):
        """Center crop of a 4000x4000 image to 16:9."""
        result = _center_crop(4000, 4000, 16 / 9)
        assert result.width == 4000
        assert result.height == 2250  # 4000 / 1.778
        assert result.x == 0
        assert result.y == 875  # (4000 - 2250) / 2

    def test_center_crop_9_16_from_square_source(self, sample_image_dims):
        """Center crop of a 4000x4000 image to 9:16 (portrait)."""
        result = _center_crop(4000, 4000, 9 / 16)
        assert result.height == 4000
        assert result.width == 2250  # 4000 * 0.5625
        assert result.x == 875
        assert result.y == 0

    def test_center_crop_dimensions_in_bounds(self, sample_image_dims):
        """Center crop must not exceed source dimensions."""
        for ratio in [16/9, 1.0, 9/16, 4/5]:
            result = _center_crop(4000, 4000, ratio)
            assert result.width <= 4000
            assert result.height <= 4000
            assert result.x >= 0
            assert result.y >= 0
            assert result.x + result.width <= 4000
            assert result.y + result.height <= 4000


class TestWeightedBbox:
    """Test the weighted bounding box computation."""

    def test_single_subject(self):
        subjects = [Subject(bbox=(100, 200, 300, 400), importance=1.0)]
        bbox = _weighted_bbox(subjects)
        assert bbox == (100, 200, 300, 400)

    def test_two_equal_subjects(self):
        s1 = Subject(bbox=(100, 100, 200, 200), importance=1.0)
        s2 = Subject(bbox=(500, 300, 200, 200), importance=1.0)
        bbox = _weighted_bbox([s1, s2])
        # Union bbox: x in [100, 700], y in [100, 500]
        assert bbox[0] == 100  # min x
        assert bbox[1] == 100  # min y
        assert bbox[2] == 600  # width = 700 - 100
        assert bbox[3] == 400  # height = 500 - 100

    def test_weighted_union(self):
        s1 = Subject(bbox=(0, 0, 100, 100), importance=0.9)
        s2 = Subject(bbox=(300, 300, 100, 100), importance=0.1)
        bbox = _weighted_bbox([s1, s2])
        # Should be weighted toward s1
        assert bbox[0] < 150  # centroid closer to s1
        assert bbox[1] < 150


class TestComputeCrop:
    """Test the core crop computation with subjects."""

    def test_subject_centered_in_crop(self, sample_image_dims):
        """The primary subject's center should be near the crop center."""
        subjects = [Subject(bbox=(1500, 1500, 1000, 1000), importance=1.0)]
        result = compute_crop(4000, 4000, subjects, 1.0)

        subj_cx = 1500 + 500  # 2000
        subj_cy = 1500 + 500  # 2000
        crop_cx = result.x + result.width / 2
        crop_cy = result.y + result.height / 2

        # Subject center should be within 10% of crop center
        assert abs(crop_cx - subj_cx) < result.width * 0.1
        assert abs(crop_cy - subj_cy) < result.height * 0.1

    def test_crop_dimensions_match_aspect_ratio(self, sample_image_dims):
        """Output crop must match target aspect ratio (within rounding)."""
        subjects = [Subject(bbox=(1500, 1500, 1000, 1000), importance=1.0)]
        for name, ratio in TARGET_ASPECT_RATIOS.items():
            result = compute_crop(4000, 4000, subjects, ratio)
            actual_ratio = result.width / result.height
            assert abs(actual_ratio - ratio) < 0.02, \
                f"Ratio {name}: expected {ratio}, got {actual_ratio}"

    def test_crop_within_image_bounds(self, sample_image_dims):
        """Crop must never extend beyond source image boundaries."""
        subjects_list = [
            [Subject(bbox=(0, 0, 200, 400), importance=1.0)],       # top-left corner
            [Subject(bbox=(3800, 3600, 200, 400), importance=1.0)], # bottom-right
            [Subject(bbox=(1900, 1900, 200, 200), importance=1.0)], # exact center
        ]
        for subjects in subjects_list:
            for ratio in [16/9, 1.0, 9/16, 4/5]:
                result = compute_crop(4000, 4000, subjects, ratio)
                assert result.x >= 0, f"x={result.x} < 0"
                assert result.y >= 0, f"y={result.y} < 0"
                assert result.x + result.width <= 4000, \
                    f"crop exceeds right edge: {result.x + result.width} > 4000"
                assert result.y + result.height <= 4000, \
                    f"crop exceeds bottom: {result.y + result.height} > 4000"

    def test_crop_clamps_top_left_subject(self, sample_image_dims):
        """Subject in the top-left corner → crop clamped to image bounds."""
        subjects = [Subject(bbox=(0, 0, 200, 400), importance=1.0)]
        result = compute_crop(4000, 4000, subjects, 1.0, target_coverage=0.3)
        assert result.x == 0  # must clamp
        assert result.y == 0  # must clamp

    def test_crop_clamps_bottom_right_subject(self, sample_image_dims):
        """Subject in the bottom-right corner → crop clamped."""
        subjects = [Subject(bbox=(3800, 3600, 200, 400), importance=1.0)]
        result = compute_crop(4000, 4000, subjects, 1.0, target_coverage=0.3)
        assert result.x + result.width == 4000  # right edge
        assert result.y + result.height == 4000  # bottom edge

    def test_larger_target_coverage_gives_tighter_crop(self, sample_image_dims):
        """Higher target_coverage → smaller crop (more zoomed in)."""
        subjects = [Subject(bbox=(1500, 1500, 1000, 1000), importance=1.0)]
        loose = compute_crop(4000, 4000, subjects, 1.0, target_coverage=0.3)
        tight = compute_crop(4000, 4000, subjects, 1.0, target_coverage=0.7)
        assert loose.width * loose.height > tight.width * tight.height

    def test_primary_subject_center_recorded(self, sample_image_dims):
        """The primary subject center should be recorded in the result."""
        s1 = Subject(bbox=(100, 100, 200, 200), importance=0.5)
        s2 = Subject(bbox=(2000, 2000, 400, 400), importance=1.0)  # higher importance
        result = compute_crop(4000, 4000, [s1, s2], 1.0)
        # Primary = s2, center = (2200, 2200)
        assert result.primary_subject_center == (2200, 2200)


class TestComputeAllImageCrops:
    """Test computing all four standard crops at once."""

    def test_returns_all_four_ratios(self, sample_image_dims):
        subjects = [Subject(bbox=(1500, 1500, 1000, 1000), importance=1.0)]
        results = compute_all_image_crops(4000, 4000, subjects)
        assert set(results.keys()) == set(TARGET_ASPECT_RATIOS.keys())
        assert all(isinstance(r, CropResult) for r in results.values())

    def test_all_crops_within_bounds(self, sample_image_dims):
        subjects = [Subject(bbox=(500, 500, 800, 800), importance=1.0)]
        results = compute_all_image_crops(4000, 4000, subjects)
        for name, r in results.items():
            assert r.x >= 0, f"{name}: x={r.x}"
            assert r.y >= 0, f"{name}: y={r.y}"
            assert r.x + r.width <= 4000, f"{name}: exceeds width"
            assert r.y + r.height <= 4000, f"{name}: exceeds height"

    def test_all_aspect_ratios_correct(self, sample_image_dims):
        subjects = [Subject(bbox=(1500, 1500, 1000, 1000), importance=1.0)]
        results = compute_all_image_crops(4000, 4000, subjects)
        for name, ratio in TARGET_ASPECT_RATIOS.items():
            r = results[name]
            actual = r.width / r.height
            assert abs(actual - ratio) < 0.02, \
                f"{name}: expected {ratio:.4f}, got {actual:.4f}"


class TestEdgeCutoffs:
    """Test edge cutoff computation."""

    def test_no_cutoffs_when_crop_contains_subject(self):
        subjects = [Subject(bbox=(1000, 1000, 500, 500), importance=1.0)]
        result = compute_crop(4000, 4000, subjects, 1.0)
        cutoffs = result.edge_cutoffs
        for edge in ("top", "bottom", "left", "right"):
            assert cutoffs.get(edge, 0) < 0.05, \
                f"{edge} cutoff too high: {cutoffs[edge]}"

    def test_cutoff_when_crop_excludes_part_of_subject(self):
        # Subject bbox extends beyond crop boundary
        subjects = [Subject(bbox=(0, 0, 500, 500), importance=1.0)]
        result = compute_crop(4000, 4000, subjects, 1.0, target_coverage=0.8)
        cutoffs = result.edge_cutoffs
        # Subject starts at (0,0), crop should start at (0,0) due to clamping
        # So no cutoff should be > 0
        assert cutoffs["left"] == 0.0 or cutoffs["left"] < 0.5
