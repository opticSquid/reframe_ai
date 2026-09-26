"""Tests for the validation engine (src/validator.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validator import (
    load_spec,
    load_manifest,
    validate_asset,
    validate_manifest_file,
    ValidationResult,
)
from src.config import PLATFORM_SPEC_PATH


class TestSpecLoading:
    """Test that the platform spec loads correctly."""

    def test_spec_loads_from_default_path(self):
        spec = load_spec()
        assert "common" in spec
        assert "asset_types" in spec
        assert "validation_rules" in spec

    def test_spec_has_all_asset_types(self):
        spec = load_spec()
        at = spec["asset_types"]
        assert "image_variant" in at
        assert "video_reel" in at
        assert "still_frame" in at

    def test_spec_has_watermark_config(self):
        spec = load_spec()
        assert spec["common"]["watermark"]["must_be_present"] is True

    def test_spec_has_image_variants(self):
        spec = load_spec()
        variants = spec["asset_types"]["image_variant"]["formats"]
        names = [f["name"] for f in variants]
        assert "16_9" in names
        assert "1_1" in names
        assert "9_16" in names
        assert "4_5" in names


class TestManifestLoading:
    """Test manifest JSON loading."""

    def test_load_valid_manifest(self, valid_manifest_path):
        manifest = load_manifest(valid_manifest_path)
        assert manifest["asset_type"] == "image_variant"
        assert manifest["asset_name"] == "test_image_16_9"

    def test_load_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.json")


class TestValidateImageVariant:
    """Test validation of image variant assets."""

    def test_valid_16_9_manifest_passes(self, empty_manifest, tmp_path):
        # Create a dummy output file with correct dimensions
        from PIL import Image
        output_path = tmp_path / "test_16_9.png"
        Image.new("RGB", (2880, 1620), (128, 128, 128)).save(output_path)

        manifest = empty_manifest.copy()
        manifest["output_path"] = str(output_path)
        manifest["crop_params"]["edge_cutoffs"] = {"top": 0.0, "bottom": 0.0,
                                                    "left": 0.0, "right": 0.0}

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=True)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_wrong_aspect_ratio_fails(self, empty_manifest):
        manifest = empty_manifest.copy()
        manifest["output_dimensions"] = {"width": 1000, "height": 1000}  # 1:1, not 16:9

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=False)
        assert result.passed is False
        assert any("aspect ratio" in e.lower() for e in result.errors)

    def test_dimensions_below_minimum_fails(self, empty_manifest):
        manifest = empty_manifest.copy()
        manifest["output_dimensions"] = {"width": 100, "height": 56}  # too small

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=False)
        assert result.passed is False
        assert any("below minimum" in e for e in result.errors)

    def test_missing_required_field_fails(self, empty_manifest):
        manifest = empty_manifest.copy()
        del manifest["output_path"]

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=False)
        assert result.passed is False
        assert any("output_path" in e for e in result.errors)

    def test_file_dimensions_mismatch_fails(self, empty_manifest, tmp_path):
        from PIL import Image
        output_path = tmp_path / "test_16_9.png"
        Image.new("RGB", (500, 300), (128, 128, 128)).save(output_path)

        manifest = empty_manifest.copy()
        manifest["output_path"] = str(output_path)

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=True)
        assert result.passed is False
        assert any("do not match" in e for e in result.errors)


class TestValidateStillFrame:
    """Test validation of still frame assets."""

    def test_valid_still_manifest(self, tmp_path):
        from PIL import Image
        from src.validator import validate_asset

        output_path = tmp_path / "still.png"
        Image.new("RGB", (1080, 1080), (128, 128, 128)).save(output_path)

        manifest = {
            "asset_type": "still_frame",
            "asset_name": "test_still",
            "input_path": "data/input_video.mp4",
            "input_dimensions": {"width": 1920, "height": 1080},
            "input_hash": "def456",
            "output_path": str(output_path),
            "output_dimensions": {"width": 1080, "height": 1080},
            "crop_params": {
                "x": 420, "y": 0, "w": 1080, "h": 1080,
                "edge_cutoffs": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
            },
            "source_timestamp": 125.5,
            "processing_time_sec": 2.1,
            "timestamp": "2026-09-26T12:00:00Z",
        }

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=True)
        assert result.passed is True


class TestValidateVideoReel:
    """Test validation of video reel assets."""

    def test_valid_video_manifest_no_audio_flag(self):
        from src.validator import validate_asset

        manifest = {
            "asset_type": "video_reel",
            "asset_name": "test_reel",
            "input_path": "data/input_video.mp4",
            "input_dimensions": {"width": 1920, "height": 1080},
            "input_hash": "vid123",
            "output_path": "output/video_reels/test_reel.mp4",
            "output_dimensions": {"width": 720, "height": 1280},
            "crop_params": {
                "x": 0, "y": 0, "w": 720, "h": 1280,
                "edge_cutoffs": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
                "crop_trajectory": [],
                "speaker_ids": ["speaker_1"],
                "scene_boundaries": [0, 50, 100, 190],
                "keyframe_crops": [],
            },
            "audio_present": True,
            "processing_time_sec": 60.0,
            "timestamp": "2026-09-26T12:00:00Z",
        }

        spec = load_spec()
        # Don't verify files (no actual file) — manifest says audio_present=True
        result = validate_asset(manifest, spec, verify_files=False)
        assert result.passed is True

    def test_video_wrong_aspect_ratio_fails(self):
        from src.validator import validate_asset

        manifest = {
            "asset_type": "video_reel",
            "asset_name": "test_reel_bad",
            "input_path": "data/input_video.mp4",
            "output_path": "output/video_reels/test_reel_bad.mp4",
            "output_dimensions": {"width": 1920, "height": 1080},  # 16:9, not 9:16
            "audio_present": True,
            "crop_params": {},
        }

        spec = load_spec()
        result = validate_asset(manifest, spec, verify_files=False)
        assert result.passed is False
