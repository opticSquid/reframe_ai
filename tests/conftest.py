"""Shared pytest fixtures for ReframeAI tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_image_dims() -> tuple[int, int]:
    """4000x4000 — matches the organizer's sample image."""
    return (4000, 4000)


@pytest.fixture
def sample_video_dims() -> tuple[int, int]:
    """1920x1080 — matches the organizer's sample video."""
    return (1920, 1080)


@pytest.fixture
def two_subjects_left_right(sample_image_dims):
    """Two subjects side by side in the lower-center (like the sample image)."""
    from src.cropper import Subject

    w, h = sample_image_dims
    # Subject 1: left woman (like the sample image)
    subj1 = Subject(
        bbox=(800, 2200, 400, 1400),
        importance=0.9,
        id="person_1",
        category="person",
    )
    # Subject 2: right woman
    subj2 = Subject(
        bbox=(2600, 2200, 500, 1500),
        importance=0.95,
        id="person_2",
        category="person",
    )
    return [subj1, subj2]


@pytest.fixture
def single_center_subject(sample_image_dims):
    """One subject centered in the frame."""
    from src.cropper import Subject

    w, h = sample_image_dims
    s = Subject(
        bbox=(1300, 1300, 1400, 1400),
        importance=1.0,
        id="primary",
        category="person",
    )
    return [s]


@pytest.fixture
def subject_near_edge(sample_image_dims):
    """Subject in the top-left corner (edge case for cutoffs)."""
    from src.cropper import Subject

    w, h = sample_image_dims
    s = Subject(
        bbox=(50, 50, 300, 600),
        importance=1.0,
        id="edge_subject",
        category="person",
    )
    return [s]


@pytest.fixture
def empty_manifest():
    """A basic manifest for an image variant (16:9)."""
    return {
        "asset_type": "image_variant",
        "asset_name": "test_image_16_9",
        "asset_variant": "16_9",
        "input_path": "data/test_image.png",
        "input_hash": "abc123",
        "input_dimensions": {"width": 4000, "height": 4000},
        "output_path": "output/image_variants/test_image_16_9.png",
        "output_dimensions": {"width": 2880, "height": 1620},
        "crop_params": {
            "x": 560,
            "y": 1190,
            "w": 2880,
            "h": 1620,
            "edge_cutoffs": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
        },
        "processing_time_sec": 1.2,
        "timestamp": "2026-09-26T12:00:00Z",
        "validation_passed": None,
        "validation_warnings": [],
    }


@pytest.fixture
def valid_manifest_path(tmp_path, empty_manifest):
    """Write a valid manifest to a temp file."""
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        import json
        json.dump(empty_manifest, f)
    return manifest_path
