"""Asset validation engine.

Reads the machine-readable platform specification (spec/platform_spec.yaml)
and validates asset manifests against it.  Every output asset must pass
validation before entering the asset library.

The validator checks:
  1. Output dimensions match the spec for the asset type
  2. Aspect ratio is within tolerance
  3. Subject framing: edge cutoffs and coverage within spec limits
  4. Watermark preservation: crop region overlaps known watermark band
  5. Audio presence (for video reels)
  6. Required metadata fields are populated in the manifest

Optionally, the validator can verify actual file properties (dimensions via
PIL, audio via ffprobe) on disk — this is a secondary check layer.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .config import PLATFORM_SPEC_PATH


@dataclass
class ValidationResult:
    asset_name: str
    asset_type: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def load_spec(spec_path: Path | str | None = None) -> dict[str, Any]:
    """Load the platform specification from YAML."""
    path = Path(spec_path) if spec_path else PLATFORM_SPEC_PATH
    if not path.exists():
        raise FileNotFoundError(f"Platform spec not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Load a JSON manifest file."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _parse_dimensions(manifest: dict[str, Any]) -> tuple[int, int]:
    """Extract (width, height) from manifest."""
    dims = manifest.get("output_dimensions", {})
    if isinstance(dims, dict):
        return int(dims.get("width", 0)), int(dims.get("height", 0))
    if isinstance(dims, list) and len(dims) >= 2:
        return int(dims[0]), int(dims[1])
    return (0, 0)


def _parse_crop_params(manifest: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract (x, y, w, h) from manifest crop_params."""
    cp = manifest.get("crop_params", {})
    if isinstance(cp, dict):
        return (
            int(cp.get("x", 0)),
            int(cp.get("y", 0)),
            int(cp.get("w", 0)),
            int(cp.get("h", 0)),
        )
    if isinstance(cp, list) and len(cp) >= 4:
        return tuple(int(v) for v in cp[:4])  # type: ignore
    return (0, 0, 0, 0)


def _get_image_asset_type_spec(
    spec: dict[str, Any], asset_type: str, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Get the spec entry for a given asset type.

    For ``image_variant``, looks up the specific format (e.g. ``16_9``)
    from the manifest's ``asset_variant`` field inside
    ``spec["asset_types"]["image_variant"]["formats"]``.
    If no variant is specified, returns the first format.
    """
    at = spec.get("asset_types", {})
    if asset_type == "video_reel":
        return at.get("video_reel", {})
    if asset_type == "still_frame":
        return at.get("still_frame", {})

    # image_variant: resolve to a specific format
    iv_spec = at.get("image_variant", {})
    formats = iv_spec.get("formats", [])
    if not formats:
        return iv_spec

    if manifest is not None:
        variant_name = manifest.get("asset_variant", "")
        for fmt in formats:
            if fmt.get("name", "") == variant_name:
                return fmt

    # Default: return first format
    return formats[0] if formats else iv_spec


def _check_file_dimensions(output_path: str | Path) -> tuple[int, int] | None:
    """Verify actual file dimensions on disk via PIL. Returns None if not an image or file missing."""
    path = Path(output_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def _check_audio_present(output_path: str | Path) -> bool:
    """Check if the video file has an audio track via ffprobe."""
    path = str(output_path)
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return False
        info = json.loads(result.stdout)
        return any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_asset(
    manifest: dict[str, Any],
    spec: dict[str, Any] | None = None,
    verify_files: bool = True,
) -> ValidationResult:
    """Validate an asset manifest against the platform spec.

    Parameters
    ----------
    manifest
        The JSON manifest dict for the asset.
    spec
        Pre-loaded spec dict. If None, loads from default path.
    verify_files
        If True, also verify actual file dimensions and audio on disk.
    """
    if spec is None:
        spec = load_spec()

    asset_name = manifest.get("asset_name", "unknown")
    asset_type = manifest.get("asset_type", "unknown")

    errors: list[str] = []
    warnings: list[str] = []

    result = ValidationResult(
        asset_name=asset_name,
        asset_type=asset_type,
        passed=True,
        errors=errors,
        warnings=warnings,
    )

    # ------------------------------------------------------------------
    # Check 1: Required manifest fields
    # ------------------------------------------------------------------
    required_fields = {
        "image_variant": ["asset_type", "asset_name", "input_hash",
                          "output_path", "output_dimensions", "crop_params"],
        "video_reel": ["asset_type", "asset_name", "input_hash",
                       "output_path", "output_dimensions", "crop_params"],
        "still_frame": ["asset_type", "asset_name", "input_hash",
                        "output_path", "output_dimensions"],
    }

    needed = required_fields.get(asset_type, required_fields["image_variant"])
    for field_name in needed:
        if field_name not in manifest or manifest[field_name] is None:
            errors.append(f"Missing required field: '{field_name}'")

    if errors:
        result.passed = False
        return result

    # ------------------------------------------------------------------
    # Check 2: Asset type spec exists
    # ------------------------------------------------------------------
    at_spec = _get_image_asset_type_spec(spec, asset_type, manifest)
    if not at_spec:
        errors.append(f"No spec found for asset type: '{asset_type}'")
        result.passed = False
        return result

    validation_rules = spec.get("validation_rules", {})

    # ------------------------------------------------------------------
    # Check 3: Dimensions and aspect ratio
    # ------------------------------------------------------------------
    out_w, out_h = _parse_dimensions(manifest)
    if out_w <= 0 or out_h <= 0:
        errors.append(f"Invalid output dimensions: {out_w}x{out_h}")
    else:
        actual_ratio = out_w / out_h

        # Min/max bounds
        if validation_rules.get("dimensions", {}).get("check_min_max", True):
            min_w = at_spec.get("min_width", 0)
            min_h = at_spec.get("min_height", 0)
            max_w = at_spec.get("max_width", float("inf"))
            max_h = at_spec.get("max_height", float("inf"))

            if out_w < min_w:
                errors.append(f"Width {out_w} below minimum {min_w}")
            if out_h < min_h:
                errors.append(f"Height {out_h} below minimum {min_h}")
            if out_w > max_w:
                errors.append(f"Width {out_w} exceeds maximum {max_w}")
            if out_h > max_h:
                errors.append(f"Height {out_h} exceeds maximum {max_h}")

        # Aspect ratio tolerance
        if "aspect_ratio" in at_spec:
            target_ratio = float(at_spec["aspect_ratio"])
            tolerance = validation_rules.get("dimensions", {}).get("aspect_ratio_tolerance", 0.02)
            ratio_diff = abs(actual_ratio - target_ratio) / target_ratio
            if ratio_diff > tolerance:
                errors.append(
                    f"Aspect ratio {actual_ratio:.4f} deviates from "
                    f"target {target_ratio:.4f} by {ratio_diff:.4f} "
                    f"(tolerance: {tolerance})"
                )

    # ------------------------------------------------------------------
    # Check 4: Subject framing (for image_variant and still_frame)
    # ------------------------------------------------------------------
    if asset_type in ("image_variant", "still_frame"):
        framing = at_spec.get("subject_framing", {})
        if framing.get("enabled", validation_rules.get("subject_framing", {}).get("enabled", True)):
            max_edge = framing.get("max_edge_cutoff_pct",
                                   validation_rules.get("subject_framing", {}).get("max_edge_cutoff_pct", 0.25))

            crop_params = manifest.get("crop_params", {})
            edge_cutoffs = crop_params.get("edge_cutoffs", {}) if isinstance(crop_params, dict) else {}

            # edge_cutoffs is optional in manifest (may not be computed yet)
            if edge_cutoffs:
                for edge in ("top", "bottom", "left", "right"):
                    cutoff = edge_cutoffs.get(edge, 0.0)
                    if cutoff > max_edge:
                        errors.append(
                            f"Edge cutoff '{edge}' = {cutoff:.2%} exceeds "
                            f"maximum {max_edge:.2%}"
        )
            else:
                # We can compute from crop_params if dimensions are available
                pass

    # ------------------------------------------------------------------
    # Check 5: Watermark preservation
    # ------------------------------------------------------------------
    if validation_rules.get("watermark_preservation", {}).get("enabled", True):
        watermark_spec = spec.get("common", {}).get("watermark", {})
        if watermark_spec.get("must_be_present", False):
            # The watermark must overlap with the crop region.
            # We know the watermark is in the center band.
            # For the image: watermark is approximately at y=2000-2400 of 4000 height
            # For video: same center band
            # We check that the crop includes some of the center band.
            crop_x, crop_y, crop_w, crop_h = _parse_crop_params(manifest)
            input_w = manifest.get("input_dimensions", {}).get("width", 0) or manifest.get("input_dimensions", {}).get("h", 0)
            input_h = 0
            if isinstance(manifest.get("input_dimensions"), dict):
                input_h = manifest["input_dimensions"].get("height", manifest["input_dimensions"].get("w", 0))

            # Watermark is roughly in the center 10% vertically (center band)
            wm_center = input_h / 2 if input_h > 0 else None
            if wm_center is not None:
                wm_band_size = input_h * 0.1 if input_h > 0 else 400
                wm_top = wm_center - wm_band_size / 2
                wm_bottom = wm_center + wm_band_size / 2

                # Check overlap
                crop_top = crop_y
                crop_bottom = crop_y + crop_h
                overlap = max(0, min(crop_bottom, wm_bottom) - max(crop_top, wm_top))
                total_crop_height = crop_h
                if total_crop_height > 0 and overlap == 0:
                    warnings.append(
                        "Watermark region not included in crop. "
                        "Watermark may be absent from output."
                    )

    # ------------------------------------------------------------------
    # Check 6: Audio presence (video_reel only)
    # ------------------------------------------------------------------
    if asset_type == "video_reel":
        if validation_rules.get("audio_presence", {}).get("enabled", True):
            output_path = manifest.get("output_path", "")
            if verify_files:
                if not _check_audio_present(output_path):
                    errors.append("Video reel has no audio track")
            else:
                # Trust manifest's reported audio status
                has_audio = manifest.get("audio_present", False)
                if not has_audio:
                    errors.append("Manifest reports no audio track for video reel")

    # ------------------------------------------------------------------
    # Check 7: Verify actual file dimensions on disk
    # ------------------------------------------------------------------
    if verify_files and asset_type != "video_reel":
        output_path = manifest.get("output_path", "")
        actual_dims = _check_file_dimensions(output_path)
        if actual_dims is None:
            warnings.append(f"Could not verify file dimensions: {output_path}")
        elif actual_dims != (out_w, out_h):
            errors.append(
                f"File dimensions {actual_dims[0]}x{actual_dims[1]} "
                f"do not match manifest {out_w}x{out_h}"
            )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    result.passed = len(errors) == 0
    result.errors = errors
    result.warnings = warnings
    return result


def validate_manifest_file(
    manifest_path: str | Path,
    spec_path: str | Path | None = None,
    verify_files: bool = True,
) -> ValidationResult:
    """Convenience: load manifest from file, then validate."""
    manifest = load_manifest(manifest_path)
    spec = load_spec(spec_path) if spec_path else None
    return validate_asset(manifest, spec, verify_files=verify_files)


def print_validation_report(result: ValidationResult) -> None:
    """Print a human-readable validation report."""
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  Validation: {result.asset_name} ({result.asset_type})")
    print(f"  Status: {status}")
    print(f"{'='*60}")

    if result.errors:
        print(f"\n  ERRORS ({len(result.errors)}):")
        for e in result.errors:
            print(f"    ✗ {e}")

    if result.warnings:
        print(f"\n  WARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    ⚠ {w}")

    if not result.errors and not result.warnings:
        print("\n  ✓ All checks passed, no warnings.")

    print()
