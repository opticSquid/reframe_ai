"""Video rendering: dynamic crop, FFmpeg encoding, still extraction, debug overlay.

This module takes a crop trajectory (per-frame crop boxes) and the original
video, and produces:
  - A vertical (9:16) re-rendered video with audio preserved
  - A representative still image (best frame)
  - A debug overlay video showing tracking + speaker + crop

Rendering uses a pipe-based approach: ffmpeg decodes → Python crops → ffmpeg encodes.
This avoids the OpenCV H.264 encoder issue on this system.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .video_speaker import CropTrajectoryPoint
from .video_perception import TrackedPerson, DetectedFace, _track_color


@dataclass
class RenderResult:
    """Result of rendering a vertical video."""

    reel_path: str
    still_path: str
    debug_path: str | None
    reel_metadata: dict[str, Any]
    validation: dict[str, Any]
    still_paths: list[str] = field(default_factory=list)


def render_vertical_video(
    video_path: str | Path,
    trajectory: list[CropTrajectoryPoint],
    tracks: list[TrackedPerson],
    output_path: str | Path,
    target_ratio: float = 9 / 16,
    source_width: int = 1920,
    source_height: int = 1080,
    fps: float = 25.0,
    total_frames: int | None = None,
    generate_debug: bool = True,
    debug_path: str | Path | None = None,
    stills_dir: str | Path | None = None,
) -> RenderResult:
    """Render a vertical 9:16 video from the source using a dynamic crop trajectory.

    Uses ffmpeg to decode source frames as raw RGB, applies per-frame crop
    via numpy slicing, and pipes to ffmpeg for H.264 encoding.  Audio from
    the original is muxed in a second pass.

    Also extracts a representative still and optionally a debug overlay video.
    """
    path = str(video_path)
    out_path = str(output_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Compute output dimensions from target ratio (use spec minimum for speed)
    if target_ratio >= 1.0:
        out_h = 800
        out_w = int(round(out_h * target_ratio))
    else:
        out_h = 1280
        out_w = int(round(out_h * target_ratio))

    if total_frames is None:
        total_frames = int(fps * (len(trajectory) / fps if trajectory else 1))
        # Estimate from video
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "default=noprint_wrappers=1:nokey=1",
             path],
            capture_output=True, text=True, timeout=30,
        )
        if result.stdout.strip().isdigit():
            total_frames = int(result.stdout.strip())
        else:
            total_frames = int(fps * source_height / source_width)  # fallback

    # Ensure trajectory covers all frames
    _ensure_trajectory(trajectory, total_frames, source_width, source_height, target_ratio, fps)

    # Build a frame-index → trajectory lookup
    traj_by_frame: dict[int, CropTrajectoryPoint] = {}
    for point in trajectory:
        fi = int(round(point.time * fps))
        traj_by_frame[fi] = point

    # ---- Decode → Crop → Encode pipeline (video only, no audio) ----
    # Step 1: ffmpeg decode raw RGB frames
    decode_cmd = [
        "ffmpeg", "-y", "-v", "quiet", "-i", path,
        "-vf", "format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Step 2: ffmpeg encode
    encode_path = str(Path(out_path).with_suffix(".noaudio.mp4"))
    encode_cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "superfast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        encode_path,
    ]
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Debug video encoder
    debug_encode_path = str(Path(str(out_path).replace(".mp4", ".debug.mp4")))
    debug_encoder = None
    if generate_debug and debug_path:
        debug_path = str(debug_path)
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        debug_encode_path = debug_path
        debug_cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{source_width}x{source_height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "superfast", "-crf", "30",
            "-pix_fmt", "yuv420p",
            debug_encode_path,
        ]
        debug_encoder = subprocess.Popen(debug_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 3: Process frames
    frame_size = source_width * source_height * 3
    frames_read = 0
    still_best_score = -1.0
    still_frame: Image.Image | None = None

    # Build track bbox lookup for debug
    track_times: dict[int, list[tuple[float, DetectedFace]]] = {}
    for track in tracks:
        for f in track.face_bboxes:
            track_times.setdefault(track.id, []).append((f.timestamp, f))

    def get_track_bbox_at(track_id: int, t: float) -> DetectedFace | None:
        samples = track_times.get(track_id, [])
        if not samples:
            return None
        if len(samples) == 1:
            return samples[0][1]
        # Find nearest
        best = min(samples, key=lambda s: abs(s[0] - t))
        return best[1]

    assert decoder.stdout is not None
    try:
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            t = frames_read / fps
            frame_idx = frames_read

            # Get crop box for this frame
            point = traj_by_frame.get(frame_idx)
            if point is None:
                # Interpolate from nearest
                point = _interpolate_trajectory(trajectory, t, source_width, source_height, target_ratio, fps)

            cx, cy = point.crop_x, point.crop_y
            cw, ch = point.crop_w, point.crop_h

            # Clamp to source boundaries
            cx = max(0, min(cx, source_width - cw))
            cy = max(0, min(cy, source_height - ch))

            # Crop the frame at source resolution, then resize to output dims
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(source_height, source_width, 3).copy()
            cropped = arr[cy:cy + ch, cx:cx + cw]

            # Resize cropped region to output dimensions
            pil_cropped = Image.fromarray(cropped)
            if pil_cropped.size != (out_w, out_h):
                pil_cropped = pil_cropped.resize((out_w, out_h), Image.Resampling.LANCZOS)
            resized = np.array(pil_cropped)

            # Write to encoder
            assert encoder.stdin is not None
            encoder.stdin.write(resized.tobytes())

            # Still extraction: pick the frame with best crop quality
            # (centered on speaker, face visible). Use the resized output frame.
            still_score = _frame_still_score(arr, point, tracks, t)
            if still_score > still_best_score:
                still_best_score = still_score
                # Extract still from the resized output frame (out_w x out_h)
                still_pil = Image.fromarray(resized)
                still_frame = still_pil

            # Debug overlay
            if debug_encoder is not None and debug_encoder.stdin is not None:
                debug_frame = _draw_debug_overlay(
                    arr, tracks, point, t, track_times,
                    get_track_bbox_at, source_width, source_height,
                )
                debug_encoder.stdin.write(debug_frame.tobytes())

            frames_read += 1

    finally:
        decoder.stdout.close()
        decoder.wait()
        assert encoder.stdin is not None
        encoder.stdin.close()
        encoder.wait()
        if debug_encoder is not None and debug_encoder.stdin is not None:
            debug_encoder.stdin.close()
            debug_encoder.wait()

    # ---- Mux audio from original ----
    mux_path = out_path
    final_path = out_path
    audio_mux_cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", encode_path,
        "-i", path,
        "-c", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-movflags", "+faststart",
        mux_path,
    ]
    mux_result = subprocess.run(audio_mux_cmd, capture_output=True, timeout=120)
    if mux_result.returncode != 0:
        # Fallback: just use the video-only output
        subprocess.run(["cp", encode_path, mux_path], check=False)

    Path(encode_path).unlink(missing_ok=True)

    # Clean up temporary encode path if it's different from final
    if encode_path != out_path:
        Path(encode_path).unlink(missing_ok=True)

    # ---- Save still (multiple aspect ratios) ----
    output_stem = Path(out_path).stem.replace("_9_16", "")
    still_dir = Path(stills_dir) if stills_dir else Path(out_path).parent
    still_dir.mkdir(parents=True, exist_ok=True)

    still_paths: list[str] = []
    if still_frame is not None:
        # Generate stills in 4 aspect ratios: 1:1, 16:9, 9:16, 4:5
        aspect_ratios = {"1_1": 1.0, "16_9": 16 / 9, "9_16": 9 / 16, "4_5": 4 / 5}
        for name, ratio in aspect_ratios.items():
            still_size = 1080
            if ratio >= 1.0:
                sw = still_size
                sh = int(round(still_size / ratio))
            else:
                sh = still_size
                sw = int(round(still_size * ratio))
            # Crop centered from the still frame, then resize
            fw, fh = still_frame.size
            src_ratio = fw / fh
            if src_ratio > ratio:
                # Source is wider, crop width
                new_w = int(round(fh * ratio))
                left = (fw - new_w) // 2
                box = (left, 0, left + new_w, fh)
            else:
                new_h = int(round(fw / ratio))
                top = (fh - new_h) // 2
                box = (0, top, fw, top + new_h)
            cropped = still_frame.crop(box)
            resized_still = cropped.resize((sw, sh), Image.Resampling.LANCZOS)
            still_path = str(still_dir / f"{output_stem}_{name}.jpg")
            resized_still.save(still_path, "JPEG", quality=90)
            still_paths.append(still_path)
        # Primary still path (1:1)
        still_out = str(still_dir / f"{output_stem}_1_1.jpg")
    else:
        # Fallback: extract middle frame at multiple ratios
        still_out = str(still_dir / f"{output_stem}_1_1.jpg")
        subprocess.run([
            "ffmpeg", "-y", "-v", "quiet", "-ss", str(total_frames / fps / 2),
            "-i", path, "-vframes", "1",
            "-vf", f"scale=1080:1080:force_original_aspect_ratio=decrease,"
                   f"pad=1080:1080:(ow-iw)/2:(oh-ih)/2",
            "-q:v", "2", still_out,
        ], check=False)

    # ---- Collect metadata ----
    reel_meta = _get_video_info(mux_path)
    validation_info = {
        "frames_rendered": frames_read,
        "frames_expected": total_frames,
        "all_frames_rendered": frames_read == total_frames,
    }

    return RenderResult(
        reel_path=mux_path,
        still_path=still_out,
        still_paths=still_paths,
        debug_path=debug_encode_path if generate_debug else None,
        reel_metadata=reel_meta,
        validation=validation_info,
    )


def _ensure_trajectory(
    traj: list[CropTrajectoryPoint],
    total_frames: int,
    sw: int, sh: int,
    ratio: float, fps: float,
) -> None:
    """Ensure trajectory has a point for every frame."""
    if not traj:
        from .cropper import _center_crop
        cr = _center_crop(sw, sh, ratio)
        for fi in range(total_frames):
            traj.append(CropTrajectoryPoint(
                time=fi / fps, center_x=0.5, center_y=0.5,
                crop_x=cr.x, crop_y=cr.y, crop_w=cr.width, crop_h=cr.height,
                speaker_id=-1, source_width=sw, source_height=sh,
            ))
        return

    first = traj[0]
    last = traj[-1]

    # Fill before first point
    if first.time > 0:
        fi = 0
        while fi / fps < first.time:
            traj.insert(0, CropTrajectoryPoint(
                time=fi / fps, center_x=first.center_x, center_y=first.center_y,
                crop_x=first.crop_x, crop_y=first.crop_y,
                crop_w=first.crop_w, crop_h=first.crop_h,
                speaker_id=first.speaker_id, source_width=sw, source_height=sh,
            ))
            fi += 1

    # Fill after last point
    fi = int(round(last.time * fps)) + 1
    last_frame = int(round(total_frames / fps * fps)) if total_frames else fi
    while fi < last_frame:
        traj.append(CropTrajectoryPoint(
            time=fi / fps, center_x=last.center_x, center_y=last.center_y,
            crop_x=last.crop_x, crop_y=last.crop_y,
            crop_w=last.crop_w, crop_h=last.crop_h,
            speaker_id=last.speaker_id, source_width=sw, source_height=sh,
        ))
        fi += 1


def _interpolate_trajectory(
    traj: list[CropTrajectoryPoint],
    t: float, sw: int, sh: int, ratio: float, fps: float,
) -> CropTrajectoryPoint:
    """Interpolate or extrapolate a trajectory point at time t."""
    if not traj:
        from .cropper import _center_crop
        cr = _center_crop(sw, sh, ratio)
        return CropTrajectoryPoint(
            time=t, center_x=0.5, center_y=0.5,
            crop_x=cr.x, crop_y=cr.y, crop_w=cr.width, crop_h=cr.height,
            speaker_id=-1, source_width=sw, source_height=sh,
        )

    if t <= traj[0].time:
        p = traj[0]
        return CropTrajectoryPoint(
            time=t, center_x=p.center_x, center_y=p.center_y,
            crop_x=p.crop_x, crop_y=p.crop_y,
            crop_w=p.crop_w, crop_h=p.crop_h,
            speaker_id=p.speaker_id, source_width=sw, source_height=sh,
        )

    if t >= traj[-1].time:
        p = traj[-1]
        return CropTrajectoryPoint(
            time=t, center_x=p.center_x, center_y=p.center_y,
            crop_x=p.crop_x, crop_y=p.crop_y,
            crop_w=p.crop_w, crop_h=p.crop_h,
            speaker_id=p.speaker_id, source_width=sw, source_height=sh,
        )

    # Find bracketing points
    for i in range(len(traj) - 1):
        if traj[i].time <= t <= traj[i + 1].time:
            p1, p2 = traj[i], traj[i + 1]
            if p2.time == p1.time:
                return p1
            ratio_t = (t - p1.time) / (p2.time - p1.time)
            return CropTrajectoryPoint(
                time=t,
                center_x=p1.center_x + (p2.center_x - p1.center_x) * ratio_t,
                center_y=p1.center_y + (p2.center_y - p1.center_y) * ratio_t,
                crop_x=int(round(p1.crop_x + (p2.crop_x - p1.crop_x) * ratio_t)),
                crop_y=int(round(p1.crop_y + (p2.crop_y - p1.crop_y) * ratio_t)),
                crop_w=p1.crop_w, crop_h=p1.crop_h,
                speaker_id=p1.speaker_id,
                source_width=sw, source_height=sh,
            )
    return traj[-1]


def _frame_still_score(
    arr: np.ndarray,
    point: CropTrajectoryPoint,
    tracks: list[TrackedPerson],
    t: float,
) -> float:
    """Score a frame for still extraction quality.

    Higher = better.  Rewards frames where the active speaker is well-framed.
    """
    # Base: speaker visible
    score = 0.0
    if point.speaker_id >= 0:
        score += 0.3

    # Check if speaker face is within crop bounds
    for track in tracks:
        if track.id == point.speaker_id:
            for bbox in reversed(track.face_bboxes):
                if abs(bbox.timestamp - t) < 2.0:
                    # Is face in the crop region?
                    cx = point.crop_x + point.crop_w / 2
                    cy = point.crop_y + point.crop_h / 2
                    face_cx = bbox.x + bbox.width / 2
                    face_cy = bbox.y + bbox.height / 2
                    dist = np.sqrt((face_cx - cx) ** 2 + (face_cy - cy) ** 2)
                    # Closer to center = better (normalize by image diagonal)
                    diag = max(arr.shape[1], arr.shape[0])
                    score += max(0, 0.3 - dist / diag * 0.3)
                    break
            break

    return score


def _draw_debug_overlay(
    arr: np.ndarray,
    tracks: list[TrackedPerson],
    point: CropTrajectoryPoint,
    t: float,
    track_times: dict[int, list[tuple[float, DetectedFace]]],
    get_bbox_fn: Any,
    sw: int, sh: int,
) -> np.ndarray:
    """Draw tracking + speaker + crop overlay on a frame."""
    # We can't draw directly on a numpy array with PIL easily, so use PIL
    img = Image.fromarray(arr).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay)

    # Draw tracked faces
    for track in tracks:
        # Find nearest bbox
        samples = track_times.get(track.id, [])
        if not samples:
            continue
        nearest = min(samples, key=lambda s: abs(s[0] - t))
        bbox = nearest[1]
        color = _track_color(track.id)

        # Determine if this is the active speaker
        is_speaker = track.id == point.speaker_id
        thickness = 3 if is_speaker else 1

        # Draw bbox
        draw.rectangle(
            [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
            outline=color + (255,), width=thickness,
        )

        # Label
        label = f"P{track.id}"
        if is_speaker:
            label += " [SPEAKING]"
        draw.text((bbox.x, bbox.y - 20), label, fill=color + (255,))

    # Draw crop rectangle
    draw.rectangle(
        [point.crop_x, point.crop_y, point.crop_x + point.crop_w, point.crop_y + point.crop_h],
        outline=(255, 255, 255, 255), width=2,
    )

    # Draw timestamp
    draw.text((10, 10), f"t={t:.1f}s", fill=(255, 255, 255, 255))

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(result)


def _get_video_info(path: str | Path) -> dict[str, Any]:
    """Get basic video info via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        info = json.loads(result.stdout)
    except Exception:
        return {}

    v_stream = None
    a_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and v_stream is None:
            v_stream = s
        elif s.get("codec_type") == "audio" and a_stream is None:
            a_stream = s

    return {
        "width": int(v_stream.get("width", 0)) if v_stream else 0,
        "height": int(v_stream.get("height", 0)) if v_stream else 0,
        "duration": float(info.get("format", {}).get("duration", 0)),
        "video_codec": v_stream.get("codec_name") if v_stream else None,
        "audio_codec": a_stream.get("codec_name") if a_stream else None,
        "fps": float(v_stream.get("r_frame_rate", "0/1").split("/")[0]) if v_stream else 0,
    }


def extract_still_from_video(
    video_path: str | Path,
    frame_time: float,
    output_path: str | Path,
    width: int = 1080,
    height: int | None = None,
) -> str:
    """Extract a single still frame from a video at a given timestamp."""
    if height is None:
        height = width  # square

    cmd = [
        "ffmpeg", "-y", "-v", "quiet", "-ss", str(frame_time),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return str(output_path)
