"""Video ingestion and metadata extraction.

Uses ffprobe to extract technical metadata from a video file:
  - duration, FPS, resolution
  - codec information
  - audio track presence and format

This is the first stage of the video pipeline.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoMetadata:
    """Technical metadata extracted from a video file."""

    width: int
    height: int
    duration_sec: float
    fps: float
    video_codec: str
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    total_frames: int
    bitrate_bps: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "total_frames": self.total_frames,
            "bitrate_bps": self.bitrate_bps,
        }


def extract_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract technical metadata from a video file using ffprobe."""
    path = str(video_path)
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
    info = json.loads(result.stdout)

    # Video stream
    v_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            v_stream = s
            break

    if v_stream is None:
        raise RuntimeError(f"No video stream found in {path}")

    width = int(v_stream["width"])
    height = int(v_stream["height"])
    fps_num, fps_den = v_stream.get("r_frame_rate", "0/1").split("/")
    fps = float(fps_num) / float(fps_den) if float(fps_den) else 0.0

    duration_str = info.get("format", {}).get("duration", "0")
    duration_sec = float(duration_str)
    total_frames = int(duration_sec * fps) if fps > 0 else 0

    # Audio stream
    a_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            a_stream = s
            break

    audio_codec = a_stream.get("codec_name") if a_stream else None
    audio_sample_rate = int(a_stream["sample_rate"]) if a_stream and a_stream.get("sample_rate") else None
    audio_channels = int(a_stream["channels"]) if a_stream and a_stream.get("channels") else None

    bitrate_bps = int(info.get("format", {}).get("bit_rate", 0)) if info.get("format", {}).get("bit_rate") else None

    return VideoMetadata(
        width=width,
        height=height,
        duration_sec=duration_sec,
        fps=fps,
        video_codec=v_stream.get("codec_name", "unknown"),
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        total_frames=total_frames,
        bitrate_bps=bitrate_bps,
    )
