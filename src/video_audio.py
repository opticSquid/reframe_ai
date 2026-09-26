"""Audio/speech analysis for the video pipeline.

Extracts audio from video via ffmpeg, then performs speech activity
detection using numpy RMS energy analysis.  No heavy ML dependencies
(librosa/scipy/torchaudio are not installed).

The approach:
  1. Extract mono 16kHz PCM audio from video via ffmpeg.
  2. Compute RMS energy in 50ms windows.
  3. Threshold (relative to max energy) to identify speech-active regions.
  4. Apply morphological smoothing (min-gap, min-duration) to clean up segments.

This is intentionally lightweight — good enough for the hackathon scenario
of a two-person interview where speech clearly dominates silence.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .video_ingestion import VideoMetadata


@dataclass
class SpeechSegment:
    """A contiguous interval where speech is detected."""

    start: float
    end: float
    energy: float  # normalized RMS energy (0–1)


@dataclass
class SpeechActivity:
    """Speech activity result for a full video."""

    sample_rate: int
    window_ms: int
    energy_curve: list[float]  # normalized RMS per window
    speech_mask: list[bool]    # True where speech detected
    segments: list[SpeechSegment]
    duration_sec: float

    def is_speech_at(self, time_sec: float) -> bool:
        """Check if speech is active at the given timestamp."""
        if not self.speech_mask:
            return False
        idx = int(time_sec / (self.window_ms / 1000.0))
        idx = max(0, min(idx, len(self.speech_mask) - 1))
        return self.speech_mask[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "window_ms": self.window_ms,
            "duration_sec": self.duration_sec,
            "segments": [
                {"start": s.start, "end": s.end, "energy": s.energy}
                for s in self.segments
            ],
            "speech_ratio": sum(self.speech_mask) / max(len(self.speech_mask), 1),
        }


def extract_audio(
    video_path: str | Path,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """Extract mono audio from a video file as a numpy float32 array.

    Returns (samples, sample_rate) where samples are in [-1, 1].
    """
    path = str(video_path)
    # Write to a temp WAV file then read raw PCM
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-v", "quiet", "-i", path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.decode('utf-8', errors='replace')}")

        # Read the WAV file (skip 44-byte header)
        with open(wav_path, "rb") as f:
            f.seek(44)
            raw = f.read()
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sample_rate
    finally:
        Path(wav_path).unlink(missing_ok=True)


def detect_speech_activity(
    video_path: str | Path,
    metadata: VideoMetadata,
    window_ms: int = 50,
    energy_threshold_pct: float = 0.10,
    min_speech_gap_ms: int = 200,
    min_speech_dur_ms: int = 100,
) -> SpeechActivity:
    """Detect speech activity in a video's audio track.

    Parameters
    ----------
    video_path
        Path to the video file.
    metadata
        VideoMetadata for this video.
    window_ms
        Analysis window size in milliseconds.
    energy_threshold_pct
        Energy threshold as fraction of the 95th percentile.
    min_speech_gap_ms
        Minimum gap between speech segments (merge segments closer than this).
    min_speech_dur_ms
        Minimum duration for a speech segment to be kept.
    """
    samples, sr = extract_audio(video_path, sample_rate=16000)
    duration_sec = len(samples) / sr

    window_size = int(sr * window_ms / 1000.0)
    if window_size < 1:
        window_size = 1

    # Compute RMS energy per window
    num_windows = max(1, len(samples) // window_size)
    energies: list[float] = []
    for i in range(num_windows):
        start = i * window_size
        end = min(start + window_size, len(samples))
        chunk = samples[start:end]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)

    if not energies:
        return SpeechActivity(
            sample_rate=sr, window_ms=window_ms,
            energy_curve=[], speech_mask=[], segments=[],
            duration_sec=duration_sec,
        )

    # Normalize energy curve to [0, 1]
    max_e = max(energies)
    if max_e > 0:
        norm_energies = [e / max_e for e in energies]
    else:
        norm_energies = [0.0] * len(energies)

    # Threshold: use percentile-based rather than absolute
    sorted_e = sorted(norm_energies)
    p95 = sorted_e[int(len(sorted_e) * 0.95)] if len(sorted_e) > 0 else 0.0
    threshold = max(p95 * energy_threshold_pct, 0.05)

    # Initial speech mask
    speech_mask = [e > threshold for e in norm_energies]

    # Merge short gaps: if gap < min_speech_gap_ms / window_ms, fill it
    gap_max = max(1, min_speech_gap_ms // window_ms)
    i = 0
    smoothed = list(speech_mask)
    while i < len(smoothed):
        if not smoothed[i]:
            # Count gap length
            gap_start = i
            while i < len(smoothed) and not smoothed[i]:
                i += 1
            gap_len = i - gap_start
            if gap_len <= gap_max and gap_start > 0 and i < len(smoothed):
                # Fill the gap
                for j in range(gap_start, i):
                    smoothed[j] = True
        else:
            i += 1

    # Remove short speech segments
    min_dur_windows = max(1, min_speech_dur_ms // window_ms)
    final_mask = list(smoothed)
    i = 0
    while i < len(final_mask):
        if final_mask[i]:
            seg_start = i
            while i < len(final_mask) and final_mask[i]:
                i += 1
            seg_len = i - seg_start
            if seg_len < min_dur_windows:
                for j in range(seg_start, i):
                    final_mask[j] = False
        else:
            i += 1

    # Build segments
    segments: list[SpeechSegment] = []
    i = 0
    while i < len(final_mask):
        if final_mask[i]:
            seg_start = i
            while i < len(final_mask) and final_mask[i]:
                i += 1
            seg_end = i
            start_time = seg_start * window_ms / 1000.0
            end_time = seg_end * window_ms / 1000.0
            seg_energy = max(norm_energies[seg_start:seg_end]) if seg_start < seg_end else 0.0
            segments.append(SpeechSegment(
                start=start_time, end=end_time, energy=seg_energy,
            ))
        else:
            i += 1

    return SpeechActivity(
        sample_rate=sr,
        window_ms=window_ms,
        energy_curve=norm_energies,
        speech_mask=final_mask,
        segments=segments,
        duration_sec=duration_sec,
    )
