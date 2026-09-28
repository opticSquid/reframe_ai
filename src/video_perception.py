"""Video perception: frame sampling, face detection, tracking, and landmarks.

Reuses the MediaPipe blaze-face model already used by the image pipeline
(`models/blaze_face_full_range_sparse_float16.tflite`) and the face
landmarker model (`models/face_landmarker_float16.task`).

Key design: ``VideoPerceiver`` creates MediaPipe detectors once and reuses
them across all frames — creating a detector per-frame (as a naive approach
would) is ~0.4s/frame; reusing drops it to ~0.02s/frame.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from .config import (
    FACE_DETECTION_MODEL,
    PERSON_DETECTION_MODEL,
    MODELS_DIR,
    MIN_FACE_CONFIDENCE,
    MIN_POSE_CONFIDENCE,
    MIN_OBJECT_CONFIDENCE,
    POSE_LANDMARKER_MODEL,
)
from .video_ingestion import VideoMetadata

# Face landmark indices for mouth aspect ratio (MAR) computation
MAR_TOP = 13
MAR_BOTTOM = 14
MAR_LEFT = 78
MAR_RIGHT = 308

# Distinct colors for track visualization
_TRACK_COLORS: list[tuple[int, int, int]] = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255),
    (255, 255, 50), (255, 50, 255), (50, 255, 255),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


@dataclass
class DetectedFace:
    """A face detection on a single frame."""

    frame_idx: int
    timestamp: float
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class DetectedPerson:
    """A person detection on a single frame (full-body bbox)."""

    frame_idx: int
    timestamp: float
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class DetectedPose:
    """A person pose detection on a single frame."""

    frame_idx: int
    timestamp: float
    # 33 pose landmarks normalized (0–1) × (width, height) of detection frame
    landmarks: list[tuple[float, float]]
    nose: tuple[float, float] | None = None       # (x, y) normalized
    left_shoulder: tuple[float, float] | None = None
    right_shoulder: tuple[float, float] | None = None
    left_hip: tuple[float, float] | None = None
    right_hip: tuple[float, float] | None = None
    left_knee: tuple[float, float] | None = None
    right_knee: tuple[float, float] | None = None
    left_ankle: tuple[float, float] | None = None
    right_ankle: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None  # (x, y, w, h) in pixels


@dataclass
class TrackedPerson:
    """A person tracked across frames with a stable ID."""

    id: int
    face_bboxes: list[DetectedFace] = field(default_factory=list)
    person_bboxes: list[DetectedPerson] = field(default_factory=list)
    """Full-body person bboxes (larger, more stable than face bboxes).
    Used for trajectory / crop centering. Falls back to face bboxes when no
    person detector is available."""
    lost_count: int = 0
    last_mar: float | None = None
    color: tuple[int, int, int] = (0, 255, 0)


@dataclass
class FrameSample:
    """A sampled frame with its detections and track associations."""

    frame_idx: int
    timestamp: float
    faces: list[DetectedFace]
    persons: list[DetectedPerson] = field(default_factory=list)


@dataclass
class PerceptionResult:
    """Output of the perception stage."""

    metadata: VideoMetadata
    sample_interval: int
    tracks: list[TrackedPerson]
    sample_results: list[FrameSample]
    shot_boundaries: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames_by_interval(
    video_path: str | Path,
    metadata: VideoMetadata,
    target_fps: float = 4.0,
    max_dim: int = 640,
) -> list[tuple[int, float, bytes]]:
    """Extract frames at a target sampling FPS using ffmpeg.

    Returns list of (frame_idx, timestamp, raw_rgb_bytes) tuples.
    Scales to ``max_dim`` (longest side) to keep MediaPipe inference fast.
    """
    path = str(video_path)
    fps = metadata.fps

    orig_w, orig_h = metadata.width, metadata.height
    if orig_w >= orig_h:
        out_w = max_dim
        out_h = int(round(max_dim * orig_h / orig_w))
    else:
        out_h = max_dim
        out_w = int(round(max_dim * orig_w / orig_h))

    cmd = [
        "ffmpeg", "-y", "-v", "quiet", "-i", path,
        "-vf", f"fps={target_fps},scale={out_w}:{out_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed: {result.stderr.decode('utf-8', errors='replace')}"
        )

    frame_size = out_w * out_h * 3
    num_frames = len(result.stdout) // frame_size

    frames: list[tuple[int, float, bytes]] = []
    for i in range(num_frames):
        offset = i * frame_size
        raw = result.stdout[offset:offset + frame_size]
        orig_frame_idx = int(round(i * fps / target_fps))
        timestamp = orig_frame_idx / fps
        frames.append((orig_frame_idx, timestamp, raw))

    return frames


def raw_to_numpy(raw: bytes, width: int, height: int) -> np.ndarray:
    """Convert raw RGB bytes to an HxW×3 numpy array."""
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def get_scaled_dims(metadata: VideoMetadata, max_dim: int = 640) -> tuple[int, int]:
    """Compute scaled dimensions when scaling to max_dim on the longest side."""
    ow, oh = metadata.width, metadata.height
    if ow >= oh:
        return max_dim, int(round(max_dim * oh / ow))
    return int(round(max_dim * ow / oh)), max_dim


# ---------------------------------------------------------------------------
# Face landmark / MAR
# ---------------------------------------------------------------------------
def compute_mar(landmarks: list[Any]) -> float:
    """Compute Mouth Aspect Ratio from 478 face landmarks.

    MAR = (vertical_lip_distance) / (horizontal_lip_width)
    High MAR = mouth open = likely speaking.
    """
    p13 = landmarks[MAR_TOP]
    p14 = landmarks[MAR_BOTTOM]
    p78 = landmarks[MAR_LEFT]
    p308 = landmarks[MAR_RIGHT]

    vertical = abs(p14.y - p13.y)
    horizontal = abs(p308.x - p78.x)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


# ---------------------------------------------------------------------------
# Simple IOU + centroid tracker
# ---------------------------------------------------------------------------
_BBoxLike = DetectedFace | DetectedPerson


def _bbox_center(f: _BBoxLike) -> tuple[float, float]:
    return (f.x + f.width / 2, f.y + f.height / 2)


def _centroid_dist(a: _BBoxLike, b: _BBoxLike, max_dim: float) -> float:
    """Normalized centroid distance (0–1) between two bboxes."""
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2) / max_dim)


def _iou(a: _BBoxLike, b: _BBoxLike) -> float:
    ax1, ay1, aw, ah = a.x, a.y, a.width, a.height
    bx1, by1, bw, bh = b.x, b.y, b.width, b.height
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax1 + aw, bx1 + bw)
    iy2 = min(ay1 + ah, by1 + bh)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_contains(inner: DetectedFace, outer: DetectedPerson) -> bool:
    """Check if face bbox *inner* is contained within person bbox *outer*."""
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.width <= outer.x + outer.width
        and inner.y + inner.height <= outer.y + outer.height
    )


class SimpleTracker:
    """Lightweight IOU + centroid tracker with ID persistence.

    Primary tracking uses **person bboxes** (EfficientDet) when available —
    these are ~5-10x larger than face bboxes, making them far more stable
    for IOU matching.  Face bboxes are associated to tracks by containment
    for MAR/speaker inference.

    Falls back to face-only tracking when person detection is unavailable.
    """

    def __init__(
        self,
        max_lost: int = 30,
        iou_threshold: float = 0.15,
        dist_threshold: float = 0.5,
    ):
        self._next_id = 0
        self._tracks: list[TrackedPerson] = []
        self._max_lost = max_lost
        self._iou_threshold = iou_threshold
        self._dist_threshold = dist_threshold

    @property
    def tracks(self) -> list[TrackedPerson]:
        return self._tracks

    def update(
        self,
        faces: list[DetectedFace],
        img_width: int,
        img_height: int,
        persons: list[DetectedPerson] | None = None,
    ) -> list[int]:
        """Update tracker with detections from a new frame.

        When ``persons`` is provided, they are used as the primary tracking
        signal (larger, more stable bboxes).  Each face is then associated to
        the track whose person bbox contains it.  Returns track IDs
        corresponding to the **person** detections (or faces when no
        persons were detected).
        """
        img_diag = max(img_width, img_height) * 1.414
        primary = persons if persons is not None else faces

        if not self._tracks:
            assignments: list[int] = []
            for p in primary:
                track = TrackedPerson(id=self._next_id)
                if persons is not None:
                    track.person_bboxes.append(p)  # type: ignore[arg-type]
                else:
                    track.face_bboxes.append(p)  # type: ignore[arg-type]
                track.color = _track_color(self._next_id)
                self._tracks.append(track)
                assignments.append(self._next_id)
                self._next_id += 1
            # Associate faces to the newly created tracks
            self._associate_faces(primary, faces)
            return assignments

        assignments = [-1] * len(primary)
        used_tracks: set[int] = set()

        for fi, det in enumerate(primary):
            best_track_idx = -1
            best_score = 0.0
            for ti, track in enumerate(self._tracks):
                if ti in used_tracks:
                    continue
                last_primary = (
                    track.person_bboxes[-1] if track.person_bboxes
                    else track.face_bboxes[-1] if track.face_bboxes
                    else None
                )
                if last_primary is None:
                    continue
                i = _iou(det, last_primary)
                d = _centroid_dist(det, last_primary, img_diag)
                score = i if i >= self._iou_threshold else 0.0
                if score == 0.0 and d < self._dist_threshold:
                    score = (1.0 - d) * 0.5
                if score > best_score:
                    best_score = score
                    best_track_idx = ti

            if best_track_idx >= 0:
                used_tracks.add(best_track_idx)
                if persons is not None:
                    track = self._tracks[best_track_idx]
                    track.person_bboxes.append(det)  # type: ignore[arg-type]
                else:
                    self._tracks[best_track_idx].face_bboxes.append(det)  # type: ignore[arg-type]
                self._tracks[best_track_idx].lost_count = 0
                assignments[fi] = self._tracks[best_track_idx].id
            else:
                track = TrackedPerson(id=self._next_id)
                if persons is not None:
                    track.person_bboxes.append(det)  # type: ignore[arg-type]
                else:
                    track.face_bboxes.append(det)  # type: ignore[arg-type]
                track.color = _track_color(self._next_id)
                self._tracks.append(track)
                assignments[fi] = self._next_id
                self._next_id += 1

        # When tracking persons, associate face bboxes to tracks by containment
        if persons is not None:
            self._associate_faces(persons, faces)

        # Increment lost_count for tracks that weren't updated
        for ti, track in enumerate(self._tracks):
            if ti not in used_tracks:
                track.lost_count += 1

        self._tracks = [t for t in self._tracks if t.lost_count <= self._max_lost]
        return assignments

    def _associate_faces(self, primary: list, faces: list[DetectedFace]) -> None:
        """Associate face bboxes to tracks by containment in person bbox.

        When a face falls inside a track's most recent person bbox, append
        it to that track's ``face_bboxes`` for MAR computation.
        """
        for face in faces:
            assigned = False
            for track in self._tracks:
                if track.person_bboxes:
                    person = track.person_bboxes[-1]
                    if _bbox_contains(face, person):
                        track.face_bboxes.append(face)
                        assigned = True
                        break
            if not assigned:
                # Fallback: append to nearest track by centroid distance
                best_track = None
                best_dist = float("inf")
                fcx, fcy = _bbox_center(face)
                for track in self._tracks:
                    if track.face_bboxes:
                        lx, ly = _bbox_center(track.face_bboxes[-1])
                    elif track.person_bboxes:
                        lx, ly = _bbox_center(track.person_bboxes[-1])
                    else:
                        continue
                    dist = abs(fcx - lx) + abs(fcy - ly)
                    if dist < best_dist:
                        best_dist = dist
                        best_track = track
                if best_track is not None:
                    best_track.face_bboxes.append(face)
                    assigned = True
            # If still unassigned, face is orphaned (tracked via persons)


# ---------------------------------------------------------------------------
# Reusable MediaPipe detector wrapper
# ---------------------------------------------------------------------------
class VideoPerceiver:
    """Reusable MediaPipe face detector + face landmarker for video.

    Creates detectors once and reuses them across all frames.
    """

    def __init__(
        self,
        min_face_confidence: float = MIN_FACE_CONFIDENCE,
        num_faces: int = 10,
    ):
        self._min_face_confidence = min_face_confidence
        self._face_detector: Any = None
        self._face_landmarker: Any = None
        self._pose_landmarker: Any = None
        self._person_detector: Any = None
        self._num_faces = num_faces

    def _get_face_detector(self) -> Any:
        if self._face_detector is not None:
            return self._face_detector
        model_path = str(FACE_DETECTION_MODEL)
        if not Path(model_path).exists():
            return None
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=self._min_face_confidence,
            min_suppression_threshold=0.3,
        )
        self._face_detector = mp_vision.FaceDetector.create_from_options(opts)
        return self._face_detector

    def _get_face_landmarker(self) -> Any:
        if self._face_landmarker is not None:
            return self._face_landmarker
        model_path = str(MODELS_DIR / "face_landmarker_float16.task")
        if not Path(model_path).exists():
            return None
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_faces=self._num_faces,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        return self._face_landmarker

    def _get_pose_landmarker(self) -> Any:
        """Create or return the cached pose landmarker."""
        if self._pose_landmarker is not None:
            return self._pose_landmarker
        from .config import POSE_LANDMARKER_MODEL
        model_path = str(POSE_LANDMARKER_MODEL)
        if not Path(model_path).exists():
            return None
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_poses=10,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        return self._pose_landmarker

    def detect_faces(self, rgb_array: np.ndarray) -> list[DetectedFace]:
        """Run face detection on a frame. Returns empty list if model missing."""
        detector = self._get_face_detector()
        if detector is None:
            return []

        h, w = rgb_array.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_array))
        try:
            results = detector.detect(mp_img)
        except Exception:
            return []

        faces: list[DetectedFace] = []
        if results.detections:
            for det in results.detections:
                bbox = det.bounding_box
                score = det.categories[0].score if det.categories else 0.5
                faces.append(DetectedFace(
                    frame_idx=0, timestamp=0.0,
                    x=int(bbox.origin_x), y=int(bbox.origin_y),
                    width=int(bbox.width), height=int(bbox.height),
                    confidence=float(score),
                ))
        return faces

    def compute_mars(
        self,
        rgb_array: np.ndarray,
        faces: list[DetectedFace],
    ) -> list[float | None]:
        """Compute Mouth Aspect Ratio for each face.

        Returns a list of MAR values (or None if landmarks unavailable).
        """
        if not faces:
            return []

        landmarker = self._get_face_landmarker()
        if landmarker is None:
            return [None] * len(faces)

        h, w = rgb_array.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_array))
        try:
            results = landmarker.detect(mp_img)
        except Exception:
            return [None] * len(faces)

        if not results.face_landmarks:
            return [None] * len(faces)

        # Match landmarks to faces by centroid proximity
        face_centers = [_bbox_center(f) for f in faces]
        img_diag = max(w, h) * 1.414
        mars: list[float | None] = [None] * len(faces)

        for lm_list in results.face_landmarks:
            if len(lm_list) < 309:
                continue
            lm_cx = sum(l.x for l in lm_list[:468]) / 468
            lm_cy = sum(l.y for l in lm_list[:468]) / 468
            lm_cx_px = lm_cx * w
            lm_cy_px = lm_cy * h

            best_idx = -1
            best_dist = float("inf")
            for i, (fcx, fcy) in enumerate(face_centers):
                dist = ((lm_cx_px - fcx) ** 2 + (lm_cy_px - fcy) ** 2) / (img_diag ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0 and mars[best_idx] is None:
                mars[best_idx] = compute_mar(lm_list)

        return mars

    def detect_poses(self, rgb_array: np.ndarray) -> list[DetectedPose]:
        """Run pose detection on a frame. Returns empty list if model missing."""
        landmarker = self._get_pose_landmarker()
        if landmarker is None:
            return []

        h, w = rgb_array.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_array))
        try:
            results = landmarker.detect(mp_img)
        except Exception:
            return []

        poses: list[DetectedPose] = []
        # MediaPipe pose landmark indices
        LANDMARK_NAMES = {
            0: "nose", 11: "left_shoulder", 12: "right_shoulder",
            23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle",
        }
        if results.pose_landmarks:
            for lm_list in results.pose_landmarks:
                if len(lm_list) < 33:
                    continue
                landmarks = [(lm.x, lm.y) for lm in lm_list[:33]]
                named = {LANDMARK_NAMES[i]: (lm_list[i].x, lm_list[i].y)
                         for i in LANDMARK_NAMES if i < len(lm_list)}
                # Compute bounding box from all landmarks
                valid_x = [lm.x for lm in lm_list[:33] if lm.visibility > 0.5]
                valid_y = [lm.y for lm in lm_list[:33] if lm.visibility > 0.5]
                if valid_x and valid_y:
                    bx = int(min(valid_x) * w)
                    by = int(min(valid_y) * h)
                    bw = int((max(valid_x) - min(valid_x)) * w)
                    bh = int((max(valid_y) - min(valid_y)) * h)
                    bbox = (float(bx), float(by), float(bw), float(bh))
                else:
                    bbox = None
                poses.append(DetectedPose(
                    frame_idx=0, timestamp=0.0,
                    landmarks=landmarks,
                    nose=named.get("nose"),
                    left_shoulder=named.get("left_shoulder"),
                    right_shoulder=named.get("right_shoulder"),
                    left_hip=named.get("left_hip"),
                    right_hip=named.get("right_hip"),
                    left_knee=named.get("left_knee"),
                    right_knee=named.get("right_knee"),
                    left_ankle=named.get("left_ankle"),
                    right_ankle=named.get("right_ankle"),
                    bbox=bbox,
                ))
        return poses

    def _get_person_detector(self) -> Any:
        """Create or return the cached EfficientDet person detector."""
        if self._person_detector is not None:
            return self._person_detector
        model_path = str(PERSON_DETECTION_MODEL)
        if not Path(model_path).exists():
            return None
        opts = mp_vision.ObjectDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            score_threshold=MIN_OBJECT_CONFIDENCE,
            max_results=10,
        )
        self._person_detector = mp_vision.ObjectDetector.create_from_options(opts)
        return self._person_detector

    def detect_persons(self, rgb_array: np.ndarray) -> list[DetectedPerson]:
        """Run person detection (EfficientDet) on a frame.

        Returns full-body bounding boxes — larger and more stable than face-only
        bboxes for tracking and crop centering.
        """
        detector = self._get_person_detector()
        if detector is None:
            return []

        h, w = rgb_array.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_array))
        try:
            results = detector.detect(mp_img)
        except Exception:
            return []

        persons: list[DetectedPerson] = []
        if results.detections:
            for det in results.detections:
                category = det.categories[0] if det.categories else None
                if category is None:
                    continue
                name = getattr(category, "category_name", "") or ""
                if name.lower() not in ("person", "person ", "people"):
                    continue
                bbox = det.bounding_box
                persons.append(DetectedPerson(
                    frame_idx=0, timestamp=0.0,
                    x=int(bbox.origin_x), y=int(bbox.origin_y),
                    width=int(bbox.width), height=int(bbox.height),
                    confidence=float(category.score),
                ))
        return persons

    def detect_all(self, rgb_array: np.ndarray) -> tuple[list[DetectedFace], list[DetectedPerson], list[DetectedPose]]:
        """Run all three detectors in one call — avoids redundant MediaPipe setup."""
        faces = self.detect_faces(rgb_array)
        persons = self.detect_persons(rgb_array)
        poses = self.detect_poses(rgb_array)
        return faces, persons, poses

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None
        if self._face_landmarker is not None:
            self._face_landmarker.close()
            self._face_landmarker = None
        if self._person_detector is not None:
            self._person_detector.close()
            self._person_detector = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None


# ---------------------------------------------------------------------------
# Shot boundary detection
# ---------------------------------------------------------------------------
def detect_shot_boundaries(
    frames: list[tuple[int, float, bytes]],
    width: int,
    height: int,
    threshold: float = 0.25,
) -> list[int]:
    """Detect shot boundaries using frame-to-frame pixel difference.

    Returns a list of frame indices where a shot change is detected.
    """
    boundaries: list[int] = []
    prev_gray: np.ndarray | None = None

    for frame_idx, timestamp, raw in frames:
        arr = raw_to_numpy(raw, width, height)
        gray = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.float32) / 255.0

        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff > threshold:
                boundaries.append(frame_idx)

        prev_gray = gray

    return boundaries
