from __future__ import annotations

"""
Phase 3 Technical Spike: Subject-aware 9:16 crop experiment (final).

Tests the full MediaPipe detection stack:
  1. Face Detection (blaze_face_full_range_sparse_float16.tflite)
  2. Face Landmarker (face_landmarker_float16.task) — precise eye/nose/mouth
  3. Pose Landmarker (pose_landmarker_full_float16.task) — body keypoints
  4. efficientdet_lite0_float16.tflite — object detection fallback

Crops derived from weighted centroid of all detections.
Compares proposed crop vs. naive center crop.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output/spike_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = DATA_DIR / "input_image.png"
TARGET_RATIO = 9 / 16  # width / height
TARGET_WIDTH = 576
TARGET_HEIGHT = int(TARGET_WIDTH / TARGET_RATIO)  # 1024

# ---------------------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------------------
img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]
print(f"Image: {w}x{h}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))

# ---------------------------------------------------------------------------
# 2. Face Detection
# ---------------------------------------------------------------------------
faces = []
face_model = MODELS_DIR / "blaze_face_full_range_sparse_float16.tflite"
print(f"\n[1] Face Detection: {face_model.name}")
try:
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(face_model)),
        min_detection_confidence=0.3,
    )
    detector = mp_vision.FaceDetector.create_from_options(opts)
    results = detector.detect(mp_image)
    if results.detections:
        for det in results.detections:
            bbox = det.bounding_box
            x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
            bw, bh = int(bbox.width), int(bbox.height)
            score = det.categories[0].score
            faces.append({"bbox": [x1, y1, bw, bh], "confidence": round(score, 4)})
            print(f"  Face: ({x1},{y1}) {bw}x{bh} conf={score:.3f}")
    else:
        print("  No faces detected")
    detector.close()
except Exception as e:
    print(f"  Error: {e}")

# ---------------------------------------------------------------------------
# 3. Face Landmarker (eyes, nose, mouth for precise centering)
# ---------------------------------------------------------------------------
face_lm_results = []
lm_model = MODELS_DIR / "face_landmarker_float16.task"
print(f"\n[2] Face Landmarker: {lm_model.name}")
try:
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(lm_model)),
        num_faces=5,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    lm_results = landmarker.detect(mp_image)
    if lm_results.face_landmarks:
        # Key indices: nose tip=1, left eye center=33, right eye center=263,
        # left mouth corner=61, right mouth corner=291
        key_indices = [1, 33, 263, 61, 291]
        for i, landmarks in enumerate(lm_results.face_landmarks):
            key_pts = []
            for idx in key_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    key_pts.append((int(lm.x * w), int(lm.y * h)))
            face_lm_results.append({
                "face_id": i,
                "landmark_count": len(landmarks),
                "key_points": key_pts,
            })
            cx = int(np.mean([p[0] for p in key_pts]))
            cy = int(np.mean([p[1] for p in key_pts]))
            print(f"  Face {i}: {len(landmarks)} landmarks, facial centroid ({cx},{cy})")
    else:
        print("  No face landmarks detected")
    landmarker.close()
except Exception as e:
    print(f"  Error: {e}")

# ---------------------------------------------------------------------------
# 4. Pose Landmarker (body keypoints for watermark-masked / body fallback)
# ---------------------------------------------------------------------------
pose_kps = []
pose_model = MODELS_DIR / "pose_landmarker_full_float16.task"
print(f"\n[3] Pose Landmarker: {pose_model.name}")
try:
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model)),
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(opts)
    pose_results = pose_detector.detect(mp_image)
    if pose_results.pose_landmarks:
        # Key indices: left shoulder=11, right shoulder=12, left hip=23, right hip=24
        key_indices = [11, 12, 23, 24]
        for i, landmarks in enumerate(pose_results.pose_landmarks):
            for idx in key_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    pose_kps.append((int(lm.x * w), int(lm.y * h), lm.visibility))
            print(f"  Pose {i}: {len(landmarks)} landmarks")
    else:
        print("  No pose landmarks detected")
    pose_detector.close()
except Exception as e:
    print(f"  Error: {e}")

# ---------------------------------------------------------------------------
# 5. Crop derivation: weighted centroid from all detections
# ---------------------------------------------------------------------------
detections_for_centroid = []

# Face landmark centroids (highest weight ×3)
for f in face_lm_results:
    cx = int(np.mean([p[0] for p in f["key_points"]]))
    cy = int(np.mean([p[1] for p in f["key_points"]]))
    detections_for_centroid.append((cx, cy, 3.0))

# Face bounding box centroids (weight 2)
for f in faces:
    cx = f["bbox"][0] + f["bbox"][2] // 2
    cy = f["bbox"][1] + f["bbox"][3] // 2
    detections_for_centroid.append((cx, cy, 2.0))

# Pose keypoints (weight 1)
for px, py, vis in pose_kps:
    if vis > 0.5:
        detections_for_centroid.append((px, py, 1.0))

if detections_for_centroid:
    total_w = sum(d[2] for d in detections_for_centroid)
    cx = sum(d[0] * d[2] for d in detections_for_centroid) / total_w
    cy = sum(d[1] * d[2] for d in detections_for_centroid) / total_w
    cx, cy = int(cx), int(cy)
    print(f"\n[4] Combined centroid: ({cx}, {cy}) from {len(detections_for_centroid)} detections")
else:
    cx, cy = w // 2, h // 2
    print(f"\n[4] No detections, using image center: ({cx}, {cy})")

# ---------------------------------------------------------------------------
# 6. Crop derivation
# ---------------------------------------------------------------------------
def crop_for_ratio(img, cx, cy, ratio, target_h=1024):
    target_w = int(target_h * ratio)
    half_w, half_h = target_w // 2, target_h // 2
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(w, x1 + target_w)
    y2 = min(h, y1 + target_h)
    x1 = max(0, x2 - target_w)
    y1 = max(0, y2 - target_h)
    return img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

proposed_crop, proposed_box = crop_for_ratio(img, cx, cy, TARGET_RATIO, TARGET_HEIGHT)
center_crop, center_box = crop_for_ratio(img, w // 2, h // 2, TARGET_RATIO, TARGET_HEIGHT)

px1, py1, pw, ph = proposed_box
cx1, cy1, cw, ch = center_box

print(f"\nProposed crop: x={px1} y={py1} {pw}x{ph}")
print(f"Center crop:   x={cx1} y={cy1} {cw}x{ch}")
print(f"Centroid offset: ({cx - w//2}, {cy - h//2}) px = ({round((cx-w//2)/w*100,2)}%, {round((cy-h//2)/h*100,2)}%)")

# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------
vis = img.copy()

# Face detection boxes (green)
for face in faces:
    bx, by, bw, bh = face["bbox"]
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 8)
    cv2.putText(vis, f"{face['confidence']:.2f}", (bx, by - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

# Face landmarks (blue circles)
for f in face_lm_results:
    for (lx, ly) in f["key_points"]:
        cv2.circle(vis, (lx, ly), 10, (255, 0, 0), -1)

# Pose keypoints (orange)
for px, py, _ in pose_kps:
    cv2.circle(vis, (px, py), 12, (0, 165, 255), -1)

# Proposed crop box (magenta)
cv2.rectangle(vis, (px1, py1), (px1 + pw, py1 + ph), (255, 0, 255), 8)
cv2.putText(vis, "PROPOSED 9:16", (px1, py1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
cv2.circle(vis, (cx, cy), 15, (255, 0, 255), -1)

# Center crop box (yellow)
cv2.rectangle(vis, (cx1, cy1), (cx1 + cw, cy1 + ch), (0, 255, 255), 8)
cv2.putText(vis, "CENTER 9:16", (cx1, cy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
cv2.circle(vis, (w // 2, h // 2), 15, (0, 255, 255), -1)

# Save
cv2.imwrite(str(OUTPUT_DIR / "01_original_with_overlays.png"), vis)
vis_small = cv2.resize(vis, (1000, 1000))
cv2.imwrite(str(OUTPUT_DIR / "01_original_with_overlays_small.jpg"), vis_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(str(OUTPUT_DIR / "02_proposed_crop.png"), proposed_crop)
cv2.imwrite(str(OUTPUT_DIR / "02_proposed_crop_small.jpg"),
            cv2.resize(proposed_crop, (TARGET_WIDTH, TARGET_HEIGHT)),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(str(OUTPUT_DIR / "03_center_crop.png"), center_crop)
cv2.imwrite(str(OUTPUT_DIR / "03_center_crop_small.jpg"),
            cv2.resize(center_crop, (TARGET_WIDTH, TARGET_HEIGHT)),
            [cv2.IMWRITE_JPEG_QUALITY, 85])

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------
report = {
    "image": str(IMAGE_PATH),
    "image_size": {"width": w, "height": h},
    "target_ratio": TARGET_RATIO,
    "models": {
        "face_detection": "blaze_face_full_range_sparse_float16.tflite",
        "face_landmarker": "face_landmarker_float16.task",
        "pose_landmarker": "pose_landmarker_full_float16.task",
        "object_detection": "efficientdet_lite0_float16.tflite (fallback, not used)",
    },
    "face_detections": faces,
    "face_landmarks": face_lm_results,
    "pose_keypoints": [{"x": p[0], "y": p[1], "visibility": round(p[2], 4)} for p in pose_kps],
    "combined_centroid": [cx, cy],
    "center_centroid": [w // 2, h // 2],
    "centroid_offset": {
        "px": cx - w // 2, "py": cy - h // 2,
        "pct_x": round((cx - w // 2) / w * 100, 2),
        "pct_y": round((cy - h // 2) / h * 100, 2),
    },
    "proposed_crop": {"box": [px1, py1, pw, ph]},
    "center_crop": {"box": [cx1, cy1, cw, ch]},
}
with open(OUTPUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✅ All outputs saved to {OUTPUT_DIR}/")
print(json.dumps(report, indent=2))
