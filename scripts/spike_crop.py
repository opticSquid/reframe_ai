from __future__ import annotations

"""
Phase 3 Technical Spike: Subject-aware 9:16 crop experiment.

Goal: Can MediaPipe face detection produce a visibly better 9:16 crop
than naive center-cropping for the organizer's sample image?

Uses MediaPipe 1.0 Tasks API. Tries FaceDetector with face-specific model;
falls back to ObjectDetector (efficientdet_lite0) to detect 'person'/'face'.
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
TARGET_HEIGHT = int(TARGET_WIDTH / TARGET_RATIO)

# ---------------------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------------------
img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]
print(f"Image: {w}x{h}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))

# ---------------------------------------------------------------------------
# 2. MediaPipe Face Detection (BlazeFace full-range)
# ---------------------------------------------------------------------------
# Try face-specific model first
face_model = None
for name in [
    "blaze_face_full_range_sparse.tflite",
    "face_detection_short.tflite",
    "face_detection_front.tflite",
    "face_detection_short_front.tflite",
]:
    p = MODELS_DIR / name
    if p.exists():
        face_model = p
        break

faces = []
if face_model:
    print(f"Using face model: {face_model.name}")
    try:
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(face_model))
        face_opts = mp_vision.FaceDetectorOptions(
            base_options=base_opts,
            min_detection_confidence=0.3,
        )
        detector = mp_vision.FaceDetector.create_from_options(face_opts)
        results = detector.detect(mp_image)
        if results.detections:
            for det in results.detections:
                bbox = det.bounding_box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2 = x1 + int(bbox.width)
                y2 = y1 + int(bbox.height)
                score = det.categories[0].score
                faces.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": round(score, 4)})
                print(f"  Face: ({x1},{y1}) size={bbox.width}x{bbox.height} conf={score:.3f}")
        else:
            print("  FaceDetector: no detections")
        detector.close()
    except Exception as e:
        print(f"  FaceDetector error: {e}")

# Fallback: ObjectDetector with efficientdet_lite0 (general object detection)
if not faces:
    print("No faces detected or FaceDetector unavailable, trying ObjectDetector...")
    try:
        model_path = MODELS_DIR / "efficientdet_lite0.tflite"
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        obj_opts = mp_vision.ObjectDetectorOptions(
            base_options=base_opts,
            max_results=50,
            score_threshold=0.3,
        )
        detector_obj = mp_vision.ObjectDetector.create_from_options(obj_opts)
        results_obj = detector_obj.detect(mp_image)
        if results_obj.detections:
            people = []
            for det in results_obj.detections:
                # efficientdet_lite0 categories: person is the key one
                cat = det.categories[0]
                if cat.category_name in ("person", "face", "human"):
                    x1, y1 = int(det.bounding_box.origin_x), int(det.bounding_box.origin_y)
                    bw, bh = int(det.bounding_box.width), int(det.bounding_box.height)
                    faces.append({"bbox": [x1, y1, bw, bh], "confidence": round(cat.score, 4),
                                  "category": cat.category_name})
                    print(f"  {cat.category_name}: ({x1},{y1}) size={bw}x{bh} conf={cat.score:.3f}")
            if not faces:
                # Show all detections
                for det in results_obj.detections:
                    cat = det.categories[0]
                    x1, y1 = int(det.bounding_box.origin_x), int(det.bounding_box.origin_y)
                    bw, bh = int(det.bounding_box.width), int(det.bounding_box.height)
                    print(f"  {cat.category_name}: ({x1},{y1}) size={bw}x{bh} conf={cat.score:.3f}")
        detector_obj.close()
    except Exception as e:
        print(f"  ObjectDetector error: {e}")

# ---------------------------------------------------------------------------
# 3. Crop derivation
# ---------------------------------------------------------------------------
if faces:
    all_x1 = min(f["bbox"][0] for f in faces)
    all_y1 = min(f["bbox"][1] for f in faces)
    all_x2 = max(f["bbox"][0] + f["bbox"][2] for f in faces)
    all_y2 = max(f["bbox"][1] + f["bbox"][3] for f in faces)
    union_bbox = [all_x1, all_y1, all_x2 - all_x1, all_y2 - all_y1]
    print(f"Union bbox: {union_bbox}")

    total_w = sum(f["confidence"] ** 2 for f in faces)
    cx = sum((f["bbox"][0] + f["bbox"][2] / 2) * (f["confidence"] ** 2) for f in faces) / total_w
    cy = sum((f["bbox"][1] + f["bbox"][3] / 2) * (f["confidence"] ** 2) for f in faces) / total_w
    cx, cy = int(cx), int(cy)
    print(f"Weighted centroid: ({cx}, {cy})")
else:
    cx, cy = w // 2, h // 2
    union_bbox = None
    print("No detections, falling back to image center")

def crop_for_ratio(img, cx, cy, ratio, target_h=1024):
    target_w = int(target_h * ratio)
    half_w = target_w // 2
    half_h = target_h // 2
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(w, x1 + target_w)
    y2 = min(h, y1 + target_h)
    x1 = max(0, x2 - target_w)
    y1 = max(0, y2 - target_h)
    return img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1, cx, cy)

proposed_crop, proposed_info = crop_for_ratio(img, cx, cy, TARGET_RATIO, TARGET_HEIGHT)
_, center_info = crop_for_ratio(img, w // 2, h // 2, TARGET_RATIO, TARGET_HEIGHT)
center_crop = img[center_info[1]:center_info[1] + center_info[3],
                  center_info[0]:center_info[0] + center_info[2]]

px1, py1, pw, ph, pcx, pcy = proposed_info
cx1, cy1, cw, ch, ccx, ccy = center_info

# Visualization
vis = img.copy()
for face in faces:
    bx, by, bw, bh = face["bbox"]
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 10)
    label = f"{face['confidence']:.2f}"
    cv2.putText(vis, label, (bx, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

if union_bbox:
    ux, uy, ubw, ubh = union_bbox
    cv2.rectangle(vis, (ux, uy), (ux + ubw, uy + ubh), (0, 165, 255), 10)
    cv2.putText(vis, "UNION", (ux, uy + ubh + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)

cv2.rectangle(vis, (px1, py1), (px1 + pw, py1 + ph), (255, 0, 255), 10)
cv2.putText(vis, "PROPOSED 9:16", (px1, py1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
cv2.circle(vis, (pcx, pcy), 15, (255, 0, 255), -1)

cv2.rectangle(vis, (cx1, cy1), (cx1 + cw, cy1 + ch), (0, 255, 255), 10)
cv2.putText(vis, "CENTER 9:16", (cx1, cy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
cv2.circle(vis, (ccx, ccy), 15, (0, 255, 255), -1)

# Save
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

report = {
    "image": str(IMAGE_PATH),
    "image_size": {"width": w, "height": h},
    "target_ratio": TARGET_RATIO,
    "model": "MediaPipe FaceDetector (blaze_face_full_range_sparse.tflite)",
    "detections": faces,
    "union_bbox": union_bbox,
    "crop_results": {
        "proposed": {"centroid": [pcx, pcy], "box": [px1, py1, pw, ph],
                     "method": "weighted detection centroid (score²)"},
        "center_crop": {"centroid": [ccx, ccy], "box": [cx1, cy1, cw, ch],
                        "method": "image center"},
    },
}
with open(OUTPUT_DIR / "report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n--- SPIKE REPORT ---")
print(f"Subjects detected: {len(faces)}")
print(f"Proposed centroid: ({pcx}, {pcy})")
print(f"Center centroid:   ({ccx}, {ccy})")
print(f"Offset: ({pcx - ccx}, {pcy - ccy}) px = ({round((pcx-ccx)/w*100,2)}%, {round((pcy-ccy)/h*100,2)}%)")
print(f"\nOutputs in {OUTPUT_DIR}/")
