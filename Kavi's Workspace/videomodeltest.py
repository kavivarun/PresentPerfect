# video_detect.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time
import os
from collections import defaultdict, Counter

frame_index = 0
class_per_second = defaultdict(list)  # stores class labels per second

# ----------- Configuration -----------
VIDEO_PATH = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\input_video.mp4"
OUTPUT_PATH = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\output_video.mp4"
MODEL_PATH = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\detect\train6\weights\best.pt"
DEVICE = "cuda:0"  # or "cpu" if CUDA not available

# ----------- Load Model --------------
model = YOLO(MODEL_PATH)

mp_fd = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.2)

HEAD_PAD_TOP = 0.25
HEAD_PAD_SIDE = 0.25
HEAD_PAD_BOTTOM = 0.15

# ----------- Load Video --------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

# ----------- Process Loop ------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    second = int(frame_index / FPS)
    frame_index += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = face_detector.process(rgb)
    annotated = frame.copy()

    if mp_res.detections:
        for det in mp_res.detections:
            bb = det.location_data.relative_bounding_box
            fx1 = int(bb.xmin * W)
            fy1 = int(bb.ymin * H)
            fw = int(bb.width * W)
            fh = int(bb.height * H)
            fx2 = fx1 + fw
            fy2 = fy1 + fh

            pad_top = int(fh * HEAD_PAD_TOP)
            pad_side = int(fw * HEAD_PAD_SIDE)
            pad_bottom = int(fh * HEAD_PAD_BOTTOM)

            hx1 = max(0, fx1 - pad_side)
            hy1 = max(0, fy1 - pad_top)
            hx2 = min(W - 1, fx2 + pad_side)
            hy2 = min(H - 1, fy2 + pad_bottom)

            head_roi = frame[hy1:hy2, hx1:hx2]
            if head_roi.size == 0:
                continue

            yolo_res = model.predict(
                head_roi,
                imgsz=640,
                conf=0.25,
                device=DEVICE,
                stream=False,
                verbose=False
            )

            preds = yolo_res[0].boxes
            if preds is not None and preds.cls.numel() > 0:
                for cls_tensor in preds.cls:
                    class_id = int(cls_tensor.item())
                    class_name = model.names[class_id]
                    class_per_second[second].append(class_name)

            ann_head = yolo_res[0].plot()

            if ann_head.shape[:2] != head_roi.shape[:2]:
                ann_head = cv2.resize(ann_head, (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_LINEAR)

            annotated[hy1:hy2, hx1:hx2] = ann_head

    out.write(annotated)

# ----------- Summary Output ------------
dominant_classes = {
    sec: Counter(classes).most_common(1)[0][0]
    for sec, classes in class_per_second.items()
}

print("Most common class per second:")
for sec, cls in dominant_classes.items():
    print(f"Second {sec}: {cls}")
    
# ----------- Cleanup -----------------
cap.release()
out.release()
print(f"Video saved to: {OUTPUT_PATH}")
