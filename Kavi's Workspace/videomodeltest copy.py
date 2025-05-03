# video_head_classify.py

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time

# ─── Paths ───────────────────────────────────────────────────────────────
VIDEO_PATH  = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\input_video.mp4"
OUTPUT_PATH = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\output_video.mp4"
MODEL_PATH  = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\bestclassification.pt"

# ─── Load classification model ────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ─── Set up MediaPipe face detector ───────────────────────────────────────
mp_fd = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.2
)

# ─── Head-crop padding percentages ────────────────────────────────────────
HEAD_PAD_TOP    = 0.25
HEAD_PAD_SIDE   = 0.25
HEAD_PAD_BOTTOM = 0.15

# ─── Open input video ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

# ─── Fetch source FPS & frame size ───────────────────────────────────────
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Unable to read first frame from input video")
H, W = frame.shape[:2]

# ─── Set up output writer ─────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H), True)

# ─── Process frames ───────────────────────────────────────────────────────
frame_idx = 0
while ret:
    t0 = time.time()
    annotated = frame.copy()

    # 1) detect faces
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = face_detector.process(rgb)

    if mp_res.detections:
        for det in mp_res.detections:
            bb  = det.location_data.relative_bounding_box
            fx1 = int(bb.xmin * W)
            fy1 = int(bb.ymin * H)
            fw  = int(bb.width  * W)
            fh  = int(bb.height * H)
            fx2, fy2 = fx1 + fw, fy1 + fh

            # compute padded head ROI
            pad_top    = int(fh * HEAD_PAD_TOP)
            pad_side   = int(fw * HEAD_PAD_SIDE)
            pad_bottom = int(fh * HEAD_PAD_BOTTOM)

            hx1 = max(0,     fx1 - pad_side)
            hy1 = max(0,     fy1 - pad_top)
            hx2 = min(W - 1, fx2 + pad_side)
            hy2 = min(H - 1, fy2 + pad_bottom)

            head_roi_color = frame[hy1:hy2, hx1:hx2]
            head_roi_gray = cv2.cvtColor(head_roi_color, cv2.COLOR_BGR2GRAY)      # convert to grayscale
            head_roi = cv2.cvtColor(head_roi_gray, cv2.COLOR_GRAY2RGB)    
            if head_roi.size == 0:
                continue

            # 2) classify
            cls_res     = model.predict(head_roi_gray, imgsz=640, device="cuda:0", stream=False, conf=0.0)[0]
            torch_probs = cls_res.probs.data
            np_probs    = torch_probs.cpu().numpy().flatten()
            class_id    = int(np.argmax(np_probs))
            class_name  = cls_res.names[class_id]
            class_prob  = float(np_probs[class_id])

            # 3) annotate
            label = f"{class_name}: {class_prob:.2f}"
            cv2.putText(annotated, label, (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2),
                          (0, 255, 0), 2)

    # 4) FPS overlay
    fps_disp = 1.0 / (time.time() - t0)
    cv2.putText(annotated, f"{fps_disp:.1f} FPS", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # 5) write & advance
    out.write(annotated)
    ret, frame = cap.read()
    frame_idx += 1

# ─── Cleanup ──────────────────────────────────────────────────────────────
cap.release()
out.release()
print(f"Done! Annotated video saved to: {OUTPUT_PATH}")