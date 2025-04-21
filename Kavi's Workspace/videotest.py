# webcam_detect.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time
# 1. Load the best‑trained weights
model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\detect\train6\weights\best.pt")

mp_fd = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(
    model_selection=1,              # full‑range model handles large head pose
    min_detection_confidence=0.2
)
 
# Padding percentages (relative to the detected face‑box width / height)
HEAD_PAD_TOP    = 0.25   # extra space above the face  (hair / forehead)
HEAD_PAD_SIDE   = 0.25  # left + right expansion      (ears / turned head)
HEAD_PAD_BOTTOM = 0.15   # extra below the chin / neck
 
# ------------------------------------------------------------------
# 2.  Webcam setup
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
 
# ------------------------------------------------------------------
# 3.  Main loop
# ------------------------------------------------------------------
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break
 
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = face_detector.process(rgb)
 
    # Work on a copy for nice display
    annotated = frame.copy()
 
    if mp_res.detections:
        for det in mp_res.detections:
            bb = det.location_data.relative_bounding_box
            # Face‑box in pixels
            fx1 = int(bb.xmin * W)
            fy1 = int(bb.ymin * H)
            fw  = int(bb.width  * W)
            fh  = int(bb.height * H)
            fx2 = fx1 + fw
            fy2 = fy1 + fh
 
            # ------------------------------------------------------
            # Expand to a "head box" with asymmetric padding
            # ------------------------------------------------------
            pad_top    = int(fh * HEAD_PAD_TOP)
            pad_side   = int(fw * HEAD_PAD_SIDE)
            pad_bottom = int(fh * HEAD_PAD_BOTTOM)
 
            hx1 = max(0,     fx1 - pad_side)
            hy1 = max(0,     fy1 - pad_top)
            hx2 = min(W - 1, fx2 + pad_side)
            hy2 = min(H - 1, fy2 + pad_bottom)
 
            head_roi = frame[hy1:hy2, hx1:hx2]
            if head_roi.size == 0:
                continue
 
            # ------------------------------------------------------
            # YOLO on the *entire head* crop
            # ------------------------------------------------------
            yolo_res = model.predict(
                head_roi,
                imgsz=640,
                conf=0.25,
                device="cuda:0",
                stream=False
            )
            ann_head = yolo_res[0].plot()
 
            # Resize if YOLO outputs a different shape
            if ann_head.shape[:2] != head_roi.shape[:2]:
                ann_head = cv2.resize(
                    ann_head,
                    (hx2 - hx1, hy2 - hy1),
                    interpolation=cv2.INTER_LINEAR
                )
 
            annotated[hy1:hy2, hx1:hx2] = ann_head
 
            # Optional outline around the head crop so you can see it
            cv2.rectangle(
                annotated, (hx1, hy1), (hx2, hy2),
                (0, 255, 0), 2
            )
 
    # ------------------------------------------------------------------
    # 4.  FPS overlay & display
    # ------------------------------------------------------------------
    fps = 1.0 / (time.time() - t0)
    cv2.putText(annotated, f"{fps:.1f} FPS", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
 
    cv2.imshow("YOLOv8 Live – Head‑cropped", annotated)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):   # ESC / q to quit
        break
 
# ------------------------------------------------------------------
# 5.  Cleanup
# ------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()