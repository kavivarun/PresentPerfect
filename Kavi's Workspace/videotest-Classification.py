# webcam_classify.py
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time

# 1. Load the best-trained weights for CLASSIFICATION
model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\models\bestclassification.pt")  # ← classification model

mp_fd = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.2
)

# Padding percentages
HEAD_PAD_TOP    = 0.25
HEAD_PAD_SIDE   = 0.25
HEAD_PAD_BOTTOM = 0.15

# ------------------------------------------------------------------
# 2. Webcam setup
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ------------------------------------------------------------------
# 3. Main loop
# ------------------------------------------------------------------
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = face_detector.process(rgb)
    annotated = frame.copy()

    if mp_res.detections:
        for det in mp_res.detections:
            bb = det.location_data.relative_bounding_box
            fx1 = int(bb.xmin * W)
            fy1 = int(bb.ymin * H)
            fw  = int(bb.width  * W)
            fh  = int(bb.height * H)
            fx2, fy2 = fx1 + fw, fy1 + fh

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

            cls_res = model.predict(
                head_roi,
                imgsz=640,
                device="cuda:0",
                stream=False,
                conf=0.0
            )[0]

            # extract probabilities correctly
            torch_probs = cls_res.probs.data           # get the raw torch.Tensor
            np_probs    = torch_probs.cpu().numpy()    # to NumPy array
            np_probs    = np_probs.flatten()           # flatten to 1-D

            class_id   = int(np.argmax(np_probs))
            class_name = cls_res.names[class_id]
            class_prob = float(np_probs[class_id])

            # overlay label + box
            label = f"{class_name}: {class_prob:.2f}"
            cv2.putText(annotated, label, (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2),
                        (0, 255, 0), 2)

    # ------------------------------------------------------------------
    # 4. FPS overlay & display
    # ------------------------------------------------------------------
    fps = 1.0 / (time.time() - t0)
    cv2.putText(
        annotated, f"{fps:.1f} FPS", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
    )

    cv2.imshow("YOLOv8 Live – Head Classification", annotated)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

# ------------------------------------------------------------------
# 5. Cleanup
# ------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()