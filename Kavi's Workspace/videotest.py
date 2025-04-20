# webcam_detect.py

import cv2
from ultralytics import YOLO

# 1. Load the best‑trained weights
model = YOLO(r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\detect\train6\weights\best.pt")

# 2. Open the default camera (0). Change to 1,2… for other cameras.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 3. Loop over frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Run detection on this frame
    results = model.predict(  # or model(frame) for detect()
        frame, 
        stream=False,       # single‑frame inference
        imgsz=640,          # resize for speed/accuracy trade‑off
        conf=0.25,          # confidence threshold
        device="cuda:0"     # or "cpu"
    )

    # 5. Overlay boxes on the frame
    annotated = results[0].plot()  # returns a NumPy array with boxes drawn

    # 6. Display
    cv2.imshow("YOLOv8 Live", annotated)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # ESC or 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()