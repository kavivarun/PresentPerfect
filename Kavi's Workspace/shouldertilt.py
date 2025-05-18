import cv2
import numpy as np
import mediapipe as mp
import math

# === CONFIGURATION ===
VIDEO_PATH  = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\SampleVideos\Video1.mp4"
WINDOW_NAME = "Shoulder Straightness Check"
FONT        = cv2.FONT_HERSHEY_SIMPLEX
STRAIGHT_THRESHOLD_DEG = 7.0  # max degrees off horizontal to call "straight"

# --- set up MediaPipe Pose ----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open video file: {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    # Convert to RGB and process with Pose
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Get pixel coords of shoulders
        left = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        p1 = (int(left.x * w), int(left.y * h))
        p2 = (int(right.x * w), int(right.y * h))

        # Draw line between shoulders
        cv2.line(frame, p1, p2, (0,255,0), 2)

        # Compute angle: arctan2(dy, dx)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize angle to [-90, +90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        # Determine straightness
        straight = abs(angle_deg) <= STRAIGHT_THRESHOLD_DEG
        status = "STRAIGHT" if straight else "TILTED"

        # Overlay text
        cv2.putText(frame, f"Shoulder tilt: {angle_deg:+.1f} deg", (10, 30), FONT, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Status: {status}",               (10, 60), FONT, 0.7, (0,255,255) if straight else (0,0,255), 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
