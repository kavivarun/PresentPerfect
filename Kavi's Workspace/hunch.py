import cv2
import numpy as np
import mediapipe as mp

# === CONFIGURATION ===
VIDEO_PATH       = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\SampleVideos\Video1.mp4"
WINDOW_NAME      = "Hand Position: Side vs Gesturing"
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# threshold: if a wrist is more than this fraction of frame-width away from hip midpoint → gesturing
GESTURE_RATIO    = 0.25  

# --- initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
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

    # resize and prep
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img)

    status = "HANDS AT SIDE"
    color  = (0,255,0)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        # helper to get pixel coords
        def P(pt): return np.array([pt.x * w, pt.y * h])

        # key points
        lh     = P(lm[mp_pose.PoseLandmark.LEFT_HIP])
        rh     = P(lm[mp_pose.PoseLandmark.RIGHT_HIP])
        lw     = P(lm[mp_pose.PoseLandmark.LEFT_WRIST])
        rw     = P(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
        hip_mid = (lh + rh) * 0.5

        # draw for debug
        cv2.circle(frame, tuple(hip_mid.astype(int)), 5, (255,255,0), -1)
        cv2.circle(frame, tuple(lw.astype(int)),        5, (255,0,0),   -1)
        cv2.circle(frame, tuple(rw.astype(int)),        5, (255,0,0),   -1)

        # measure max distance of wrists from hip midpoint
        dist_l = np.linalg.norm(lw - hip_mid)
        dist_r = np.linalg.norm(rw - hip_mid)
        max_dist = max(dist_l, dist_r)

        # decide
        if max_dist > GESTURE_RATIO * w:
            status, color = "GESTURING", (0,0,255)

    # overlay and show
    cv2.putText(frame, status, (10, 40), FONT, 1.0, color, 2)
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
