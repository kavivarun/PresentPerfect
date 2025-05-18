import cv2
import numpy as np
import mediapipe as mp

# === CONFIGURATION ===
CAMERA_ID       = 0
WINDOW_NAME     = "Live Head Pose Estimation"
ANNOTATE_VIDEO  = False  # Not saving video, just live view
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# --- DIRECTION THRESHOLDS ---
YAW_THR  = 20    # degrees: left/right threshold
VERT_THR = 140   # degrees: vertical “straight” around 180↔-180
# ----------------------------

# 3D model points for head pose (in mm)
MODEL_POINTS = np.array([
    (0.0,  0.0,   0.0),   # Nose tip
    (0.0, -63.6, -12.5),  # Chin
    (-43.3, 32.7, -26.0), # Left eye corner
    (43.3,  32.7, -26.0), # Right eye corner
    (-28.9,-28.9, -24.1), # Left mouth corner
    (28.9, -28.9, -24.1)  # Right mouth corner
], dtype=np.float64)

# Corresponding landmark indices in MediaPipe Face Mesh
LANDMARK_IDS = {
    "nose_tip":       1,
    "chin":          199,
    "left_eye_outer": 33,
    "right_eye_outer":263,
    "mouth_left":    61,
    "mouth_right":   291
}

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper to bucket yaw/pitch into one of 9 directions
def get_direction(yaw, pitch):
    # horizontal
    if yaw >  YAW_THR:   h = "right"
    elif yaw < -YAW_THR: h = "left"
    else:                h = ""
    # vertical: "straight" zone is pitch ≥+140 or ≤-140 (wraps at 180↔-180)
    if 180 >= pitch >= 155 or -180 <= pitch <= -150:
        v = ""
    else:
        # inside the non‑straight zone → decide up/down by sign
        v = "down" if pitch < 0 else "up"
    # combine
    if not h and not v: return "straight"
    if not v:           return h
    if not h:           return v
    return f"{v}-{h}"    # e.g. "up-left", "down-right"

# Start video capture
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Unable to access the camera")
h, w, _ = frame.shape

# Camera intrinsics (approximate)
focal_length = w
center = (w/2, h/2)
CAMERA_MATRIX = np.array([
    [focal_length, 0,           center[0]],
    [0,            focal_length,center[1]],
    [0,            0,           1        ]
], dtype=np.float64)
DIST_COEFFS = np.zeros((4,1))  # Assume no lens distortion

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]

        # 2D image points
        image_points = np.array([
            (lm.landmark[LANDMARK_IDS["nose_tip"]     ].x * w,
             lm.landmark[LANDMARK_IDS["nose_tip"]     ].y * h),
            (lm.landmark[LANDMARK_IDS["chin"]         ].x * w,
             lm.landmark[LANDMARK_IDS["chin"]         ].y * h),
            (lm.landmark[LANDMARK_IDS["left_eye_outer"]].x * w,
             lm.landmark[LANDMARK_IDS["left_eye_outer"]].y * h),
            (lm.landmark[LANDMARK_IDS["right_eye_outer"]].x * w,
             lm.landmark[LANDMARK_IDS["right_eye_outer"]].y * h),
            (lm.landmark[LANDMARK_IDS["mouth_left"]   ].x * w,
             lm.landmark[LANDMARK_IDS["mouth_left"]   ].y * h),
            (lm.landmark[LANDMARK_IDS["mouth_right"]  ].x * w,
             lm.landmark[LANDMARK_IDS["mouth_right"]  ].y * h)
        ], dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS, image_points,
            CAMERA_MATRIX, DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            # Draw head direction line
            nose_end_3D = np.array([(0.0, 0.0, 1000.0)])
            nose_end_2D, _ = cv2.projectPoints(
                nose_end_3D, rotation_vec, translation_vec,
                CAMERA_MATRIX, DIST_COEFFS
            )
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_2D[0][0][0]), int(nose_end_2D[0][0][1]))
            cv2.arrowedLine(frame, p1, p2, (0,255,0), 2, tipLength=0.2)

            # Euler angles from rotation matrix
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch, roll = angles[1], angles[0], angles[2]

            # Get text label
            direction = get_direction(yaw, pitch)

            # Overlay angles + direction
            cv2.putText(frame, f"YAW:   {yaw:.1f}",   (10, 30), FONT, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"PITCH: {pitch:.1f}", (10, 60), FONT, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"ROLL:  {roll:.1f}",  (10, 90), FONT, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"DIRECTION: {direction}", (10, 120), FONT, 0.7, (0,255,255), 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
