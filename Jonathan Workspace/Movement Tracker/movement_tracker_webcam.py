import cv2
import mediapipe as mp

# === CONFIG FLAGS ===
ANNOTATE_VIDEO      = True   # Set to False if you don't want to draw video
SAVE_VIDEO          = False  # Set to True if you want to save video
OUTPUT_VIDEO_PATH   = "webcam_output.mp4"

# Initialize MediaPipe Pose for pose detection
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to access the webcam")

# Get capture metadata (fallback to 30 FPS if webcam doesn't report one)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 0

# Only create video writer if saving is enabled
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# === Tracking Data ===
positions     = []  # list of (second, normalized_x)
trail_pixels  = []  # list of (px, py)

# === Process webcam frames ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(frame_rgb)
    sec       = int(frame_count / fps)

    if results.pose_landmarks:
        # Draw pose landmarks if requested
        if ANNOTATE_VIDEO:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Compute center between shoulders
        lm_left  = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lm_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        norm_x   = (lm_left.x + lm_right.x) / 2
        positions.append((sec, norm_x))

        center_px = (
            int(norm_x * width),
            int(((lm_left.y + lm_right.y) / 2) * height)
        )
        trail_pixels.append(center_px)

        if ANNOTATE_VIDEO:
            # Draw center point
            cv2.circle(frame, center_px, 6, (0, 0, 255), -1)
            # Draw movement trail
            for i in range(1, len(trail_pixels)):
                cv2.line(frame, trail_pixels[i-1], trail_pixels[i], (255, 0, 0), 2)

    # Write (possibly annotated) frame to output
    if SAVE_VIDEO:
        out.write(frame)

    cv2.imshow("Webcam Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Cleanup
cap.release()
if SAVE_VIDEO:
    out.release()
    print("✅ Annotated webcam video saved to:", OUTPUT_VIDEO_PATH)
cv2.destroyAllWindows()

# === Aggregate movement by second ===
seconds = int(frame_count / fps)
movement_by_second = []
for s in range(seconds):
    pts = [x for t, x in positions if t == s]
    if pts:
        movement_by_second.append((s, sum(pts)/len(pts)))
    else:
        movement_by_second.append((s, None))

# === Print summary ===
print("\n=== Presenter Movement Summary ===")
for s, pos in movement_by_second:
    if pos is not None:
        print(f"Second {s}: Normalized X Position = {pos:.3f}")
    else:
        print(f"Second {s}: No presenter detected")
