import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# === CONFIG FLAGS ===
ANNOTATE_VIDEO = True   # Set to False if you don't want to save or draw video
video_path = "walking.mp4"
output_video_path = "walking_output.mp4"

# Initialize MediaPipe Pose for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # (Optional) used for drawing pose landmarks

# Start video capture
cap = cv2.VideoCapture(video_path)

# Get video metadata
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 0

# Only create video writer if annotation is enabled
if ANNOTATE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === Tracking Data ===
positions = []       # (second, normalized_x) only when detected
trail_pixels = []    # (x, y) pixel centers only when detected

# === Process Video Frame-by-Frame ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_sec = int(frame_count / fps)

    if results.pose_landmarks:
        # Draw landmarks if requested
        if ANNOTATE_VIDEO:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Compute center between shoulders
        left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        norm_x = (left.x + right.x) / 2
        positions.append((frame_sec, norm_x))

        center_px = (int(norm_x * width),
                     int((left.y + right.y) / 2 * height))
        trail_pixels.append(center_px)

        if ANNOTATE_VIDEO:
            # Draw the center point
            cv2.circle(frame, center_px, 6, (0, 0, 255), -1)

            # Draw movement trail (blue lines) up to this frame
            for i in range(1, len(trail_pixels)):
                cv2.line(frame, trail_pixels[i-1], trail_pixels[i], (255, 0, 0), 2)

    # Always write out the frame (annotated if detection happened)
    if ANNOTATE_VIDEO:
        out.write(frame)

    frame_count += 1

cap.release()
if ANNOTATE_VIDEO:
    out.release()
    print("✅ Annotated video saved:", output_video_path)

# === Step 3: Aggregate positions per second ===
seconds = int(frame_count / fps)
movement_by_second = []

for sec in range(seconds):
    sec_positions = [p for t, p in positions if t == sec]
    if sec_positions:
        avg_position = sum(sec_positions) / len(sec_positions)
        movement_by_second.append((sec, avg_position))
    else:
        movement_by_second.append((sec, None))

# === Step 4: Output movement data ===
print("\n=== Presenter Movement Summary ===")
for timestamp, pos in movement_by_second:
    if pos is not None:
        print(f"Second {timestamp}: Normalized X Position = {pos:.3f}")
    else:
        print(f"Second {timestamp}: No presenter detected")
