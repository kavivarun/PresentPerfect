# ──────────────────────────────  Imports & Monkey-patch  ──────────────────────────────
import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS

import os, time, tempfile, random, queue, threading, math
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, wait

import cv2
import mediapipe as mp
import numpy as np
import torch, torchvision
from ultralytics import YOLO

torch.backends.cudnn.benchmark = True

# ──────────────────────────────  Flask / Socket.IO  ────────────────────────────────
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ───────────────────────────────  Models & Consts  ─────────────────────────────────
MODEL_PATH   = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\runs\detect\train6\weights\best.pt"
emotion_model = YOLO(MODEL_PATH)
emotion_model.half()

mp_fd         = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.2)

HEAD_PAD_TOP, HEAD_PAD_SIDE, HEAD_PAD_BOTTOM = 0.25, 0.25, 0.15
DEVICE  = "cuda:0"

mp_pose  = mp.solutions.pose
pose     = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

mp_face  = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

G_MODEL_POINTS = np.array([
    (0.0,   0.0,   0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3,  32.7, -26.0),
    (-28.9,-28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

LANDMARK_IDS = dict(
    nose_tip=1, chin=199, left_eye_outer=33, right_eye_outer=263,
    mouth_left=61, mouth_right=291
)

# Posture-tracking thresholds (degrees)
SIDE_THR     = 10.0
FORWARD_THR  = 10.0
YAW_THR, VERT_THR = 20, 140

BATCH              = 1
NUM_WORKERS        = os.cpu_count()
QUEUE_SIZE         = BATCH * 32          # pending frames waiting for batching

# ───────────────────────────────  Per-request State  ───────────────────────────────
frame_index          = 0
class_per_second     = defaultdict(list)
gaze_per_second      = defaultdict(list)
movement_per_second  = defaultdict(list)
posture_per_second   = defaultdict(list)
state_lock           = threading.Lock()

# progress based on analysed frames
processed_frames     = 0
processed_lock       = threading.Lock()

FUN_MESSAGES = [
    "Detecting awkward smiles... yup, that one's forced. 😬",
    "Analyzing eye contact... or lack thereof 👀",
    "Checking if you're making strong points... or just strong gestures 💪",
    "Scanning for power poses... channel your inner TED talk 🧍‍♂️✨",
    "Detecting fidget level: approaching squirrel mode 🐿️",
    "Evaluating if your arms know what they’re doing 🙆",
    "Measuring your confidence by chin height 📏",
    "Is that a dramatic pause or a freeze? 🫠",
    "Posture alert: spine looking suspiciously like a question mark ❓",
    "Analyzing facial expressions... current emotion: existential dread 🫣",
    "Calculating presentation vibes... please wait... ☕",
    "Your body language is currently buffering... 🔄",
    "Optimizing your charisma algorithm... hang tight 🧠",
    "Face detected... now figuring out what it's trying to say 🕵️",
    "Detecting stance: 50% leader, 50% about-to-run 🏃‍♂️💼",
    "Applying motivational filter: 'You got this!' 🌟",
    "Smile check: 1 detected... was that sarcastic? 🤔",
    "Analyzing stage presence... charisma.exe launching 🚀"
]

# ─────────────────────  Utility helpers  ───────────────────────────────────────────
def reset_state():
    global frame_index, processed_frames
    with state_lock:
        frame_index = 0
        class_per_second.clear()
        gaze_per_second.clear()
        movement_per_second.clear()
        posture_per_second.clear()
    with processed_lock:
        processed_frames = 0

def get_random_message(last_change, interval=10):
    now = time.time()
    if now - last_change >= interval:
        return random.choice(FUN_MESSAGES), now
    return None, last_change

def get_direction(yaw, pitch):
    if yaw >  YAW_THR:
        h = "right"
    elif yaw < -YAW_THR:
        h = "left"
    else:
        h = ""
    if 180 >= pitch >= 160 or -180 <= pitch <= -160:
        v = ""
    else:
        v = "down" if pitch < 0 else "up"
    if not h and not v:
        return "straight"
    if not v:
        return h
    if not h:
        return v
    return f"{v}-{h}"

# ─────────────────────  Batch-aware detector functions  ────────────────────────────
def emotion_batch(batch_frames, W, H, batch_secs):
    heads, sec_idx = [], []
    for i, frame in enumerate(batch_frames):
        mp_res = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not mp_res.detections:
            continue
        for det in mp_res.detections:
            bb = det.location_data.relative_bounding_box
            fx1, fy1 = int(bb.xmin * W), int(bb.ymin * H)
            fw,  fh  = int(bb.width * W), int(bb.height * H)
            fx2, fy2 = fx1 + fw, fy1 + fh
            pad_t, pad_s, pad_b = int(fh*HEAD_PAD_TOP), int(fw*HEAD_PAD_SIDE), int(fh*HEAD_PAD_BOTTOM)
            hx1, hy1 = max(0, fx1 - pad_s), max(0, fy1 - pad_t)
            hx2, hy2 = min(W-1, fx2 + pad_s), min(H-1, fy2 + pad_b)
            roi = batch_frames[i][hy1:hy2, hx1:hx2]
            if roi.size == 0:
                continue
            heads.append(roi)
            sec_idx.append(batch_secs[i])

    if not heads:
        return
    results = emotion_model.predict(
        heads, imgsz=640, conf=0.25, device=DEVICE,
        stream=False, verbose=False, half=True
    )
    for det_res, sec in zip(results, sec_idx):
        preds = det_res.boxes
        if preds is not None and preds.cls.numel() > 0:
            for cls_tensor in preds.cls:
                class_per_second[sec].append(emotion_model.names[int(cls_tensor.item())])

def movement_batch(batch_rgbs, batch_secs):
    for img_rgb, sec in zip(batch_rgbs, batch_secs):
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            continue
        lm = res.pose_landmarks.landmark
        try:
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hp = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hp = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        except IndexError:
            continue

        # movement: midpoint of shoulders, normalized [0–1]
        mid_x = (l_sh.x + r_sh.x) / 2.0

        # quantize into 1–10 bins
        # floor(mid_x*10) gives 0–9, +1 gives 1–10, clamp at 10
        bin_idx = min(10, math.floor(mid_x * 10) + 1)
        movement_per_second[sec].append(bin_idx)

        # posture
        mid_sh = ((l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2)
        mid_hp = ((l_hp.x + r_hp.x) / 2, (l_hp.y + r_hp.y) / 2)

        vec_xy     = (mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
        angle_side = abs(math.degrees(math.atan2(vec_xy[0], vec_xy[1])))

        sh_z       = (l_sh.z + r_sh.z) / 2
        hp_z       = (l_hp.z + r_hp.z) / 2
        vec_y      = mid_sh[1] - mid_hp[1]
        vec_z      = sh_z - hp_z
        angle_fwd  = abs(math.degrees(math.atan2(vec_z, vec_y)))

        posture_per_second[sec].append((angle_side, angle_fwd))

def gaze_batch(batch_rgbs, batch_secs, CAM_MAT, DIST, W, H):
    for img_rgb, sec in zip(batch_rgbs, batch_secs):
        res = face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0]
        pts2d = np.array([
            (lm.landmark[LANDMARK_IDS["nose_tip"]].x  * W,
             lm.landmark[LANDMARK_IDS["nose_tip"]].y  * H),
            (lm.landmark[LANDMARK_IDS["chin"]].x       * W,
             lm.landmark[LANDMARK_IDS["chin"]].y       * H),
            (lm.landmark[LANDMARK_IDS["left_eye_outer"]].x  * W,
             lm.landmark[LANDMARK_IDS["left_eye_outer"]].y  * H),
            (lm.landmark[LANDMARK_IDS["right_eye_outer"]].x * W,
             lm.landmark[LANDMARK_IDS["right_eye_outer"]].y * H),
            (lm.landmark[LANDMARK_IDS["mouth_left"]].x  * W,
             lm.landmark[LANDMARK_IDS["mouth_left"]].y  * H),
            (lm.landmark[LANDMARK_IDS["mouth_right"]].x * W,
             lm.landmark[LANDMARK_IDS["mouth_right"]].y * H)
        ], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(G_MODEL_POINTS, pts2d, CAM_MAT, DIST, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        yaw, pitch = angles[1], angles[0]
        gaze_per_second[sec].append(get_direction(yaw, pitch))

# ──────────────────────────────  Flask route  ──────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    reset_state()
    video = request.files['video']
    if not video:
        return {'error': 'No video uploaded'}, 400

    temp_path = os.path.join(tempfile.gettempdir(), video.filename)
    video.save(temp_path)
    print(f"[INFO] Uploaded video: {video.filename} ({os.path.getsize(temp_path) / 1024:.2f} KB)")

    socketio.start_background_task(process_video, temp_path)
    return {'status': 'processing started'}

# ───────────────────────────  Core processing  ────────────────────────────────────
def process_video(temp_path):
    global frame_index
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_to_process = (total_frames + 7) // 8   # every 4th frame

    focal  = W
    center = (W / 2, H / 2)
    CAM_MAT = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], np.float64)
    DIST = np.zeros((4, 1))

    q        = queue.Queue(maxsize=QUEUE_SIZE)
    SENTINEL = object()

    last_emit_time = time.time()
    emit_interval  = 1.0
    last_msg_time  = time.time()
    current_msg    = random.choice(FUN_MESSAGES)

    # Producer – reads frames and pushes every 4th into queue
    def reader():
        global frame_index
        idx = 0
        nonlocal last_emit_time, last_msg_time, current_msg
        while True:
            ok, frame = cap.read()
            if not ok:
                for _ in range(NUM_WORKERS):
                    q.put(SENTINEL)
                break

            # sample every 8th frame
            if idx % 8 == 0:
                q.put((idx, frame))

            idx += 1
            with state_lock:
                frame_index += 1   # still useful if you log raw read progress

            # fun message throttling
            new_msg, last_msg_time = get_random_message(last_msg_time, 5)
            if new_msg:
                current_msg = new_msg

            time.sleep(0)  # yield
    threading.Thread(target=reader, daemon=True).start()

    # Consumer – pulls frames, builds batches, runs detectors
    def consumer():
        batch_frames, batch_secs, batch_rgbs = [], [], []
        while True:
            item = q.get()
            if item is SENTINEL:
                if batch_frames:
                    run_batch(batch_frames, batch_secs, batch_rgbs)
                q.task_done()
                break
            idx, frame = item
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            batch_secs.append(int(idx / FPS))
            batch_rgbs.append(frame_rgb)
            q.task_done()
            if len(batch_frames) == BATCH:
                run_batch(batch_frames, batch_secs, batch_rgbs)
                batch_frames, batch_secs, batch_rgbs = [], [], []

    pool = ThreadPoolExecutor(max_workers=8)

    def run_batch(frames, secs, rgbs):
        nonlocal last_emit_time
        futs = [
            pool.submit(emotion_batch,  frames, W, H, secs),
            pool.submit(gaze_batch,     rgbs, secs, CAM_MAT, DIST, W, H),
            pool.submit(movement_batch, rgbs, secs)
        ]
        wait(futs)

        # update progress based on frames analysed in this batch
        with processed_lock:
            global processed_frames
            processed_frames += len(frames)
            pct_done = int(processed_frames / expected_to_process * 100)

        now = time.time()
        if now - last_emit_time >= emit_interval:
            last_emit_time = now
            socketio.emit('processing-update',
                          {'message': current_msg, 'progress': pct_done})
            socketio.sleep(0)

    workers = [threading.Thread(target=consumer, daemon=True) for _ in range(NUM_WORKERS)]
    for t in workers:
        t.start()
    q.join()
    for t in workers:
        t.join()

    # ── Aggregation / final emit ───────────────────────────────────
    dom_emotion = {s: Counter(v).most_common(1)[0][0] for s, v in class_per_second.items()}
    dom_gaze    = {s: Counter(v).most_common(1)[0][0] for s, v in gaze_per_second.items()}
    move_avg    = {s: sum(xs) / len(xs) for s, xs in movement_per_second.items()}

    tilt_avg  = {s: sum(a for a, _ in vals) / len(vals) for s, vals in posture_per_second.items()}
    hunch_avg = {s: sum(b for _, b in vals) / len(vals) for s, vals in posture_per_second.items()}

    # print("Most common class per second:"); [print(f"  Sec {s}: {c}") for s, c in dom_emotion.items()]
    # print("Most common gaze per second:");  [print(f"  Sec {s}: {c}") for s, c in dom_gaze.items()]
    # print("Avg horizontal position per second (0-left, 1-right):")
    # [print(f"  Sec {s}: {move_avg[s]:.3f}") for s in sorted(move_avg)]
    # print(f"Avg side tilt per second (> {SIDE_THR}° indicates lean):")
    # [print(f"  Sec {s}: {tilt_avg[s]:.1f}°") for s in sorted(tilt_avg)]
    # print(f"Avg forward hunch per second (> {FORWARD_THR}° indicates hunch):")
    # [print(f"  Sec {s}: {hunch_avg[s]:.1f}°") for s in sorted(hunch_avg)]
    payload = {
        'emotion':    dom_emotion,
        'gaze':       dom_gaze,
        'horizontal': move_avg,
        'tilt':       tilt_avg,
        'hunch':      hunch_avg,
    }
    socketio.emit('processing-complete',
                  {'message': 'Processing done!', 'progress': '100','data': payload})

# ───────────────────────────────  Main  ───────────────────────────────────────────
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, debug=True)
