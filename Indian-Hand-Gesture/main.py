import os
import sys
import time
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIG  (tune these)
# ============================================================

CAM_INDEX = 0
MIRROR = True            # must match how the training data was captured
DETECT_CONF = 0.6
SMOOTH_FRAMES = 7

ML_HIGH = 0.80           # ML wins outright above this
ML_MIN = 0.60            # ML accepted (with margin) only if no rule matched
MIN_MARGIN = 0.15        # gap between top-1 and top-2 probability

FINGER_RATIO = 1.10
THUMB_RATIO = 1.10
OK_RATIO = 0.35
PINCH_RATIO = 0.30

TTA_SAMPLES = 9          # predictions averaged per frame (1 = off)
TTA_ROT_DEG = 5          # small random rotation used for test-time augmentation
TTA_NOISE = 0.003        # small landmark jitter
MIN_HANDEDNESS = 0.70    # MediaPipe hand-detection score required
MIN_HAND_SIZE = 0.04     # hand too far from camera below this
EDGE = 0.01              # landmarks closer to the frame edge than this = cut off
LOCK_VOTES = 5           # votes (out of SMOOTH_FRAMES) needed to change the shown label


# ============================================================
# PATHS + LOAD
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_model.pkl")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

print("=" * 60)
print("       INDIAN HAND GESTURE RECOGNITION")
print("=" * 60)

for path in (MODEL_PATH, HAND_MODEL_PATH):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {os.path.basename(path)} not found")

model = joblib.load(MODEL_PATH)
CLASSES = model.classes_
FEATURE_COUNT = 63 * 2 + 1

n_expected = getattr(model, "n_features_in_", None)
if n_expected is not None and n_expected != FEATURE_COUNT:
    sys.exit(f"ERROR: model expects {n_expected} features, script gives {FEATURE_COUNT}.")

print(f"Model loaded ({len(CLASSES)} classes).")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

detector = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=DETECT_CONF,
        min_hand_presence_confidence=DETECT_CONF,
        min_tracking_confidence=DETECT_CONF,
    )
)
print("MediaPipe loaded.")


# ============================================================
# FEATURES (same as original, so the existing model still works)
# + test-time augmentation for steadier predictions
# ============================================================

RNG = np.random.default_rng(0)


def to_points(hand):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def normalize_points(points):
    points = points - points[0]
    max_distance = np.max(np.linalg.norm(points, axis=1))
    if max_distance > 0:
        points = points / max_distance
    return points.flatten()


def build_feature(point_sets):
    h0 = normalize_points(point_sets[0])
    h1 = normalize_points(point_sets[1]) if len(point_sets) >= 2 else np.zeros(63, np.float32)
    return np.concatenate([h0, h1, [min(len(point_sets), 2)]])


def perturb(point_sets):
    angle = np.deg2rad(RNG.uniform(-TTA_ROT_DEG, TTA_ROT_DEG))
    scale = RNG.uniform(0.95, 1.05)
    c, s_ = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s_], [s_, c]], dtype=np.float32)
    centre = np.concatenate(point_sets)[:, :2].mean(axis=0)

    out = []
    for p in point_sets:
        q = p.copy()
        q[:, :2] = ((q[:, :2] - centre) @ rot.T) * scale + centre
        q[:, 2] *= scale
        q += RNG.normal(0, TTA_NOISE, q.shape).astype(np.float32)
        out.append(q)
    return out


def predict_tta(hands):
    base = [to_points(h) for h in hands]
    batch = [build_feature(base)]
    for _ in range(TTA_SAMPLES - 1):
        batch.append(build_feature(perturb(base)))
    return model.predict_proba(np.array(batch, dtype=np.float32)).mean(axis=0)


# ============================================================
# RULE-BASED GESTURES (rotation-tolerant)
# ============================================================

def dist(a, b):
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))


def finger_up(hand, tip, pip):
    return dist(hand[0], hand[tip]) > dist(hand[0], hand[pip]) * FINGER_RATIO


def thumb_out(hand):
    return dist(hand[4], hand[17]) > dist(hand[3], hand[17]) * THUMB_RATIO


def recognize_rule(hand):
    idx = finger_up(hand, 8, 6)
    mid = finger_up(hand, 12, 10)
    rng = finger_up(hand, 16, 14)
    pnk = finger_up(hand, 20, 18)
    thumb = thumb_out(hand)

    size = max(dist(hand[0], hand[9]), 1e-6)
    pinch = dist(hand[4], hand[8]) / size

    if pinch < OK_RATIO and mid and rng and pnk:
        return "OK"
    if pinch < PINCH_RATIO and not (mid or rng or pnk):
        return "PINCH"

    if (thumb and not (idx or mid or rng or pnk)
            and hand[4].y < hand[3].y < hand[2].y
            and hand[4].y < hand[5].y):
        return "THUMBS_UP"

    if idx and mid and rng and pnk:
        return "FIVE" if thumb else "FOUR"
    if idx and mid and rng:
        return "THREE"
    if idx and mid:
        return "TWO"
    if idx and not (mid or rng or pnk):
        horizontal = abs(hand[8].x - hand[5].x) > abs(hand[8].y - hand[5].y)
        return "POINTING" if horizontal else "ONE"

    return None


# ============================================================
# DECISION LOGIC
# ============================================================

def hand_ok(hand):
    if dist(hand[0], hand[9]) < MIN_HAND_SIZE:
        return False
    return all(EDGE <= lm.x <= 1 - EDGE and EDGE <= lm.y <= 1 - EDGE for lm in hand)


def decide(probs, rule):
    order = np.argsort(probs)[::-1]
    top = float(probs[order[0]])
    second = float(probs[order[1]]) if len(order) > 1 else 0.0
    label = str(CLASSES[order[0]])

    if top >= ML_HIGH:
        return label, top, "ML"
    if rule is not None:
        return rule, top, "Rule"
    if top >= ML_MIN and (top - second) >= MIN_MARGIN:
        return label, top, "ML"
    return "UNKNOWN", top, "-"


# ============================================================
# DRAWING
# ============================================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hand(frame, hand, number):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 3)

    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 9, (255, 255, 255), 2)
        cv2.putText(frame, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    cv2.putText(frame, f"Hand {number}", (pts[0][0] + 15, pts[0][1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def put(frame, text, pos, scale, color, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


# ============================================================
# CAMERA + MAIN LOOP
# ============================================================

camera = cv2.VideoCapture(CAM_INDEX)
if not camera.isOpened():
    detector.close()
    sys.exit("ERROR: Camera could not be opened.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prob_history = deque(maxlen=SMOOTH_FRAMES)
label_history = deque(maxlen=SMOOTH_FRAMES)
last_hand_count = 0
displayed_label = None

t0 = time.monotonic()
last_ts = -1
prev_time = time.monotonic()

print("\nCamera started. Press Q to quit.\n")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera frame error.")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        now = time.monotonic()
        fps = 1.0 / max(now - prev_time, 0.001)
        prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((now - t0) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        result = detector.detect_for_video(mp_image, ts)
        pairs = sorted(zip(result.hand_landmarks, result.handedness),
                       key=lambda p: p[0][0].x)
        hands = [p[0] for p in pairs]
        scores = [p[1][0].score for p in pairs]

        def reset_state():
            global last_hand_count, displayed_label
            prob_history.clear()
            label_history.clear()
            displayed_label = None
            last_hand_count = 0

        if not hands:
            reset_state()
            put(frame, "NO HAND DETECTED", (30, 60), 1, (0, 0, 255), 3)
        else:
            for number, hand in enumerate(hands, start=1):
                draw_hand(frame, hand, number)

            good = all(hand_ok(h) and sc >= MIN_HANDEDNESS
                       for h, sc in zip(hands, scores))

            if not good:
                reset_state()
                put(frame, "MOVE HAND FULLY INTO VIEW", (30, 60), 0.9, (0, 165, 255), 3)
            else:
                if len(hands) != last_hand_count:
                    prob_history.clear()
                    label_history.clear()
                    last_hand_count = len(hands)

                rule = recognize_rule(hands[0]) if len(hands) == 1 else None

                prob_history.append(predict_tta(hands))
                avg_probs = np.mean(prob_history, axis=0)

                label, conf, source = decide(avg_probs, rule)
                label_history.append(label)

                top_label, top_count = Counter(label_history).most_common(1)[0]
                if displayed_label is None or top_count >= LOCK_VOTES:
                    displayed_label = top_label

                cv2.rectangle(frame, (15, 15), (570, 145), (20, 20, 20), -1)
                put(frame, "GESTURE", (35, 48), 0.55, (200, 200, 200))
                put(frame, displayed_label, (35, 90), 1.15, (0, 255, 0), 3)
                put(frame, f"Confidence: {conf * 100:.1f}%  [{source}]",
                    (35, 120), 0.55, (255, 255, 255))
                put(frame, f"Hands: {len(hands)}", (350, 48), 0.55, (255, 255, 255))
                put(frame, f"FPS: {fps:.1f}", (350, 80), 0.55, (255, 255, 255))

        cv2.rectangle(frame, (0, frame.shape[0] - 45),
                      (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        put(frame, "Q = Quit", (20, frame.shape[0] - 15), 0.55, (255, 255, 255))

        cv2.imshow("Indian Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nProgram closed.")