
import os
import base64
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp

from flask import Flask, jsonify, render_template, request

# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "isl_model.pkl")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("isl_model.pkl not found!")

if not os.path.exists(HAND_MODEL_PATH):
    raise FileNotFoundError("hand_landmarker.task not found!")

# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

print("Loading Indian Hand Gesture Recognition Model...")

model = joblib.load(MODEL_PATH)
CLASSES = model.classes_

if getattr(model, "n_features_in_", 127) != 127:
    raise ValueError("Expected a two-hand model with 127 features.")

print("ML model loaded successfully!")

# =====================================================
# 3. MEDIAPIPE INITIALIZATION
# =====================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = HandLandmarker.create_from_options(options)

# =====================================================
# 4. FLASK APPLICATION
# =====================================================

app = Flask(__name__)

history = deque(maxlen=5)
rng = np.random.default_rng(0)

# =====================================================
# 5. LANDMARK PROCESSING
# =====================================================

def to_points(hand):

    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )


def normalize_points(points):

    points = points - points[0]

    max_distance = np.max(
        np.linalg.norm(points, axis=1)
    )

    if max_distance > 0:
        points = points / max_distance

    return points.flatten()


def build_feature(point_sets):

    hand1 = normalize_points(point_sets[0])

    if len(point_sets) >= 2:
        hand2 = normalize_points(point_sets[1])
    else:
        hand2 = np.zeros(63, dtype=np.float32)

    return np.concatenate([
        hand1,
        hand2,
        [min(len(point_sets), 2)]
    ])


# =====================================================
# 6. DATA AUGMENTATION FOR PREDICTION
# =====================================================

def perturb(point_sets):

    angle = np.deg2rad(
        rng.uniform(-5, 5)
    )

    scale = rng.uniform(0.95, 1.05)

    c = np.cos(angle)
    s = np.sin(angle)

    rotation = np.array([
        [c, -s],
        [s, c]
    ], dtype=np.float32)

    centre = np.concatenate(
        point_sets
    )[:, :2].mean(axis=0)

    output = []

    for points in point_sets:

        q = points.copy()

        q[:, :2] = (
            (q[:, :2] - centre) @ rotation.T
        ) * scale + centre

        q[:, 2] *= scale

        q += rng.normal(
            0, 0.003, q.shape
        ).astype(np.float32)

        output.append(q)

    return output


def predict_tta(hands):

    base = [to_points(h) for h in hands]

    batch = [build_feature(base)]

    for _ in range(8):

        augmented = perturb(base)

        batch.append(
            build_feature(augmented)
        )

    probabilities = model.predict_proba(
        np.array(batch, dtype=np.float32)
    )

    return probabilities.mean(axis=0)


# =====================================================
# 7. RULE-BASED GESTURE RECOGNITION
# =====================================================

def dist(a, b):

    return float(np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    ))


def finger_up(hand, tip, pip):

    return (
        dist(hand[0], hand[tip]) >
        dist(hand[0], hand[pip]) * 1.10
    )


def thumb_out(hand):

    return (
        dist(hand[4], hand[17]) >
        dist(hand[3], hand[17]) * 1.10
    )


def recognize_rule(hand):

    index = finger_up(hand, 8, 6)
    middle = finger_up(hand, 12, 10)
    ring = finger_up(hand, 16, 14)
    pinky = finger_up(hand, 20, 18)

    thumb = thumb_out(hand)

    size = max(
        dist(hand[0], hand[9]),
        1e-6
    )

    pinch = dist(hand[4], hand[8]) / size

    if pinch < 0.35 and middle and ring and pinky:
        return "OK"

    if pinch < 0.30 and not (
        middle or ring or pinky
    ):
        return "PINCH"

    if (
        thumb
        and not (index or middle or ring or pinky)
        and hand[4].y < hand[3].y < hand[2].y
        and hand[4].y < hand[5].y
    ):
        return "THUMBS_UP"

    if index and middle and ring and pinky:
        return "FIVE" if thumb else "FOUR"

    if index and middle and ring:
        return "THREE"

    if index and middle:
        return "TWO"

    if index and not (middle or ring or pinky):

        horizontal = (
            abs(hand[8].x - hand[5].x) >
            abs(hand[8].y - hand[5].y)
        )

        return "POINTING" if horizontal else "ONE"

    return None


# =====================================================
# 8. FINAL PREDICTION
# =====================================================

def decide(probs, rule):

    order = np.argsort(probs)[::-1]

    top = float(probs[order[0]])

    second = (
        float(probs[order[1]])
        if len(order) > 1 else 0.0
    )

    label = str(CLASSES[order[0]])

    if top >= 0.80:
        return label, top, "ML"

    if rule is not None:
        return rule, top, "Rule"

    if top >= 0.60 and top - second >= 0.15:
        return label, top, "ML"

    return "UNKNOWN", top, "-"


def hand_ok(hand):

    if dist(hand[0], hand[9]) < 0.04:
        return False

    return all(
        0.01 <= lm.x <= 0.99
        and 0.01 <= lm.y <= 0.99
        for lm in hand
    )


# =====================================================
# 9. WEBSITE HOME PAGE
# =====================================================

@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# =====================================================
# 10. LIVE WEBCAM PREDICTION API
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    payload = request.get_json(silent=True) or {}

    data_url = payload.get("image", "")

    if "," not in data_url:

        return jsonify({
            "error": "No camera frame received"
        }), 400

    try:

        raw = base64.b64decode(
            data_url.split(",", 1)[1]
        )

        arr = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "error": "Could not read camera frame"
            }), 400

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        pairs = sorted(
            zip(
                result.hand_landmarks,
                result.handedness
            ),
            key=lambda p: p[0][0].x
        )

        hands = [p[0] for p in pairs]

        scores = [p[1][0].score for p in pairs]

        # No hands detected

        if not hands:

            history.clear()

            return jsonify({
                "label": "NO HAND",
                "confidence": 0,
                "source": "-",
                "hands": 0,
                "landmarks": []
            })

        # Prepare landmarks for website display

        landmarks = [
            [
                [
                    round(float(lm.x), 5),
                    round(float(lm.y), 5)
                ]
                for lm in hand
            ]
            for hand in hands
        ]

        # Check hand visibility

        if not all(
            hand_ok(h) and score >= 0.70
            for h, score in zip(hands, scores)
        ):

            history.clear()

            return jsonify({
                "label": "MOVE HAND INTO VIEW",
                "confidence": 0,
                "source": "-",
                "hands": len(hands),
                "landmarks": landmarks
            })

        # Rule-based prediction for one hand

        rule = (
            recognize_rule(hands[0])
            if len(hands) == 1
            else None
        )

        # Machine learning prediction

        probs = predict_tta(hands)

        label, confidence, source = decide(
            probs,
            rule
        )

        # Smooth predictions

        history.append(label)

        shown = Counter(history).most_common(1)[0][0]

        return jsonify({

            "label": shown,

            "confidence": round(
                confidence * 100, 1
            ),

            "source": source,

            "hands": len(hands),

            "landmarks": landmarks

        })

    except Exception as exc:

        app.logger.exception(
            "Gesture prediction failed"
        )

        return jsonify({
            "error": str(exc)
        }), 500


# =====================================================
# 11. START APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
