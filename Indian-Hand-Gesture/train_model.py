import os
import cv2
import time
import joblib
import numpy as np
import mediapipe as mp
from collections import deque


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "isl_model.pkl")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("       INDIAN HAND GESTURE RECOGNITION")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print("ERROR: isl_model.pkl not found")
    exit()

if not os.path.exists(HAND_MODEL_PATH):
    print("ERROR: hand_landmarker.task not found")
    exit()

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)

print("MediaPipe loaded successfully.")


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(hand):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    wrist = points[0].copy()

    points = points - wrist

    distances = np.linalg.norm(
        points,
        axis=1
    )

    max_distance = np.max(distances)

    if max_distance > 0:
        points = points / max_distance

    return points.flatten()


# ============================================================
# ML FEATURES
# ============================================================

def extract_features(hands):

    if len(hands) == 0:
        return None

    hands = sorted(
        hands,
        key=lambda hand: hand[0].x
    )

    hand0 = normalize_hand(hands[0])

    if len(hands) >= 2:
        hand1 = normalize_hand(hands[1])
    else:
        hand1 = np.zeros(63, dtype=np.float32)

    num_hands = min(len(hands), 2)

    features = np.concatenate([
        hand0,
        hand1,
        [num_hands]
    ])

    return features


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def distance(a, b):

    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def finger_extended(hand, tip, pip):

    return hand[tip].y < hand[pip].y


def get_fingers(hand):

    index = finger_extended(hand, 8, 6)
    middle = finger_extended(hand, 12, 10)
    ring = finger_extended(hand, 16, 14)
    pinky = finger_extended(hand, 20, 18)

    return index, middle, ring, pinky


# ============================================================
# RULE-BASED OLD GESTURES
# ============================================================

def recognize_old_gesture(hand):

    index, middle, ring, pinky = get_fingers(hand)

    # --------------------------------------------------------
    # THUMBS UP
    # --------------------------------------------------------

    thumb_up = (
        hand[4].y < hand[3].y
        and hand[4].y < hand[2].y
    )

    if (
        thumb_up
        and not index
        and not middle
        and not ring
        and not pinky
    ):
        return "THUMBS_UP"


    # --------------------------------------------------------
    # POINTING
    # --------------------------------------------------------

    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "POINTING"


    # --------------------------------------------------------
    # FIVE
    # --------------------------------------------------------

    if (
        index
        and middle
        and ring
        and pinky
    ):
        return "FIVE"


    # --------------------------------------------------------
    # ONE
    # --------------------------------------------------------

    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "ONE"


    # --------------------------------------------------------
    # TWO
    # --------------------------------------------------------

    if (
        index
        and middle
        and not ring
        and not pinky
    ):
        return "TWO"


    # --------------------------------------------------------
    # THREE
    # --------------------------------------------------------

    if (
        index
        and middle
        and ring
        and not pinky
    ):
        return "THREE"


    # --------------------------------------------------------
    # FOUR
    # --------------------------------------------------------

    if (
        index
        and middle
        and ring
        and pinky
        and not thumb_up
    ):
        return "FOUR"


    # --------------------------------------------------------
    # OK
    # --------------------------------------------------------

    thumb_index_distance = distance(
        hand[4],
        hand[8]
    )

    if (
        thumb_index_distance < 0.08
        and middle
        and ring
        and pinky
    ):
        return "OK"


    # --------------------------------------------------------
    # PINCH
    # --------------------------------------------------------

    if thumb_index_distance < 0.07:

        return "PINCH"


    return None


# ============================================================
# CLEAR LANDMARK DRAWING
# ============================================================

def draw_hand(frame, hand, hand_number):

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Connections
    # --------------------------------------------------------

    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (5, 9),
        (9, 13),
        (13, 17)
    ]


    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start, end in connections:

        x1 = int(hand[start].x * width)
        y1 = int(hand[start].y * height)

        x2 = int(hand[end].x * width)
        y2 = int(hand[end].y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )


    # --------------------------------------------------------
    # Draw numbered nodes
    # --------------------------------------------------------

    for i, landmark in enumerate(hand):

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            7,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (x, y),
            9,
            (255, 255, 255),
            2
        )

        # Landmark number
        cv2.putText(
            frame,
            str(i),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1
        )


    # --------------------------------------------------------
    # Hand number
    # --------------------------------------------------------

    wrist_x = int(hand[0].x * width)
    wrist_y = int(hand[0].y * height)

    cv2.putText(
        frame,
        f"Hand {hand_number}",
        (wrist_x + 15, wrist_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    detector.close()

    exit()


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# SMOOTHING
# ============================================================

prediction_history = deque(maxlen=7)


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

print("\nCamera started.")
print("Press Q to quit.\n")


start_time = time.time()


while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame error.")
        break


    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp = int(
        (time.time() - start_time) * 1000
    )


    # --------------------------------------------------------
    # DETECT
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )


    hands = result.hand_landmarks


    # --------------------------------------------------------
    # NO HAND
    # --------------------------------------------------------

    if not hands:

        prediction_history.clear()

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    else:

        # ----------------------------------------------------
        # SORT HANDS
        # ----------------------------------------------------

        hands = sorted(
            hands,
            key=lambda hand: hand[0].x
        )


        # ----------------------------------------------------
        # DRAW HANDS
        # ----------------------------------------------------

        for number, hand in enumerate(hands, start=1):

            draw_hand(
                frame,
                hand,
                number
            )


        # ====================================================
        # RULE-BASED GESTURES
        # ====================================================

        rule_prediction = None

        if len(hands) == 1:

            rule_prediction = recognize_old_gesture(
                hands[0]
            )


        # ====================================================
        # ML PREDICTION
        # ====================================================

        ml_prediction = None
        ml_confidence = 0.0


        features = extract_features(
            hands
        )


        if features is not None:

            features = features.reshape(
                1,
                -1
            )


            ml_prediction = model.predict(
                features
            )[0]


            probabilities = model.predict_proba(
                features
            )[0]


            ml_confidence = float(
                np.max(probabilities)
            )


        # ====================================================
        # CHOOSE RESULT
        # ====================================================

        # Rule-based old gestures take priority
        # when clearly detected.

        if rule_prediction is not None:

            final_prediction = rule_prediction

            confidence_text = "Rule Based"

        elif ml_prediction is not None:

            if ml_confidence >= 0.55:

                final_prediction = ml_prediction

                confidence_text = (
                    f"{ml_confidence * 100:.1f}%"
                )

            else:

                final_prediction = "UNKNOWN"

                confidence_text = (
                    f"{ml_confidence * 100:.1f}%"
                )

        else:

            final_prediction = "UNKNOWN"

            confidence_text = "0%"


        # ====================================================
        # SMOOTH ML PREDICTIONS
        # ====================================================

        if rule_prediction is None:

            prediction_history.append(
                final_prediction
            )

            if len(prediction_history) > 0:

                counts = {}

                for prediction in prediction_history:

                    counts[prediction] = (
                        counts.get(prediction, 0) + 1
                    )

                final_prediction = max(
                    counts,
                    key=counts.get
                )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (15, 15),
            (570, 145),
            (20, 20, 20),
            -1
        )


        cv2.putText(
            frame,
            "GESTURE",
            (35, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            2
        )


        cv2.putText(
            frame,
            str(final_prediction),
            (35, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            f"Confidence: {confidence_text}",
            (35, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Hands: {len(hands)}",
            (350, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (350, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "ML + RULE",
            (350, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


    # ========================================================
    # BOTTOM INFO
    # ========================================================

    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 45),
        (frame.shape[1], frame.shape[0]),
        (20, 20, 20),
        -1
    )


    cv2.putText(
        frame,
        "Q = Quit",
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Indian Hand Gesture Recognition",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

detector.close()

print("\nProgram closed.")