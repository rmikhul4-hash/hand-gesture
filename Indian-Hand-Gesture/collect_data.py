import os
import csv
import time
import cv2
import mediapipe as mp
import numpy as np

print("Starting ISL 1-Hand + 2-Hand Dataset Collector...")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset_two_hand.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_GESTURE = 200

GESTURES = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "PLEASE",
    "HELP",
    "GOOD",
    "BAD",
    "LOVE",
    "FRIEND"
]


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nERROR: hand_landmarker.task not found!")
    print(MODEL_PATH)

    input("\nPress ENTER to exit...")
    exit()

print("MediaPipe model found.")


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    # IMPORTANT: TWO HANDS
    num_hands=2,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)

print("MediaPipe initialized.")
print("Two-hand detection enabled.")


# ============================================================
# DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    header = ["label"]

    # 2 hands × 21 landmarks × 3 coordinates
    for hand in range(2):

        for point in range(21):

            header.append(
                f"hand{hand}_x{point}"
            )

            header.append(
                f"hand{hand}_y{point}"
            )

            header.append(
                f"hand{hand}_z{point}"
            )

    # Number of detected hands
    header.append("num_hands")

    with open(
        DATASET_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)

    print("Created:")
    print(DATASET_PATH)

else:

    print("Dataset already exists:")
    print(DATASET_PATH)


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(hand):

    points = np.array(

        [
            [
                landmark.x,
                landmark.y,
                landmark.z
            ]

            for landmark in hand
        ],

        dtype=np.float32
    )

    # Wrist becomes origin
    points = points - points[0]

    # Scale normalization
    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        points = points / scale

    return points


# ============================================================
# EXTRACT TWO-HAND FEATURES
# ============================================================

def extract_features(result):

    hands = result.hand_landmarks

    # --------------------------------------------------------
    # NO HAND
    # --------------------------------------------------------

    if len(hands) == 0:
        return None

    # --------------------------------------------------------
    # SORT HANDS CONSISTENTLY
    #
    # Sort by wrist X position.
    #
    # Left side of camera = first
    # Right side of camera = second
    # --------------------------------------------------------

    hands = sorted(
        hands,
        key=lambda hand: hand[0].x
    )

    features = []

    # --------------------------------------------------------
    # FIRST HAND
    # --------------------------------------------------------

    first_hand = normalize_hand(
        hands[0]
    )

    features.extend(
        first_hand.flatten().tolist()
    )

    # --------------------------------------------------------
    # SECOND HAND
    # --------------------------------------------------------

    if len(hands) >= 2:

        second_hand = normalize_hand(
            hands[1]
        )

        features.extend(
            second_hand.flatten().tolist()
        )

    else:

        # Fill missing second hand with zeros
        features.extend(
            [0.0] * 63
        )

    # Number of hands
    features.append(
        float(min(len(hands), 2))
    )

    return features


# ============================================================
# COLLECT ONE GESTURE
# ============================================================

def collect_gesture(label):

    print()
    print("=" * 60)
    print("GESTURE:", label)
    print("=" * 60)

    print()
    print("Show the ACTUAL ISL sign.")
    print()
    print("The program automatically detects:")
    print("  1 hand")
    print("  2 hands")
    print()
    print("Press SPACE to start.")
    print("Press Q to cancel.")
    print()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Camera could not be opened.")

        return

    collecting = False

    samples = 0

    timestamp = 0

    last_save = 0

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read camera.")

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 33

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        # ====================================================
        # UI
        # ====================================================

        cv2.putText(
            frame,
            f"GESTURE: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )

        hand_count = len(
            result.hand_landmarks
        )

        cv2.putText(
            frame,
            f"HANDS: {hand_count}/2",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"SAMPLES: {samples}/{SAMPLES_PER_GESTURE}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        if not collecting:

            cv2.putText(
                frame,
                "PRESS SPACE TO START",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "COLLECTING...",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # ====================================================
        # CAMERA WINDOW
        # ====================================================

        cv2.imshow(
            "ISL Dataset Collector - 1 & 2 Hands",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # SPACE
        # ----------------------------------------------------

        if key == 32:

            if not collecting:

                collecting = True

                print()
                print(
                    f"Collecting {label}..."
                )

        # ----------------------------------------------------
        # Q
        # ----------------------------------------------------

        elif key == ord("q"):

            print("\nCollection cancelled.")

            break

        # ====================================================
        # SAVE SAMPLE
        # ====================================================

        if collecting:

            current_time = time.time()

            # Don't save too quickly
            if current_time - last_save >= 0.12:

                features = extract_features(
                    result
                )

                # At least one hand must be detected
                if features is not None:

                    row = [
                        label
                    ] + features

                    with open(
                        DATASET_PATH,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(
                            file
                        )

                        writer.writerow(row)

                    samples += 1

                    last_save = current_time

                    print(
                        f"\r{label}: "
                        f"{samples}/"
                        f"{SAMPLES_PER_GESTURE} "
                        f"| Hands: "
                        f"{hand_count}",
                        end=""
                    )

        # ====================================================
        # FINISHED
        # ====================================================

        if samples >= SAMPLES_PER_GESTURE:

            print()
            print()
            print(
                f"{label} completed!"
            )

            break

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# MENU
# ============================================================

print()
print("=" * 60)
print("       ISL ML DATASET COLLECTOR")
print("       1 HAND + 2 HAND SUPPORT")
print("=" * 60)

print()

for number, gesture in enumerate(
    GESTURES,
    start=1
):

    print(
        f"{number}. {gesture}"
    )

print()
print("0. EXIT")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    choice = input(
        "\nChoose gesture number: "
    ).strip()

    if choice == "0":
        break

    try:

        number = int(choice)

        if 1 <= number <= len(GESTURES):

            gesture = GESTURES[
                number - 1
            ]

            collect_gesture(
                gesture
            )

        else:

            print(
                "Invalid number."
            )

    except ValueError:

        print(
            "Enter a number."
        )


detector.close()

print()
print("=" * 60)
print("DATA COLLECTION FINISHED")
print("=" * 60)

print()
print("Dataset:")
print(DATASET_PATH)

input("\nPress ENTER to close...")