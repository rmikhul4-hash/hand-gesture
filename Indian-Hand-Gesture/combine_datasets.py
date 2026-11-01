import pandas as pd
import os

print("=" * 60)
print("       COMBINING ISL DATASETS")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ONE_HAND_FILE = os.path.join(BASE_DIR, "dataset.csv")
TWO_HAND_FILE = os.path.join(BASE_DIR, "dataset_two_hand.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_final.csv")


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

print("\nLoading one-hand dataset...")
one_hand = pd.read_csv(ONE_HAND_FILE)

print("Loading two-hand dataset...")
two_hand = pd.read_csv(TWO_HAND_FILE)

print("\nOne-hand dataset shape:", one_hand.shape)
print("Two-hand dataset shape:", two_hand.shape)


# --------------------------------------------------
# SHOW LABELS
# --------------------------------------------------

print("\nOne-hand gestures:")
print(one_hand["label"].value_counts())

print("\nTwo-hand gestures:")
print(two_hand["label"].value_counts())


# --------------------------------------------------
# CHECK FEATURES
# --------------------------------------------------

one_hand_features = [
    col for col in one_hand.columns
    if col != "label"
]

two_hand_features = [
    col for col in two_hand.columns
    if col != "label"
]

print("\nOne-hand features:", len(one_hand_features))
print("Two-hand features:", len(two_hand_features))


# --------------------------------------------------
# CREATE NEW ONE-HAND DATASET
# USING EXACT TWO-HAND COLUMN NAMES
# --------------------------------------------------

print("\nConverting one-hand dataset...")


# Create empty dataframe with the exact
# same columns as the two-hand dataset
converted = pd.DataFrame(
    0.0,
    index=one_hand.index,
    columns=two_hand_features
)


# --------------------------------------------------
# COPY LABEL
# --------------------------------------------------

converted["label"] = one_hand["label"]


# --------------------------------------------------
# COPY ONE-HAND LANDMARKS
# --------------------------------------------------

# Your one-hand dataset contains:
#
# x0 y0 z0
# x1 y1 z1
# ...
# x20 y20 z20
#
# We put them into hand0_x0,
# hand0_y0, hand0_z0, etc.

for i in range(21):

    converted[f"hand0_x{i}"] = one_hand.iloc[:, 1 + i * 3]
    converted[f"hand0_y{i}"] = one_hand.iloc[:, 2 + i * 3]
    converted[f"hand0_z{i}"] = one_hand.iloc[:, 3 + i * 3]


# --------------------------------------------------
# SECOND HAND
# --------------------------------------------------

# hand1_x0 ... hand1_z20 remain 0
# because these samples contain only one hand.


# --------------------------------------------------
# NUMBER OF HANDS
# --------------------------------------------------

converted["num_hands"] = 1


# --------------------------------------------------
# REORDER COLUMNS
# --------------------------------------------------

final_columns = list(two_hand.columns)

converted = converted[final_columns]


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

print("\nCombining datasets...")

final_dataset = pd.concat(
    [two_hand, converted],
    ignore_index=True
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

final_dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("       DATASET COMBINATION COMPLETE")
print("=" * 60)

print("\nTotal samples:", len(final_dataset))

print("\nSamples per gesture:")
print(final_dataset["label"].value_counts())

print("\nTotal columns:", len(final_dataset.columns))

print("Total features:", len(final_dataset.columns) - 1)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nDone!")