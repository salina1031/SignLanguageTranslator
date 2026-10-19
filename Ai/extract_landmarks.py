import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
from tqdm import tqdm

# ========== SETTINGS ==========
DATASET_PATH = "data/asl_alphabet_train"  # path to your dataset
OUTPUT_FILE = "data/landmarks.csv"
MAX_IMAGES_PER_CLASS = 300   # limit per letter (to keep it faster)
# ==============================

os.makedirs("data", exist_ok=True)

# Load MediaPipe model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Prepare CSV
csv_file = open(OUTPUT_FILE, mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Header
header = ["label"]
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
csv_writer.writerow(header)

# Get all class folders (A, B, C, ...)
classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
print(f"Found classes: {classes}")

total_saved = 0

for label in classes:
    folder = os.path.join(DATASET_PATH, label)
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = images[:MAX_IMAGES_PER_CLASS]  # limit

    print(f"\nProcessing {label} ({len(images)} images)...")

    for img_name in tqdm(images):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            row = [label]
            for lm in hand:
                row.extend([lm.x, lm.y, lm.z])
            csv_writer.writerow(row)
            total_saved += 1

csv_file.close()
detector.close()

print(f"\nDone! Total samples saved: {total_saved}")
print(f"Saved to: {OUTPUT_FILE}")