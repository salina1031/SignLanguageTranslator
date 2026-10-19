import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import time

# ========== SETTINGS ==========
LABEL = "A"          # Change this to the letter/word you want to collect (A, B, Hello, etc.)
SAMPLES_TO_COLLECT = 100
OUTPUT_FILE = "data/landmarks.csv"
# ==============================

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Prepare CSV file
file_exists = os.path.isfile(OUTPUT_FILE)
csv_file = open(OUTPUT_FILE, mode="a", newline="")
csv_writer = csv.writer(csv_file)

# Write header only once
if not file_exists:
    header = ["label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    csv_writer.writerow(header)

cap = cv2.VideoCapture(0)
collected = 0

print(f"Collecting data for label: '{LABEL}'")
print("Show your hand clearly and press 's' to save a sample")
print("Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    # Draw landmarks
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Show instruction
        cv2.putText(frame, f"Label: {LABEL} | Collected: {collected}/{SAMPLES_TO_COLLECT}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to save | 'q' to quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Data Collection - Sign Language", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and result.hand_landmarks:
        # Save the first detected hand
        hand = result.hand_landmarks[0]
        row = [LABEL]
        for lm in hand:
            row.extend([lm.x, lm.y, lm.z])
        csv_writer.writerow(row)
        collected += 1
        print(f"Saved sample {collected}/{SAMPLES_TO_COLLECT}")

        if collected >= SAMPLES_TO_COLLECT:
            print("Collection complete for this label!")
            break

    elif key == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
detector.close()
print(f"Data saved to {OUTPUT_FILE}")