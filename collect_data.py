import cv2
import mediapipe as mp
import csv
import os

GESTURE_LABEL = "peace"   # 👈 change gesture name each time
DATA_FILE = "hand_gesture_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

file_exists = os.path.isfile(DATA_FILE)

with open(DATA_FILE, mode='a', newline='') as f:
    writer = csv.writer(f)

    if not file_exists:
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        writer.writerow(header)

    print("Press 's' to save data | 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand.landmark:
                landmarks.append(lm.x)
            for lm in hand.landmark:
                landmarks.append(lm.y)

        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1)

        if key == ord('s') and landmarks is not None:
            writer.writerow([GESTURE_LABEL] + landmarks)
            print("✅ Saved:", GESTURE_LABEL)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
