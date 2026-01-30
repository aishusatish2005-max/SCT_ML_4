import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Make mediapipe lighter (important)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,   # 🔥 reduces memory usage
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
        return landmarks
    
    return None

def run_camera(source):
    cap = cv2.VideoCapture(source)

    # 🔥 reduce resolution to avoid memory error
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not found!")
        return

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = get_landmarks(frame)

        if landmarks is not None:
            data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(data)[0]

            cv2.putText(frame, f"Gesture: {prediction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("""
===============================
 HAND GESTURE RECOGNITION SYSTEM
===============================
1 - Webcam Mode
2 - Video File Mode
3 - External Camera
0 - Exit
""")

choice = input("Select Mode (1/2/3): ")

if choice == "1":
    run_camera(0)
elif choice == "2":
    path = input("Enter video file path: ")
    run_camera(path)
elif choice == "3":
    run_camera(1)
else:
    print("Exited")
