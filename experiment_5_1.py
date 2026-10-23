import os
import urllib.request
import cv2
import numpy as np

# 1. Automatically download Google's official pretrained Gesture Recognizer model
MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading pretrained MediaPipe Gesture Recognizer model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

# Hand skeleton connections for drawing landmarks
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         # Thumb
    (0,5), (5,6), (6,7), (7,8),         # Index
    (5,9), (9,10), (10,11), (11,12),    # Middle
    (9,13), (13,14), (14,15), (15,16),  # Ring
    (13,17), (17,18), (18,19), (19,20), (0,17) # Pinky & Palm
]

# 2. Try loading the modern MediaPipe Pretrained Task API first
MODE = "CONTOUR_FALLBACK"
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    MODE = "MP_PRETRAINED_TASKS"
    print("Loaded: MediaPipe Pretrained Gesture Recognizer (.task)")
except Exception as e:
    print(f"Tasks API unavailable ({e}), checking legacy solutions...")
    try:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.9
        )
        MODE = "MP_LEGACY_SOLUTIONS"
        print("Loaded: MediaPipe Legacy Hands + Landmark Gesture Classifier")
    except ImportError:
        print("Loaded: OpenCV HSV Contour Fallback")


def classify_landmarks_fallback(landmarks):
    """Rule-based gesture backup if only raw 21 landmarks are available."""
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    pips = [6, 10, 14, 18]
    up = [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips)]
    thumb_up = landmarks[4].y < landmarks[3].y < landmarks[2].y

    if up == [False, False, False, False]:
        return "Thumb_Up" if thumb_up else "Closed_Fist"
    if up == [True, True, True, True]:
        return "Open_Palm"
    if up == [True, True, False, False]:
        return "Victory (Peace)"
    if up == [True, False, False, False]:
        return "Pointing_Up"
    if up == [True, False, False, True]:
        return "ILoveYou / Rock"
    return f"{sum(up)} Fingers Up"


cap = cv2.VideoCapture(0)
timestamp_ms = 0

print("Press 'q' to quit...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # --- MODE 1: Pretrained MediaPipe Gesture Recognizer Model ---
    if MODE == "MP_PRETRAINED_TASKS":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms += 33  # Increment timestamp for video mode
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                # Draw connections
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                # Read the pretrained model's predicted gesture
                if result.gestures and idx < len(result.gestures):
                    top_gesture = result.gestures[idx][0]
                    gesture_name = top_gesture.category_name
                    score = top_gesture.score
                    label = f"Gesture: {gesture_name} ({score:.0%})"
                else:
                    label = "Gesture: None"

                # Display banner at top-left and near wrist
                cv2.rectangle(frame, (10, 10), (430, 55), (0, 0, 0), -1)
                cv2.putText(frame, label, (20, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # --- MODE 2: Legacy mp.solutions.hands + Landmark Classifier ---
    elif MODE == "MP_LEGACY_SOLUTIONS":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                gesture_name = classify_landmarks_fallback(hand_landmarks.landmark)
                cv2.rectangle(frame, (10, 10), (400, 55), (0, 0, 0), -1)
                cv2.putText(frame, f"Gesture: {gesture_name}", (20, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # --- MODE 3: OpenCV HSV Contour Fallback ---
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 2000:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, "Hand Detected (Fallback)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Experiment 5: Gesture & Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if MODE == "MP_PRETRAINED_TASKS":
    recognizer.close()
cap.release()
cv2.destroyAllWindows()