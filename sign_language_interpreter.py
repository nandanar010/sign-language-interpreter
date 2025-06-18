import cv2
import mediapipe as mp
import pyttsx3

# Initialize modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

spoken = ""

def speak(text):
    global spoken
    if text != spoken:
        print(f"Detected: {text}")
        engine.say(text)
        engine.runAndWait()
        spoken = text

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = ""

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = hand.landmark

            # Thumb up
            if landmarks[4].y < landmarks[3].y and landmarks[8].y > landmarks[6].y:
                gesture = "Thumbs Up"
            # Thumb down
            elif landmarks[4].y > landmarks[3].y and landmarks[8].y > landmarks[6].y:
                gesture = "Thumbs Down"
            # Open palm (Hi)
            elif all(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]):
                gesture = "Hi"
            # Only index up
            elif landmarks[8].y < landmarks[6].y and all(landmarks[i].y > landmarks[i - 2].y for i in [12, 16, 20]):
                gesture = "One"
            # I Love You (thumb, index, pinky up)
            elif (
                landmarks[4].y < landmarks[3].y and
                landmarks[8].y < landmarks[6].y and
                landmarks[20].y < landmarks[18].y and
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y
            ):
                gesture = "I Love You"

            if gesture:
                cv2.putText(frame, gesture, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                speak(gesture)
            else:
                spoken = ""

    cv2.imshow("Sign Language Interpreter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
