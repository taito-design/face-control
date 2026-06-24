import cv2
import mediapipe as mp
import pyautogui
import math

pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# landmark IDs
FOREHEAD = 10
CHIN = 152
UPPER_LIP = 13
LOWER_LIP = 14
RIGHT_BROW = 105
RIGHT_EYE_FOR_BROW = 386
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOT = 145
LEFT_EYE_TOP = 374
LEFT_EYE_BOT = 386

# thresholds
MOUTH_OPEN_RATIO = 0.15
BROW_RAISE_RATIO = 0.18
WINK_RATIO = 0.015

COOLDOWN_FRAMES = 15

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def get_ratio(lm, id_a, id_b, face_height):
    return dist(lm[id_a], lm[id_b]) / face_height if face_height > 0 else 0

def draw_label(frame, text, color):
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (w // 2 - 150, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

cap = cv2.VideoCapture(0)

system_active = True
cooldown_right = 0
cooldown_left = 0
label_text = ""
label_color = (0, 255, 0)
label_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_color = (0, 255, 0) if system_active else (0, 0, 255)
    status_text = "SYSTEM: ON" if system_active else "SYSTEM: PAUSED (left wink to resume)"
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        face_height = dist(lm[FOREHEAD], lm[CHIN])
        if face_height == 0:
            cv2.imshow("Face Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        mouth_ratio = get_ratio(lm, UPPER_LIP, LOWER_LIP, face_height)
        brow_ratio = get_ratio(lm, RIGHT_BROW, RIGHT_EYE_FOR_BROW, face_height)
        right_wink_ratio = get_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOT, face_height)
        left_wink_ratio = get_ratio(lm, LEFT_EYE_TOP, LEFT_EYE_BOT, face_height)

        if cooldown_right > 0:
            cooldown_right -= 1
        if cooldown_left > 0:
            cooldown_left -= 1

        # left wink — toggle system (always active)
        if left_wink_ratio < WINK_RATIO and cooldown_left == 0:
            system_active = not system_active
            label_text = "SYSTEM ON" if system_active else "SYSTEM PAUSED"
            label_color = (0, 255, 0) if system_active else (0, 0, 255)
            label_frames = 30
            cooldown_left = COOLDOWN_FRAMES

        if system_active:
            # mouth open → scroll down
            if mouth_ratio > MOUTH_OPEN_RATIO:
                pyautogui.scroll(-100)
                label_text = "SCROLL DOWN"
                label_color = (0, 165, 255)
                label_frames = 10

            # brow raise → scroll up
            elif brow_ratio > BROW_RAISE_RATIO:
                pyautogui.scroll(100)
                label_text = "SCROLL UP"
                label_color = (255, 200, 0)
                label_frames = 10

            # right wink → play/pause
            if right_wink_ratio < WINK_RATIO and cooldown_right == 0:
                pyautogui.press("space")
                label_text = "PLAY / PAUSE"
                label_color = (0, 255, 0)
                label_frames = 30
                cooldown_right = COOLDOWN_FRAMES

    if label_frames > 0:
        draw_label(frame, label_text, label_color)
        label_frames -= 1

    cv2.imshow("Face Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
