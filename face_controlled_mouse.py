import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ---------------- FUNCTIONS ----------------

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

def eye_ratio(landmarks, eye):
    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    horizontal = distance(p1, p4)
    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)

    return (vertical1 + vertical2) / (2 * horizontal)

# ---------------- MEDIAPIPE ----------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- LANDMARKS ----------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Nose landmark
NOSE = 1

# ---------------- SCREEN ----------------

screen_w, screen_h = pyautogui.size()

smooth_x = screen_w // 2
smooth_y = screen_h // 2

smooth_factor = 0.20

# ---------------- BLINK ----------------

blink_threshold = 0.20

left_eye_closed = False
right_eye_closed = False

left_blinks = 0
right_blinks = 0

last_left_blink = 0
last_right_blink = 0

double_blink_gap = 0.45

left_clicks = 0
right_clicks = 0

last_left_click = 0
last_right_click = 0

click_cooldown = 1.0

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera Open Failed")
    exit()

pyautogui.FAILSAFE = False

print("Accessibility System Started")

while True:

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]
        # ---------------- CURSOR ----------------

        nose = face_landmarks.landmark[NOSE]

        target_x = int(nose.x * screen_w)
        target_y = int(nose.y * screen_h)

        smooth_x = smooth_x + (target_x - smooth_x) * smooth_factor
        smooth_y = smooth_y + (target_y - smooth_y) * smooth_factor

        pyautogui.moveTo(int(smooth_x), int(smooth_y))

        # ---------------- EAR ----------------

        left_ear = eye_ratio(face_landmarks.landmark, LEFT_EYE)
        right_ear = eye_ratio(face_landmarks.landmark, RIGHT_EYE)

        cv2.putText(
            frame,
            f"L:{left_ear:.2f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"R:{right_ear:.2f}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        # ---------------- DRAW LEFT EYE ----------------

        for idx in LEFT_EYE:

            point = face_landmarks.landmark[idx]

            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(frame, (x, y), 3, (0,255,0), -1)

        # ---------------- DRAW RIGHT EYE ----------------

        for idx in RIGHT_EYE:

            point = face_landmarks.landmark[idx]

            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(frame, (x, y), 3, (0,0,255), -1)
        # ---------------- BLINK STATE MACHINE ----------------

        current_time = time.time()

        # ---------- LEFT EYE ---------

        if left_ear < blink_threshold:

            if not left_eye_closed:
                left_eye_closed = True

        else:

            if left_eye_closed:

                left_eye_closed = False

                left_blinks += 1

                if current_time - last_left_blink <= double_blink_gap:

                    if current_time - last_left_click > click_cooldown:

                        pyautogui.click()

                        left_clicks += 1

                        last_left_click = current_time

                        print("LEFT CLICK")

                else:

                    print("LEFT BLINK")

                last_left_blink = current_time
        # ---------- RIGHT EYE ----------

        if right_ear < blink_threshold:

            if not right_eye_closed:
                right_eye_closed = True

        else:

            if right_eye_closed:

                right_eye_closed = False

                right_blinks += 1

                if current_time - last_right_blink <= double_blink_gap:

                    if current_time - last_right_click > click_cooldown:

                        pyautogui.rightClick()

                        right_clicks += 1

                        last_right_click = current_time

                        print("RIGHT CLICK")

                else:

                    print("RIGHT BLINK")

                last_right_blink = current_time

        # ---------------- SHOW COUNTS ----------------

        cv2.putText(
            frame,
            f"Left Blinks : {left_blinks}",
            (20,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Right Blinks : {right_blinks}",
            (20,240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )
        cv2.putText(
            frame,
            f"Left Clicks : {left_clicks}",
            (20,280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            frame,
            f"Right Clicks : {right_clicks}",
            (20,320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    cv2.imshow("Accessibility System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()