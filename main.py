import cv2
import mediapipe as mp
import time
from directkeys import PressKey, ReleaseKey, right_pressed, left_pressed

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

tipIds = [4, 8, 12, 16, 20]
video = cv2.VideoCapture(0)
current_key_pressed = set()

def draw_fancy_box(img, text, color=(0, 255, 0), pos=(20, 300), size=(270, 425)):
    x1, y1 = pos
    x2, y2 = size
    overlay = img.copy()
    alpha = 0.5
    cv2.rectangle(overlay, pos, size, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, pos, size, color, 2)

    font_scale = 1.5
    font_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = x1 + ((x2 - x1 - text_size[0]) // 2)
    text_y = y1 + ((y2 - y1 + text_size[1]) // 2)

    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        keyPressed = False
        key_count = 0
        key_pressed = 0

        ret, image = video.read()
        if not ret:
            continue

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                myHands = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHands.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
       
        fingers = []
        if len(lmList) != 0:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)
            if total == 0:
                draw_fancy_box(image, "BRAKE", color=(0, 0, 255))
                PressKey(left_pressed)
                keyPressed = True
                key_pressed = left_pressed
                current_key_pressed.add(left_pressed)
                key_count += 1
            elif total == 5:
                draw_fancy_box(image, "GAS", color=(0, 255, 0))
                PressKey(right_pressed)
                keyPressed = True
                key_pressed = right_pressed
                current_key_pressed.add(right_pressed)
                key_count += 1

        if not keyPressed and len(current_key_pressed) != 0:
            for key in current_key_pressed:
                ReleaseKey(key)
            current_key_pressed = set()

        elif key_count == 1 and len(current_key_pressed) == 2:
            for key in current_key_pressed:
                if key_pressed != key:
                    ReleaseKey(key)
            current_key_pressed = set()

        cv2.imshow("Frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()




