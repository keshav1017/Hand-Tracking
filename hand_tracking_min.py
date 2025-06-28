import cv2
import time
import mediapipe as mp

# Use DirectShow backend for better performance on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraws = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

            mpDraws.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    # Optionally resize for smoother display
    img = cv2.resize(img, (960, 540))

    # to display the fps
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
