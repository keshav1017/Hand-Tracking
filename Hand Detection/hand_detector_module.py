import time

import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands

        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraws = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)

        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraws.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)

            
        return img
    
    def find_positions(self, img, hand_no=0, draw=True):
        
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

        return lm_list

def main():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    prevTime = 0
    currTime = 0

    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_positions(img)
        if len(lm_list):
            print(lm_list[0])

        # resize the image
        img = cv2.resize(img, (960, 540))

        # to display the fps
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
        )

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
