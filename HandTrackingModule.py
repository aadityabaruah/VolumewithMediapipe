import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.hands = hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.hands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)#
        if results.multi_hand_landmarks:
            for count in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, count, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        self.lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]

            for idnumber, lms in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lms.x * w), int(lms.y * h)
                self.lmList.append([idnumber, cx, cy])

        return self.lmList

    def fingersUp(self):
        finger = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            finger.append(1)
        else:
            finger.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                finger.append(1)
            else:
                finger.append(0)

        return finger

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[4][1], self.lmList[4][2]
        x2, y2 = self.lmList[8][1], self.lmList[8][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)


def main():
    PreviousTime = 0

    capture = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = capture.read()
        img = detector.findHands(img)

        CurrentTime = time.time()
        fps = int(1 / (CurrentTime - PreviousTime))
        PreviousTime = CurrentTime

        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.putText(img, str(f"FPS - {fps}"), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
