import cv2    #260,8
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

PreviousTime = 0
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx, cy = (x1+x2)//2, (y1+y2)//2

        length = math.hypot(x2-x1, y2-y1)
        
        vol = np.interp(length, [100, 240], [minVol, maxVol])
        if vol > minVol:
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED, cv2.LINE_AA)
        else:
            cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(img, (x2, y2), 8, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)

        volume.SetMasterVolumeLevel(vol, None)
        



    CurrentTime = time.time()
    fps = int(1 / (CurrentTime - PreviousTime))
    PreviousTime = CurrentTime

    cv2.putText(img, str(f"FPS - {fps}"), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
