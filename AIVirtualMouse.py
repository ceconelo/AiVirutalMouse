import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

######################################
wCam, hCam = 640, 480
frameR = 100 #Frame Reduction
smoothening = 1
######################################
pTime = 0
pLocX, pLocY = 0,0
cLocX, cLocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wCam) #Id 3 for Width
cap.set(4, hCam) #Id 4 for Higth

detector = htm.HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size() #get size of monitor

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
       x1, y1 = lmList[8] #Index finger
       x2, y2 = lmList[12] #Middle finger
       #print(x1,y1, x2,y2)
       # 3. Check which finges are up
       fingers = detector.fingersUp()
       #print(fingers)
       # 4. Only Index Finger: Moving Mode
       cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2) #Finger mouse point dectition area
       if fingers[1] == 1 and fingers[2] == 0:
           # 5. Convert coordinates
           x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
           y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
           # 6. Smoothen Values
           cLocX = pLocX + (x3 - pLocX) / smoothening
           cLocY = pLocY + (y3 - cLocY) / smoothening
           # 7. Move Mouse
           autopy.mouse.move(wScr-x3, y3) #Go flip the position
           cv2.circle(img, (x1, y1), 15, (255,0,255), cv2.FILLED) #Draw an circle in fingertip
           pLocX, pLocY = cLocX, cLocY

        # 8. Both Index and moddle fingers are up: Clicking mode
       if fingers[1] == 1 and fingers[2] == 1:
           # 9. Find distance between fingers
           length, img, lineInfo = detector.findDistance(8,12, img)
           # 10. Click mouse if distance short
           if length < 35.0:
               cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
               autopy.mouse.click()
    # 11. Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
