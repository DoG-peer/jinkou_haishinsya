import numpy as np
import cv2

# VIDEO = "./data/owata_ta.mkv"
# VIDEO = "./data/a.flv"
# VIDEO = "./data/unsaga.mp4"
VIDEO = 0
cap = cv2.VideoCapture(VIDEO)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    print(ret)
    fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
