import cv2 as cv
import numpy as np

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (255,0,0), 3,1)
    cv.putText(img, "Tracking", (120,75), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)

cap = cv.VideoCapture('Videos/frisbee.mp4')

#tracker = cv.legacy.TrackerBoosting_create()
tracker = cv.legacy.TrackerCSRT_create()
#tracker = cv.legacy.TrackerMIL_create()

ret, img = cap.read()

bbox = cv.selectROI(img, False)
tracker.init(img, bbox)
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv.VideoWriter("huck_sydney.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720))

while True:
    success, img = cap.read()
    if not success:
        break

    success, bbox = tracker.update(img)

    if success:
        drawBox(img,bbox)

    out.write(img)

cap.release()
out.release()
cv.destroyAllWindows()
