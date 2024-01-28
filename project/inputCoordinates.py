# get the coordinates of the corners of the field 
import cv2
import numpy as np
import random

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    r = random.randint(0,256)
    g = random.randint(0,256)
    b = random.randint(0,256)
    cv2.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

f = open("corners.txt", "w")

# Load the first frame of the video
cap = cv2.VideoCapture('frisbee.mp4')
ret, img = cap.read()

trackerList = []
bboxes = []

#Instantiate trackers
for i in range(4): 
    tracker = cv2.legacy.TrackerCSRT_create()
    trackerList.append(tracker)

#Select corners 
for j in range(4):
    corner = ["top left", "top right", "bottom right", "bottom left"]
    print('Draw a box around the ' + corner[j] + ' corner.')
    bbox = cv2.selectROI('Select Corner', img, False)
    bboxes.append(bbox)
    drawBox(img, bbox)
    trackerList[j].init(img, bboxes[j])
    cv2.destroyWindow('Select Corner')

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    success, bboxes[i] = trackerList[i].update(img)
    f.write(str(bboxes) + "\n")

cap.release()
f.close()
cv2.destroyAllWindows()