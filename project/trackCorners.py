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
corner = ["top left", "top right", "bottom right", "bottom left"]

#Instantiate trackers
for i in range(4): 
    tracker = cv2.legacy.TrackerCSRT_create()
    trackerList.append(tracker)

#Select corners 
for j in range(4):
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

    # Loop through each bounding box
    for i in range(0, len(bboxes)):

        # If tracking was lost, select new ROI of player
        if (bboxes[i] == 0 or trackerList[i] == 0):
            print("Tracking of the " + corner[i] + " corner was lost!")
            bbox = cv2.selectROI('Reselect Lost Corner', img, False)

            # If no coordinates were selected
            if bbox == (0, 0, 0, 0):
                continue

            # Insert new bbox into list and continue tracking
            else:
                bboxes[i] = bbox
                new_tracker = cv2.legacy.TrackerCSRT_create()
                trackerList[i] = new_tracker
                trackerList[i].init(img, bboxes[i])
            cv2.destroyWindow('Reselect Lost Corner')

        else:
            # Update box
            success, bboxes[i] = trackerList[i].update(img)

            # Update successful
            if success:
                drawBox(img, bboxes[i])
                
            # Unsuccessful
            else:
                bboxes[i] = 0

        f.write(str(bboxes) + "\n")

cap.release()
f.close()
cv2.destroyAllWindows()