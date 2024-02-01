# get the coordinates of the corners of the field 
import cv2
import numpy as np
import random

def drawBox(img, cornerBbox):
    x, y, w, h = int(cornerBbox[0]), int(cornerBbox[1]), int(cornerBbox[2]), int(cornerBbox[3])
    r = random.randint(0,256)
    g = random.randint(0,256)
    b = random.randint(0,256)
    cv2.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

f = open("corners.txt", "w")

# Load the first frame of the video
cap = cv2.VideoCapture('frisbee.mp4')
ret, img = cap.read()

cornerTrackerList = []
cornerBboxes = []
corner = ["top left", "top right", "bottom right", "bottom left"]

#Instantiate trackers
for i in range(4): 
    tracker = cv2.TrackerCSRT_create()
    cornerTrackerList.append(tracker)

#Select corners 
for j in range(4):
    print('Draw a box around the ' + corner[j] + ' corner.')
    cornerBbox = cv2.selectROI('Select the ' + corner[j] + ' corner', img, False)
    cornerBboxes.append(cornerBbox)
    drawBox(img, cornerBbox)
    cornerTrackerList[j].init(img, cornerBboxes[j])
    cv2.destroyWindow('Select the ' + corner[j] + ' corner')

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    # Loop through each bounding box
    for i in range(0, len(cornerBboxes)):

        # If tracking was lost, select new ROI of player
        if (cornerBboxes[i] == 0 or cornerTrackerList[i] == 0):
            print("Tracking of the " + corner[i] + " corner was lost!")
            cornerBbox = cv2.selectROI('Reselect Lost Corner', img, False)

            # If no coordinates were selected
            if cornerBbox == (0, 0, 0, 0):
                continue

            # Insert new cornerBbox into list and continue tracking
            else:
                cornerBboxes[i] = cornerBbox
                new_tracker = cv2.legacy.TrackerCSRT_create()
                cornerTrackerList[i] = new_tracker
                cornerTrackerList[i].init(img, cornerBboxes[i])
            cv2.destroyWindow('Reselect Lost Corner')

        else:
            # Update box
            success, cornerBboxes[i] = cornerTrackerList[i].update(img)

            # Update successful
            if success:
                drawBox(img, cornerBboxes[i])
                
            # Unsuccessful
            else:
                cornerBboxes[i] = 0

        f.write(str(cornerBboxes) + "\n")
    
    # grab every 10th frame to speed up testing
    for i in range(9):
        success, img = cap.read()
        if not success:
            break
    

cap.release()
f.close()
cv2.destroyAllWindows()