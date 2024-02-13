import cv2 as cv
import random
import csv
import numpy as np
from detection import detect
import time

# ==================== INITIAL SETUP ===============================================

# import video
cap = cv.VideoCapture('frisbee.mp4')
ret, img = cap.read()

# set up CSV file to write into 
f = open("playercoordinates.csv", "w", newline='')
csvWriter = csv.writer(f, delimiter=',')
colors = open("playercolors.csv", "w", newline='')
colorWriter = csv.writer(colors, delimiter=',')

# ================= MATH CONVERSION SETUP ==========================================

# corner order: top left, bottom left, bottom right, top right
topLeftCoord = [0,0]
bottomLeftCoord = [0,0]
bottomRightCoord = [0,0]
topRightCoord = [0,0]

src = np.float32([[0,0],[0,0],[0,0],[0,0]])
dst = np.float32([[0,25],[0,95],[40,95],[40,25]])
M = None

#converts pixel coordinates to field coordinates in yards from top left
def screen2fieldCoordinates(x,y, transformation_matrix):
    inputArray = np.float32([[[x,y]]])
    outputArray = cv.perspectiveTransform(inputArray, transformation_matrix)
    outputArray = outputArray[0][0]
    return outputArray

# ===================== INITIAL PLAYER/CORNER LOCATION =============================

# produce a random color with certain restrictions to make it look better 
def randomColor():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    while (((g >= 1.75*r) and (g >= 1.75*b)) or (r%5 != 0 or b%5 != 0 or g%5 != 0)):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
    return (r,g,b)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = randomColor()
    cv.rectangle (img,(x,y), ((x+w), (y+h)), color, 3,1)
    return color

# instantiate corner trackers
cornerTrackerList = []
cornerNames = ["top left", "bottom left", "bottom right", "top right"]
for i in range(4): 
    tracker = cv.legacy.TrackerCSRT_create()
    cornerTrackerList.append(tracker)

# store the boxes drawn for each player 
cornerMultiTracker = cv.legacy.MultiTracker_create()
playerMultiTracker = cv.legacy.MultiTracker_create()

# lists for storing information about players and corners 
playerBboxes = []
playerBoxColors = []
cornerBboxes = []
cornerColors = []

# have user select the corners 
""" print("Please mark a box around each corner.")
for j in range(4):
    print('Draw a box around the ' + cornerNames[j] + ' corner.')
    cornerBbox = cv.selectROI('Corner MultiTracker', img, False, printNotice=False)
    cornerBboxes.append(cornerBbox)
    drawBox(img,cornerBbox) """
cornerBboxes = [(1189, 676, 11, 15), (0, 1739, 26, 30), (3513, 1662, 27, 37), (2294, 676, 21, 17)]
# for han: 
# cornerBboxes = [(1307, 256, 22, 25), (22, 1535, 27, 30), (3580, 1577, 36, 50), (2150, 260, 33, 27)]

# initialize corner multiTracker
for bbox in cornerBboxes:
    cornerColors.append(randomColor())
    tracker = cv.legacy.TrackerCSRT_create()
    cornerMultiTracker.add(tracker, img, bbox)

# get corners to update the transformation matrix
for i, cornerBox in enumerate(cornerBboxes):
    # get middle of box coordinate
    xCoord = (cornerBox[0]+(cornerBox[2]/2))
    yCoord = (cornerBox[1]-(cornerBox[3]/2))
    # update src matrix
    src[i][0] = xCoord
    src[i][1] = yCoord
M = cv.getPerspectiveTransform(src,dst)

print("Corners Found ----------------------------------------------------------------------------------------")

# use object detection to find players 
def detectionSelection():
    # take out part of the image that isn't the field
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([src[0],src[1],src[2],src[3]])
    points = np.int32([points])
    cv.fillPoly(mask, points, (255))

    res = cv.bitwise_and(img,img,mask = mask)

    newPlayerBboxes = detect(res)
    return newPlayerBboxes

playerBboxes = detectionSelection()

# add player trackers to the multitracker
for bbox in playerBboxes:
    playerBoxColors.append(randomColor())
    tracker = cv.legacy.TrackerCSRT_create()
    playerMultiTracker.add(tracker, img, bbox)

# write the boxes on the image 
for i, box in enumerate(playerBboxes):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    cv.rectangle(img, p1, p2, playerBoxColors[i], 2, 1)

playersDetected = len(playerBboxes)
print("Detection complete: " + str(playersDetected) + " players found. ---------------------------------------------------------")

# have user select any players that were not found by object detection 
while len(playerBboxes) < 14:
    bbox = cv.selectROI('Select any unmarked players.', img, False, printNotice=False)
    playerBboxes.append(bbox)
    color = drawBox(img,bbox)
    print("Player found ------------------------------------------------------------------")
    playerBoxColors.append(color)

cv.destroyWindow('Select any unmarked players.')
playersDetected = len(playerBboxes)
print("HITL complete: " + str(playersDetected) + " players found. --------------------------------------------------------")

# add player trackers to the multitracker
for i in range(playersDetected - 1, len(playerBboxes)):
    bbox = playerBboxes[i]
    tracker = cv.legacy.TrackerCSRT_create()
    playerMultiTracker.add(tracker, img, bbox)

colorWriter.writerows(playerBoxColors)
print("Beginning tracking -------------------------------------------------------------------------")

# ==================== PLAYER/CORNER TRACKING ======================================

counter = 0
# Loop through video
while cap.isOpened():
    success, img = cap.read()
    counter += 1
    if not success:
        break

    # update tracking for corners
    success, cornerBboxes = cornerMultiTracker.update(img)
    # If tracking was lost, select new ROI of corner
    if (not success):
        print("Tracking of the " + str(cornerNames[i]) + " corner was lost!")

    for i, newCornerBox in enumerate(cornerBboxes):
        p1 = (int(newCornerBox[0]), int(newCornerBox[1]))
        p2 = (int(newCornerBox[0] + newCornerBox[2]), int(newCornerBox[1] + newCornerBox[3]))
        cv.rectangle(img, p1, p2, cornerColors[i], 2, 1)
        # get middle of box coordinate
        xCoord = (newCornerBox[0]+(newCornerBox[2]/2))
        yCoord = (newCornerBox[1]-(newCornerBox[3]/2))
        # update src matrix
        src[i][0] = xCoord
        src[i][1] = yCoord
    # update transformation matrix
    M = cv.getPerspectiveTransform(src,dst)
            
    # re detect players every x frames
    if counter >= 30000:
        counter = 0
        playerBboxes = detectionSelection()
        playerBoxColors = []
        playerMultiTracker = cv.legacy.MultiTracker_create()

        for bbox in playerBboxes:
            playerBoxColors.append(randomColor())
            tracker = cv.legacy.TrackerCSRT_create()
            playerMultiTracker.add(tracker, img, bbox)
    else:
        # update tracking for players
        success, playerBboxes = playerMultiTracker.update(img)

        # If tracking was lost, run detection again 
        if (not success):
            print("Player was lost!")
            counter = 0
            playerBboxes = detectionSelection()
            playerBoxColors = []
            playerMultiTracker = cv.legacy.MultiTracker_create()

            for bbox in playerBboxes:
                playerBoxColors.append(randomColor())
                tracker = cv.legacy.TrackerCSRT_create()
                playerMultiTracker.add(tracker, img, bbox)

    csvLine = []

    for i, newPlayerBox in enumerate(playerBboxes):
        p1 = (int(newPlayerBox[0]), int(newPlayerBox[1]))
        p2 = (int(newPlayerBox[0] + newPlayerBox[2]), int(newPlayerBox[1] + newPlayerBox[3]))
        cv.rectangle(img, p1, p2, playerBoxColors[i], 2, 1)
        if not newPlayerBox[0] > 0 :
            csvLine.append(-1)
            csvLine.append(-1)
        else:
            # get bottom middle coordinate
            xCoord = (newPlayerBox[0]+(newPlayerBox[2]/2))
            yCoord = (newPlayerBox[1]+newPlayerBox[3])
            # convert field to rectangle and translate to yards
            convertedPlayerCoords = screen2fieldCoordinates(xCoord,yCoord, M)
            csvLine.append(convertedPlayerCoords[0])
            csvLine.append(convertedPlayerCoords[1])
                    
    csvWriter.writerow(csvLine)

    cv.imshow("Tracking in progress", img)

    # Exit if ESC pressed
    k = cv.waitKey(1) & 0xff
    if k == 27 : break
    
    # grab every 10th frame to speed up testing
    for i in range(4):
        success, img = cap.read()
        counter += 1
        if not success:
            break

print("Tracking complete. -----------------------------------------------------------------------")

# ======================= CLEANUP ==================================================

cap.release()
f.close()
cv.destroyAllWindows()