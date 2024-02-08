import cv2 as cv
import random
import csv
import numpy as np
from detection import detect
import time

# ==================== INITIAL SETUP ===============================================

# import video
cap = cv.VideoCapture('frisbee.mp4')
static_cap = cv.VideoCapture('frisbee.mp4')
ret, img = cap.read()

# to store the initial photo of each player, for when that player is lost 
ret, static_image = static_cap.read()

# set up CSV file to write into 
f = open("playercoordinates.csv", "w", newline='')
csvWriter = csv.writer(f, delimiter=',')

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

# draws a box around a selected area in an img
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    while (g>100 and g>(r*2) and g>(b*2)):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

# instantiate corner trackers
cornerTrackerList = []
cornerNames = ["top left", "bottom left", "bottom right", "top right"]
for i in range(4): 
    tracker = cv.legacy.TrackerCSRT_create()
    cornerTrackerList.append(tracker)

# store the boxes drawn for each player 
cornerMultiTracker = cv.legacy.MultiTracker_create()
playerMultiTracker = cv.legacy.MultiTracker_create()

playerBboxes = []
playerImages = []
playerColors = []

cornerBboxes = []
cornerColors = []

# have user select the corners 
for j in range(4):
    print('Draw a box around the ' + cornerNames[j] + ' corner.')
    cornerBbox = cv.selectROI('Corner MultiTracker', img, False)
    cornerBboxes.append(cornerBbox)

# initialize corner multiTracker
for bbox in cornerBboxes:
    cornerColors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    tracker = cv.legacy.TrackerCSRT_create()
    cornerMultiTracker.add(tracker, img, bbox)

# recognizing corners for first frame
for i, cornerBox in enumerate(cornerBboxes):
    # get middle of box coordinate
    xCoord = (cornerBox[0]+(cornerBox[2]/2))
    yCoord = (cornerBox[1]-(cornerBox[3]/2))
    # update src matrix
    src[i][0] = xCoord
    src[i][1] = yCoord
    # update transformation matrix
M = cv.getPerspectiveTransform(src,dst)

# have user select the players 
def hitlSelection():
    while True:
        # draw bounding boxes over players
        bbox = cv.selectROI('Player MultiTracker', img)
        playerBboxes.append(bbox)
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break

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
    

# user can decide whether to manually select or whether to use object detection
# print("Would you like to use our object detection tool to try and find all of the players instead of inputing them all manually?")
# detection = input("Enter 'Y' for yes and 'N' for no. ")
# if (detection == "N" or detection == "n"): 
#     hitlSelection()
# else: 
#     playerBboxes = detectionSelection()
playerBboxes = detectionSelection()

# add player trackers to the multitracker
for bbox in playerBboxes:
    playerColors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    tracker = cv.legacy.TrackerCSRT_create()
    playerMultiTracker.add(tracker, img, bbox)

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
        playerColors = []
        playerMultiTracker = cv.legacy.MultiTracker_create()

        for bbox in playerBboxes:
            playerColors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
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
            playerColors = []
            playerMultiTracker = cv.legacy.MultiTracker_create()

            for bbox in playerBboxes:
                playerColors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                tracker = cv.legacy.TrackerCSRT_create()
                playerMultiTracker.add(tracker, img, bbox)

    csvLine = []

    for i, newPlayerBox in enumerate(playerBboxes):
        p1 = (int(newPlayerBox[0]), int(newPlayerBox[1]))
        p2 = (int(newPlayerBox[0] + newPlayerBox[2]), int(newPlayerBox[1] + newPlayerBox[3]))
        cv.rectangle(img, p1, p2, playerColors[i], 2, 1)
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

    cv.imshow("Corner MultiTracker", img)

    # Exit if ESC pressed
    k = cv.waitKey(1) & 0xff
    if k == 27 : break
    
    # grab every 10th frame to speed up testing
    for i in range(4):
        success, img = cap.read()
        counter += 1
        if not success:
            break

# ======================= CLEANUP ==================================================

cap.release()
static_cap.release()
f.close()
cv.destroyAllWindows()