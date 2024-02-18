import cv2 as cv
import random
import csv
import numpy as np
from detection import detect
import math

# ==================== INITIAL SETUP ===============================================

# import video
cap = cv.VideoCapture('frisbee.mp4')
ret, img = cap.read()
# img = cv.resize(img, (1200, 900))

# set up CSV file to write into 
f = open("playercoordinates.csv", "w", newline='')
csvWriter = csv.writer(f, delimiter=',')
colors = open("playercolors.csv", "w", newline='')
colorWriter = csv.writer(colors, delimiter=',')
teams = open("teams.csv", "w", newline='')
teamWriter = csv.writer(teams, delimiter=',')

# ================= MATH CONVERSION SETUP ==========================================

# corner order: top left, bottom left, bottom right, top right
topLeftCoord = [0,0]
bottomLeftCoord = [0,0]
bottomRightCoord = [0,0]
topRightCoord = [0,0]

#coordinates fixed as of 2/14/2024
src = np.float32([[0,0],[0,0],[0,0],[0,0]])
dst = np.float32([[0,20],[0,90],[40,90],[40,20]])
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

# new color function to differentiate between the two teams 
def randomOffensiveColor():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,130)
    while (((g >= 1.75*r) and (g >= 1.75*b)) or (r%5 != 0 or b%5 != 0 or g%5 != 0)):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,130)
    return (r,g,b)

def randomDefensiveColor():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(130,255)
    while (((g >= 1.75*r) and (g >= 1.75*b)) or (r%5 != 0 or b%5 != 0 or g%5 != 0)):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(130,255)
    return (r,g,b)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = randomColor()
    cv.rectangle (img,(x,y), ((x+w), (y+h)), color, 3,1)
    return color

def getBottomMiddleCoords(box):
    xCoord = (box[0]+(box[2]/2))
    yCoord = (box[1]+box[3])
    return [xCoord, yCoord]
def getMiddleCoords(box):
    xCoord = (box[0]+(box[2]/2))
    yCoord = (box[1]-(box[3]/2))
    return [xCoord, yCoord]
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
    cv.resize(img, (960, 540))
    cornerBbox = cv.selectROI('Corner MultiTracker', img, False, printNotice=False)
    cornerBboxes.append(cornerBbox)
    drawBox(img,cornerBbox) """
#cornerBboxes = [(1189, 676, 11, 15), (0, 1739, 26, 30), (3513, 1662, 27, 37), (2294, 676, 21, 17)]
# for han: 
cornerBboxes = [(1307, 256, 22, 25), (22, 1535, 27, 30), (3580, 1577, 36, 50), (2150, 260, 33, 27)]

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

# write the boxes on the image 
i = 0
for box in playerBboxes:
    i += 1
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    color = randomColor()
    playerBoxColors.append(color)
    (w, h), _ = cv.getTextSize(str(i), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(img, p1, p2, color, 2, 1)
    cv.rectangle(img, (int(box[0]), int(box[1])-20), (int(box[0])+w+10, int(box[1])), color, -1)
    cv.putText(img, str(i), (int(box[0])+5, int(box[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

print("Detection complete: " + str(len(playerBboxes)) + " players found. ----------------------------------------------------------------")

# have user select any players that were not found by object detection 
while len(playerBboxes) < 14:
    img = cv.resize(img, (1200, 900))
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
#teamWriter.writerows(playerTeams)
print("Beginning tracking -----------------------------------------------------------------------------------")

# ==================== PLAYER/CORNER TRACKING ======================================
kalmanFilters = []

# Initialize Kalman Filters for all 14 players
for _ in range(14):
    kalman = cv.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)
    
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)
     
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32) * 0.03
    
    kalman.measurementNoiseCov = np.array([[1, 0],
                                            [0, 1]], dtype=np.float32) * 0.1
    
    kalman.statePre = np.zeros((4, 1), dtype=np.float32)
    kalman.statePost = np.zeros((4, 1), dtype=np.float32)
    kalmanFilters.append(kalman)

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
        middleCoords = getMiddleCoords(newCornerBox)
        xCoord = middleCoords[0]
        yCoord = middleCoords[1]
        # update src matrix
        src[i][0] = xCoord
        src[i][1] = yCoord
    # update transformation matrix
    M = cv.getPerspectiveTransform(src,dst)

# ==================== PLAYER TRACKING ======================================
    def redetectPlayers(redetectAll=False):
        global playerBboxes
        global playerMultiTracker
        global kalmanFilters
        newPlayerBboxes = detectionSelection()
        # playerColors = []

        numDetectedPlayers = len(newPlayerBboxes)
        numTrackedPlayers = len(playerBboxes)
            
        if redetectAll:
            # originally numTrackedPlayers < numDetectedPlayers
            # trust detection, but preserve unique IDs

            updatedPlayerBboxes = [None] * max(numDetectedPlayers, numTrackedPlayers)

            lastIndex = 0
            numPlayersMatched = 0
            for index, trackedPlayer in enumerate(playerBboxes):
                if numPlayersMatched >= numDetectedPlayers:
                    # just add tracked player regularly to updatedPlayerBboxes
                    updatedPlayerBboxes[index] = playerBboxes[index]
                else:
                    oldLocation = getBottomMiddleCoords(trackedPlayer)

                    newLocationsDif = []
                    for newBbox in newPlayerBboxes:
                        newLocationsDif.append(math.dist(getBottomMiddleCoords(newBbox), oldLocation))
                    closestIndex = np.argmin(newLocationsDif)
                    updatedPlayerBboxes[index] = newPlayerBboxes[closestIndex]
                    numPlayersMatched += 1
                    newPlayerBboxes[closestIndex] = [999999999,999999999, 1, 1]

                lastIndex = index
            lastIndex += 1
            while lastIndex < numDetectedPlayers:
                # add all of the redetects that don't correspond to a already tracked player
                # happens if numdetectedplayers > numtrackedplayers
                toAdd = None
                for ind, newBox in enumerate(newPlayerBboxes):
                    if newBox[0] != 999999999:
                        toAdd = newBox
                        newPlayerBboxes[ind] = [999999999,999999999, 1, 1]
                        break
                updatedPlayerBboxes[lastIndex] = toAdd
                playerBoxColors.append(randomColor())
                lastIndex += 1
            

            playerBboxes = updatedPlayerBboxes

            playerMultiTracker = cv.legacy.MultiTracker_create()

            for bbox in playerBboxes:
                tracker = cv.legacy.TrackerCSRT_create()
                playerMultiTracker.add(tracker, img, bbox)

        elif numTrackedPlayers >= numDetectedPlayers:
            # trust tracking, don't use detection
            # check if one player stole box from another player
            pass

        # Update Kalman filters array with new filters if necessary
        if len(kalmanFilters) != len(playerBboxes):
            kalmanFilters = []
            for _ in range(len(playerBboxes)):
                kalman = cv.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
                kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                    [0, 1, 0, 1],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], dtype=np.float32)
                kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0]], dtype=np.float32)
                kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], dtype=np.float32) * 0.03
                kalman.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], dtype=np.float32) * 0.1
                kalman.statePost = np.zeros((4, 1), dtype=np.float32)
                kalmanFilters.append(kalman)
    # re detect players every x frames
    if counter >= 30:
        redetectPlayers(redetectAll=True)
        counter = 0
        # playerBboxes = detectionSelection()
        # playerBoxColors = []
        # playerMultiTracker = cv.legacy.MultiTracker_create()

        # for bbox in playerBboxes:
        #     playerBoxColors.append(randomColor())
        #     tracker = cv.legacy.TrackerCSRT_create()
        #     playerMultiTracker.add(tracker, img, bbox)
    else:
        # update tracking for players
        success, playerBboxes = playerMultiTracker.update(img)

        # If tracking was lost, run detection again 
        if (not success):
            print("Player was lost!")
            redetectPlayers(redetectAll=True)
            counter = 0
            
            # Update Kalman filters array with new filters if necessary
            if len(kalmanFilters) != len(playerBboxes):
                kalmanFilters = []
                for _ in range(len(playerBboxes)):
                    kalman = cv.KalmanFilter(4, 2)  # 4 states (x, y, dx, dy), 2 measurements (x, y)
                    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                        [0, 1, 0, 1],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], dtype=np.float32)
                    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0]], dtype=np.float32)
                    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], dtype=np.float32) * 0.03
                    kalman.measurementNoiseCov = np.array([[1, 0],
                                                            [0, 1]], dtype=np.float32) * 0.1
                    kalman.statePost = np.zeros((4, 1), dtype=np.float32)
                    kalmanFilters.append(kalman)

        else:
            # Loop through all players
            for i, bbox in enumerate(playerBboxes):
                # Get the middle coordinates of the bounding box
                xCoord = bbox[0] + bbox[2] / 2
                yCoord = bbox[1] + bbox[3] / 2
                measurement = np.array([[xCoord], [yCoord]], dtype=np.float32)

                # Predict the next state using Kalman filter
                prediction = kalmanFilters[i].predict()

                # Update the predicted position using information from the CSRT tracker
                kalmanFilters[i].statePre[0] = xCoord
                kalmanFilters[i].statePre[1] = yCoord

                # # Correct the Kalman filter using the measured position
                # kalmanFilters[i].correct(measurement)

                # Get the corrected position
                corrected_position = kalmanFilters[i].statePost
                # Use the corrected position for further processing or visualization
                bbox[0]= corrected_position[0] - bbox[2] / 2
                bbox[1]= corrected_position[1] - bbox[3] / 2
                

                
    csvLine = []

    for i, newPlayerBox in enumerate(playerBboxes):
        p1 = (int(newPlayerBox[0]), int(newPlayerBox[1]))
        p2 = (int(newPlayerBox[0] + newPlayerBox[2]), int(newPlayerBox[1] + newPlayerBox[3]))
        (w, h), _ = cv.getTextSize(str(i+1), cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv.rectangle(img, p1, p2, playerBoxColors[i], 2, 1)
        img = cv.rectangle(img, (int(newPlayerBox[0]), int(newPlayerBox[1])-20), (int(newPlayerBox[0])+w+10, int(newPlayerBox[1])), playerBoxColors[i], -1)
        img = cv.putText(img, str(i + 1), (int(newPlayerBox[0]), int(newPlayerBox[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        if not newPlayerBox[0] > 0 :
            csvLine.append(-1)
            csvLine.append(-1)
        else:
            # get bottom middle coordinate
            bottomMiddleCoords = getBottomMiddleCoords(newPlayerBox)
            xCoord = bottomMiddleCoords[0]
            yCoord = bottomMiddleCoords[1]
            # convert field to rectangle and translate to yards
            convertedPlayerCoords = screen2fieldCoordinates(xCoord,yCoord, M)
            csvLine.append(convertedPlayerCoords[0])
            csvLine.append(convertedPlayerCoords[1])
                    
    csvWriter.writerow(csvLine)

    img = cv.resize(img, (1200, 900))
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

print("Tracking complete. -----------------------------------------------------------------------------------")

# ======================= CLEANUP ==================================================

cap.release()
f.close()
cv.destroyAllWindows()