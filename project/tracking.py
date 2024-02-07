import cv2 as cv
import random
import csv
import numpy as np

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
# M = cv.getPerspectiveTransform(src,dst)

#converts pixel coordinates to field coordinates in yards from top left
def screen2fieldCoordinates(x,y, transformation_matrix):
    inputArray = np.float32([[[x,y]]])
    outputArray = cv.perspectiveTransform(inputArray, transformation_matrix)
    outputArray = outputArray[0][0]
    return outputArray

# ===================== INITIAL PLAYER/CORNER LOCATION =============================

# draws a randomly colored box around a selected area in an img
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    r = random.randint(0,256)
    g = random.randint(0,256)
    b = random.randint(0,256)
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

# instantiate player trackers, one for each player
trackerList = []
for i in range(2): 
    tracker = cv.TrackerCSRT_create()
    trackerList.append(tracker)

# instantiate corner trackers
cornerTrackerList = []
corner = ["top left", "bottom left", "bottom right", "top right"]
for i in range(4): 
    tracker = cv.TrackerCSRT_create()
    cornerTrackerList.append(tracker)

# store the boxes drawn for each player 
bboxes = []
player_images = []
cornerBboxes = []

# have user select the corners 
for j in range(4):
    print('Draw a box around the ' + corner[j] + ' corner.')
    cornerBbox = cv.selectROI('Select the ' + corner[j] + ' corner', img, False)
    cv.moveWindow('Select the ' + corner[j] + ' corner', 0, 0)
    cornerBboxes.append(cornerBbox)
    drawBox(img, cornerBbox)
    cornerTrackerList[j].init(img, cornerBboxes[j])
    cv.destroyWindow('Select the ' + corner[j] + ' corner')

cv.destroyAllWindows()

# have user select the players 
def hitlSelection():
    for i in range(0, len(trackerList)):
        player = i + 1
        # bbox object contains [Top_Left_X, Top_Left_Y, Width, Height]
        bbox = cv.selectROI('Select player ' + str(player), img, False)
        cv.moveWindow('Select player ' + str(player), 0, 0)
        bboxes.append(bbox)
        drawBox(img, bbox)

        # Captures images of each player
        cropped = static_image[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
        player_images.append(cropped)

        trackerList[i].init(img, bboxes[i])
        cv.destroyWindow('Select player ' + str(player))

    cv.setWindowTitle('Select player ' + str(len(trackerList) + 1), "Tracking Players...")
    cv.destroyWindow('Select player ' + str(len(trackerList) + 1))

# use object detection to find players 
def detectionSelection(): 
    print("This function is under construction....")

# user can decide whether to manually select or whether to use object detection
print("Would you like to use our object detection tool to try and find all of the players instead of inputing them all manually?")
detection = input("Enter 'Y' for yes and 'N' for no. ")
if (detection == "N" or detection == "n"): 
    hitlSelection()
else: 
    detectionSelection()

# ==================== PLAYER/CORNER TRACKING ======================================

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    # update tracking for corners
    for i in range(0, len(cornerBboxes)):

        success, cornerBboxes[i] = cornerTrackerList[i].update(img)

        # If tracking was lost, select new ROI of player
        if (not success):
            print("Tracking of the " + str(corner[i]) + " corner was lost!")
            reSelectedCorner = False
            while not reSelectedCorner:
                cornerBbox = cv.selectROI('Reselect ' + str(corner[i]), img, False)
                cv.moveWindow('Reselect ' + str(corner[i]), 0, 0)
                # If no coordinates were selected
                if cornerBbox != (0, 0, 0, 0):
                    reSelectedCorner = True

            # Insert new cornerBbox into list and continue tracking
            cornerBboxes[i] = cornerBbox
            new_tracker = cv.legacy.TrackerCSRT_create()
            cornerTrackerList[i] = new_tracker
            cornerTrackerList[i].init(img, cornerBboxes[i])
            cv.destroyWindow('Reselect ' + str(corner[i]))

        for k in range(len(cornerBboxes)):
            eachCorner = cornerBboxes[k]
            # get middle of box coordinate
            xCoord = (eachCorner[0]+(eachCorner[2]/2))
            yCoord = (eachCorner[1]-(eachCorner[3]/2))
            # update src matrix
            src[k][0] = xCoord
            src[k][1] = yCoord
            # update transformation matrix
            M = cv.getPerspectiveTransform(src,dst)
            
    # update tracking for players
    for i in range(0, len(bboxes)):

        if bboxes[i][0] != -1:
            success, bboxes[i] = trackerList[i].update(img)

            # If tracking was lost, select new ROI of player
            if (not success):
                print("Player was lost!")
                cv.imshow('Lost Player', player_images[i])
                bbox = cv.selectROI('Reselect Lost Player', img, False)
                cv.moveWindow('Reselect Lost Player', 10, 50)

                if bbox == (0, 0, 0, 0):
                    # they didn't reselect the player, so set the location to -1, which
                    # means don't draw player and don't prompt every frame to re-draw
                    bboxes[i] = (-1, -1, -1, -1)
                else:
                    bboxes[i] = bbox
                    new_tracker = cv.TrackerCSRT_create()
                    trackerList[i] = new_tracker
                    trackerList[i].init(img, bboxes[i])
                cv.destroyWindow('Reselect Lost Player')
                cv.destroyWindow('Lost Player')

        csvLine = []
        for player in bboxes:
            if player[0] == -1:
                csvLine.append(-1)
                csvLine.append(-1)
            else:
                # get bottom middle coordinate
                xCoord = (player[0]+(player[2]/2))
                yCoord = (player[1]-player[3])
                # convert field to rectangle and translate to yards
                convertedPlayerCoords = screen2fieldCoordinates(xCoord,yCoord, M)
                csvLine.append(convertedPlayerCoords[0])
                csvLine.append(convertedPlayerCoords[1])
                   
        csvWriter.writerow(csvLine)        
    
    # grab every 10th frame to speed up testing
    for i in range(9):
        success, img = cap.read()
        if not success:
            break

# ======================= CLEANUP ==================================================

cap.release()
static_cap.release()
f.close()
cv.destroyAllWindows()