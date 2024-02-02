import cv2 as cv
import random
import csv
import numpy as np

trackerList = []

# top left, bottom left, bottom right, top right

topLeftCoord = [0,0]
bottomLeftCoord = [0,0]
bottomRightCoord = [0,0]
topRightCoord = [0,0]

src = np.float32([
    [0,0], 
    [0,0],
    [0,0],
    [0,0]
])

dst = np.float32([
    [0,25],
    [0,95],
    [40,95],
    [40,25]
])

M = None
# M = cv.getPerspectiveTransform(src,dst)

lineCounter = 1

def screen2fieldCoordinates(x,y, transformation_matrix):
    #converts pixel coordinates to field coordinates in yards from top left
    inputArray = np.float32([[[x,y]]])
    # outputArray = np.float32([[[0, 0]]])
    outputArray = cv.perspectiveTransform(inputArray, transformation_matrix)
    outputArray = outputArray[0][0]
    # outputArray[0] = outputArray[0]/30.85
    # outputArray[1] = outputArray[1]/30.85
    return outputArray


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    r = random.randint(0,256)
    g = random.randint(0,256)
    b = random.randint(0,256)
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

for i in range(2): 
    tracker = cv.TrackerCSRT_create()
    trackerList.append(tracker)

cap = cv.VideoCapture('frisbee.mp4')
static_cap = cv.VideoCapture('frisbee.mp4')

ret, img = cap.read()

# Keeping track of players
ret, static_image = static_cap.read()

# set up variables to track player
bboxes = []
player_images = []

# set up variables to track corners
cornerTrackerList = []
cornerBboxes = []
corner = ["top left", "bottom left", "bottom right", "top right"]

f = open("playercoordinates.csv", "w", newline='')
csvWriter = csv.writer(f, delimiter=',')

#Instantiate corner trackers
for i in range(4): 
    tracker = cv.TrackerCSRT_create()
    cornerTrackerList.append(tracker)

#Select corners 
for j in range(4):
    print('Draw a box around the ' + corner[j] + ' corner.')
    cornerBbox = cv.selectROI('Select the ' + corner[j] + ' corner', img, False)
    cornerBboxes.append(cornerBbox)
    drawBox(img, cornerBbox)
    cornerTrackerList[j].init(img, cornerBboxes[j])
    cv.destroyWindow('Select the ' + corner[j] + ' corner')

# instantiate and select player trackers
for i in range(0, len(trackerList)):
    player = i + 1
    print('Select player ' + str(player))
    # bbox object contains [Top_Left_X, Top_Left_Y, Width, Height]
    bbox = cv.selectROI('Select player ' + str(player), img, False)
    bboxes.append(bbox)
    drawBox(img, bbox)

    # Captures images of each player
    cropped = static_image[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    player_images.append(cropped)

    trackerList[i].init(img, bboxes[i])
    cv.destroyWindow('Select player ' + str(player))

cv.setWindowTitle('Select player ' + str(len(trackerList) + 1), "Tracking Players...")
cv.destroyWindow('Select player ' + str(len(trackerList) + 1))

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    # update tracking for corners
    for i in range(0, len(cornerBboxes)):

        # Update box
        success, cornerBboxes[i] = cornerTrackerList[i].update(img)

        # If tracking was lost, select new ROI of player
        if (not success):
            print("Tracking of the " + str(corner[i]) + " corner was lost!")
            reSelectedCorner = False
            while not reSelectedCorner:
                cornerBbox = cv.selectROI('Reselect ' + str(corner[i]), img, False)

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
        csvLine.append(lineCounter)
        lineCounter += 1
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
        
        # f.write(str(bboxes) + "\n")
        
    
    # grab every 10th frame to speed up testing
    for i in range(9):
        success, img = cap.read()
        if not success:
            break

cap.release()
static_cap.release()
f.close()
cv.destroyAllWindows()