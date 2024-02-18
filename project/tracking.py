import cv2 as cv
import random
import csv
import numpy as np
from detection import detect
import math

player_bounding_boxes = None
playerMultiTracker = None
kalmanFilters = None

# Converts pixel coordinates to field coordinates in yards from top left
# Inputs:
# x_coord: x-coordinate of pixel to convert
# y_coord: y-coordinate of pixel to convert
# transformation_matrix: given transformation_matrix which is used for transformation
# Output:
# transformed_coordinates: the transformed coordinates in in yards
def screen2fieldCoordinates(x_coord, y_coord, transformation_matrix):
    input_array = np.float32([[[x_coord,y_coord]]])
    output_array = cv.perspectiveTransform(input_array, transformation_matrix)
    transformed_coordinates = output_array[0][0]
    return transformed_coordinates


# Produce a random color with certain restrictions to make it look better 
# Outputs:
# Red: Red color value (0-255)
# Green: Green color value (0-255)
# Blue: Blue color value (0-255)
def randomColor():
    red = random.randint(0,255)
    green = random.randint(0,255)
    blue = random.randint(0,255)
    while (((green >= 1.75*red) and (green >= 1.75*blue)) or (red%5 != 0 or blue%5 != 0 or green%5 != 0)):
        red = random.randint(0,255)
        green = random.randint(0,255)
        blue = random.randint(0,255)
    return (red,green,blue)


# Produce a random color with restricted blue values to indicate the player as a member of Offense
# Outputs:
# Red: Red color value (0-255)
# Green: Green color value (0-255)
# Blue: Blue color value (0-130)
def randomOffensiveColor():
    red = random.randint(0,255)
    green = random.randint(0,255)
    blue = random.randint(0,130)
    while (((green >= 1.75*red) and (green >= 1.75*blue)) or (red%5 != 0 or blue%5 != 0 or green%5 != 0)):
        red = random.randint(0,255)
        green = random.randint(0,255)
        blue = random.randint(0,130)
    return (red,green,blue)


# Produce a random color with restricted blue values to indicate the player as a member of Defense
# Outputs:
# Red: Red color value (0-255)
# Green: Green color value (0-255)
# Blue: Blue color value (130-255)
def randomDefensiveColor():
    red = random.randint(0,255)
    green = random.randint(0,255)
    blue = random.randint(130, 255)
    while (((green >= 1.75*red) and (green >= 1.75*blue)) or (red%5 != 0 or blue%5 != 0 or green%5 != 0)):
        red = random.randint(0,255)
        green = random.randint(0,255)
        blue = random.randint(130, 255)
    return (red,green,blue)

# Function which creates a multi tracker for the corners of the field
# Inputs:
# corner_bounding_boxes: Bounding boxes associated with the corners of the field
# Outputs:
# corner_multi_tracker: Multi tracker for the corners of the field
# corner_colors: Array with the associated colors of the bounding boxes for tracking the corners
def instantiateCorners(corner_bounding_boxes, img):
    # Create corner multi tracker
    corner_multi_tracker = cv.legacy.MultiTracker_create()  
    corner_colors = []
    for bbox in corner_bounding_boxes:
        corner_colors.append(randomColor())
        tracker = cv.legacy.TrackerCSRT_create()
        corner_multi_tracker.add(tracker, img, bbox)

    # NEED TO KNOW HOW THIS WORKS FOR COORDINATE TRANSFORMATION
    source = np.float32([[0,0],[0,0],[0,0],[0,0]])
    destination = np.float32([[0,20],[0,90],[40,90],[40,20]])
    # get corners to update the transformation matrix
    for i, cornerBox in enumerate(corner_bounding_boxes):
        # get middle of box coordinate
        xCoord = (cornerBox[0]+(cornerBox[2]/2))
        yCoord = (cornerBox[1]-(cornerBox[3]/2))
        # update source matrix
        source[i][0] = xCoord
        source[i][1] = yCoord
    M = cv.getPerspectiveTransform(source,destination)
    
    return corner_multi_tracker, corner_colors, M, source, destination

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


def main():
    global player_bounding_boxes
    global playerMultiTracker
    global kalmanFilters

    # Name of mp4 with frisbee film
    file_name = 'frisbee.mp4'

    # Load the vide0
    cap = cv.VideoCapture(file_name)

    # This line reads the first frame of our video and returns two values
    # ret: Boolean which is set to TRUE if frame is successfully read, FALSE if not
    # img: First frame from the video
    ret, img = cap.read()

    # Set filename of csv where we output computed player coordinates
    coordinates_filename = 'playercoordinates.csv'
    # Open coordinates file
    coordinates_file = open(coordinates_filename, "w", newline='')
    # Create csv writer for coordinates file
    coordinates_file_writer = csv.writer(coordinates_file, delimiter=',')

    # Set filename of csv where we output colors associated with players
    colors_filename = 'playercolors.csv'
    # Open colors file
    colors_file = open(colors_filename, "w", newline='')
    # Create csv write for colors file
    colors_file_writer = csv.writer(colors_file, delimiter=',')

    # ================= MATH CONVERSION SETUP ==========================================

    # HOW IS THIS USED????

    # DONT THINK WE NEED THIS
    # # Instantiate corner trackers
    # cornerTrackerList = []
    # cornerNames = ["top left", "bottom left", "bottom right", "top right"]
    # for i in range(4): 
    #     tracker = cv.legacy.TrackerCSRT_create()
    #     cornerTrackerList.append(tracker)

    # lists for storing information about players and corners 
    player_bounding_boxes = []
    player_box_colors = []
    # These are the coordinates of the bounding boxes for the specific test frisbee film we are using (need to be changed depending on the video that is being used)
    corner_bounding_boxes = [(1189, 676, 11, 15), (0, 1739, 26, 30), (3513, 1662, 27, 37), (2294, 676, 21, 17)]

    # Create a multi tracker for the corners and keep track of the associated colors for the bounding boxes
    corner_multi_tracker, corner_colors, M, source, destination = instantiateCorners(corner_bounding_boxes, img)



    playerMultiTracker = cv.legacy.MultiTracker_create()


    # for han: 
    # corner_bounding_boxes = [(1307, 256, 22, 25), (22, 1535, 27, 30), (3580, 1577, 36, 50), (2150, 260, 33, 27)]

    # This is the code we initially had to manually mark the bounding boxes of corners
    # """ print("Please mark a box around each corner.")
    # for j in range(4):
    #     print('Draw a box around the ' + cornerNames[j] + ' corner.')
    #     cv.resize(img, (960, 540))
    #     cornerBbox = cv.selectROI('Corner MultiTracker', img, False, printNotice=False)
    #     corner_bounding_boxes.append(cornerBbox)
    #     drawBox(img,cornerBbox) """


    # use object detection to find players 
    def detectionSelection():
        # take out part of the image that isn't the field
        height = img.shape[0]
        width = img.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array([source[0],source[1],source[2],source[3]])
        points = np.int32([points])
        cv.fillPoly(mask, points, (255))

        res = cv.bitwise_and(img,img,mask = mask)

        new_player_bounding_boxes = detect(res)
        return new_player_bounding_boxes

    player_bounding_boxes = detectionSelection()

    # add player trackers to the multitracker
    for bbox in player_bounding_boxes:
        player_box_colors.append(randomColor())
        tracker = cv.legacy.TrackerCSRT_create()
        playerMultiTracker.add(tracker, img, bbox)

    # write the boxes on the image 
    for i, box in enumerate(player_bounding_boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, player_box_colors[i], 2, 1)

    playersDetected = len(player_bounding_boxes)
    print("Detection complete: " + str(playersDetected) + " players found. ---------------------------------------------------------")

    # have user select any players that were not found by object detection 
    while len(player_bounding_boxes) < 14:
        # img = cv.resize(img, (1200, 900))
        bbox = cv.selectROI('Select any unmarked players.', img, False, printNotice=False)
        player_bounding_boxes.append(bbox)

        tracker = cv.legacy.TrackerCSRT_create()
        playerMultiTracker.add(tracker, img, bbox)

        newColor = randomColor()
        player_box_colors.append(newColor)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(img, p1, p2, newColor, 2, 1)
        print("Player found ------------------------------------------------------------------")

    cv.destroyWindow('Select any unmarked players.')
    playersDetected = len(player_bounding_boxes)
    print("HITL complete: " + str(playersDetected) + " players found. --------------------------------------------------------")

    colors_file_writer.writerows(player_box_colors)
    print("Beginning tracking -------------------------------------------------------------------------")

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

    def redetectPlayers(redetectAll=False):
            global player_bounding_boxes
            global playerMultiTracker
            global kalmanFilters
            new_player_bounding_boxes = detectionSelection()
            # playerColors = []

            numDetectedPlayers = len(new_player_bounding_boxes)
            numTrackedPlayers = len(player_bounding_boxes)
                
            if redetectAll:
                # originally numTrackedPlayers < numDetectedPlayers
                # trust detection, but preserve unique IDs

                updatedplayer_bounding_boxes = [None] * max(numDetectedPlayers, numTrackedPlayers)

                lastIndex = 0
                numPlayersMatched = 0
                for index, trackedPlayer in enumerate(player_bounding_boxes):
                    if numPlayersMatched >= numDetectedPlayers:
                        # just add tracked player regularly to updatedplayer_bounding_boxes
                        updatedplayer_bounding_boxes[index] = player_bounding_boxes[index]
                    else:
                        oldLocation = getBottomMiddleCoords(trackedPlayer)

                        newLocationsDif = []
                        for newBbox in new_player_bounding_boxes:
                            newLocationsDif.append(math.dist(getBottomMiddleCoords(newBbox), oldLocation))
                        closestIndex = np.argmin(newLocationsDif)
                        updatedplayer_bounding_boxes[index] = new_player_bounding_boxes[closestIndex]
                        numPlayersMatched += 1
                        new_player_bounding_boxes[closestIndex] = [999999999,999999999, 1, 1]

                    lastIndex = index
                lastIndex += 1
                while lastIndex < numDetectedPlayers:
                    # add all of the redetects that don't correspond to a already tracked player
                    # happens if numdetectedplayers > numtrackedplayers
                    toAdd = None
                    for ind, newBox in enumerate(new_player_bounding_boxes):
                        if newBox[0] != 999999999:
                            toAdd = newBox
                            new_player_bounding_boxes[ind] = [999999999,999999999, 1, 1]
                            break
                    updatedplayer_bounding_boxes[lastIndex] = toAdd
                    player_box_colors.append(randomColor())
                    lastIndex += 1
                

                player_bounding_boxes = updatedplayer_bounding_boxes

                playerMultiTracker = cv.legacy.MultiTracker_create()

                for bbox in player_bounding_boxes:
                    tracker = cv.legacy.TrackerCSRT_create()
                    playerMultiTracker.add(tracker, img, bbox)

            elif numTrackedPlayers >= numDetectedPlayers:
                # trust tracking, don't use detection
                # check if one player stole box from another player
                pass

            # Update Kalman filters array with new filters if necessary
            if len(kalmanFilters) != len(player_bounding_boxes):
                kalmanFilters = []
                for _ in range(len(player_bounding_boxes)):
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


    counter = 0
    # Loop through video
    while cap.isOpened():
        success, img = cap.read()
        counter += 1
        if not success:
            break

        # update tracking for corners
        success, corner_bounding_boxes = corner_multi_tracker.update(img)
        # If tracking was lost, select new ROI of corner
        if (not success):
            print("Tracking of the " + str(cornerNames[i]) + " corner was lost!")

        for i, newCornerBox in enumerate(corner_bounding_boxes):
            p1 = (int(newCornerBox[0]), int(newCornerBox[1]))
            p2 = (int(newCornerBox[0] + newCornerBox[2]), int(newCornerBox[1] + newCornerBox[3]))
            cv.rectangle(img, p1, p2, corner_colors[i], 2, 1)
            # get middle of box coordinate
            middleCoords = getMiddleCoords(newCornerBox)
            xCoord = middleCoords[0]
            yCoord = middleCoords[1]
            # update source matrix
            source[i][0] = xCoord
            source[i][1] = yCoord
        # update transformation matrix
        M = cv.getPerspectiveTransform(source,destination)

        # ==================== PLAYER TRACKING ======================================
        # re detect players every x frames
        if counter >= 30:
            redetectPlayers(redetectAll=True)
            counter = 0
            # player_bounding_boxes = detectionSelection()
            # player_box_colors = []
            # playerMultiTracker = cv.legacy.MultiTracker_create()

            # for bbox in player_bounding_boxes:
            #     player_box_colors.append(randomColor())
            #     tracker = cv.legacy.TrackerCSRT_create()
            #     playerMultiTracker.add(tracker, img, bbox)
        else:
            # update tracking for players
            success, player_bounding_boxes = playerMultiTracker.update(img)

            # If tracking was lost, run detection again 
            if (not success):
                print("Player was lost!")
                redetectPlayers(redetectAll=True)
                counter = 0
                
                # Update Kalman filters array with new filters if necessary
                if len(kalmanFilters) != len(player_bounding_boxes):
                    kalmanFilters = []
                    for _ in range(len(player_bounding_boxes)):
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

            # else:
            #     # Loop through all players
            #     for i, bbox in enumerate(player_bounding_boxes):
            #         # Get the middle coordinates of the bounding box
            #         xCoord = bbox[0] + bbox[2] / 2
            #         yCoord = bbox[1] + bbox[3] / 2
            #         measurement = np.array([[xCoord], [yCoord]], dtype=np.float32)

                    # # Predict the next state using Kalman filter
                    # prediction = kalmanFilters[i].predict()

                    # # Update the predicted position using information from the CSRT tracker
                    # kalmanFilters[i].statePre[0] = xCoord
                    # kalmanFilters[i].statePre[1] = yCoord

                    # # Correct the Kalman filter using the measured position
                    # kalmanFilters[i].correct(measurement)

                    # # Get the corrected position
                    # corrected_position = kalmanFilters[i].statePost
                    # # Use the corrected position for further processing or visualization
                    # bbox[0]= corrected_position[0] - bbox[2] / 2
                    # bbox[1]= corrected_position[1] - bbox[3] / 2
                    


        csvLine = []

        for i, newPlayerBox in enumerate(player_bounding_boxes):
            p1 = (int(newPlayerBox[0]), int(newPlayerBox[1]))
            p2 = (int(newPlayerBox[0] + newPlayerBox[2]), int(newPlayerBox[1] + newPlayerBox[3]))
            cv.rectangle(img, p1, p2, player_box_colors[i], 2, 1)
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
                        
        coordinates_file_writer.writerow(csvLine)   

        # img = cv.resize(img, (1200, 900))
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

    print("Tracking complete. -----------------------------------------------------------------------")

    # ======================= CLEANUP ==================================================

    cap.release()
    coordinates_file.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()