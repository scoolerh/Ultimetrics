import cv2 as cv
import csv
import numpy as np
from detection import detect
import math
import sys
import tkinter as tk
from tkinter import simpledialog

""" def getPlayerCount():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    player_count = simpledialog.askinteger("Player Count", "Enter the number of players on the field:", parent=root, minvalue=1)
    return player_count """

player_bounding_boxes = []
playerMultiTracker = None
kalmanFilters = []

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

# FOR COMPUTING Y COORDINATE WHY DO WE DO + FOR BOTTOMMID BUT - FOR MIDDLE

# Given a bounding box of a player, this computes the bottom-middle coordinates of the box. This is used to represent the location of the 'feet' of the player
# Input:
# box: Bounding box of player with 4 values [x-coordinate, y-coordinate, width, height]
# Output:
# [xCoord, yCoord] where these are the X and Y coordinates of the bottom middle location of the bounding box
def getBottomMiddleCoords(box):
    xCoord = (box[0]+(box[2]/2))
    yCoord = (box[1]-box[3])
    return [xCoord, yCoord]

# Given a bounding box of a player, this computes the middle coordinates of the box
# Input:
# box: Bounding box of player with 4 values [x-coordinate, y-coordinate, width, height]
# Output:
# [xCoord, yCoord] where these are the X and Y coordinates of the middle location of the bounding box
def getMiddleCoords(box):
    xCoord = (box[0]+(box[2]/2))
    yCoord = (box[1]-(box[3]/2))
    return [xCoord, yCoord]

# Function which creates a multi tracker for the corners of the field and gives us a matrix M to translate relative positions of screen coordinatest to field coordinates
# Inputs:
# corner_bounding_boxes: Bounding boxes associated with the corners of the field of form ["top left", "bottom left", "bottom right", "top right"] where for each corner we record [x, y, width, height]
# Outputs:
# corner_multi_tracker: Multi tracker for the corners of the field
# corner_colors: Array with the associated colors of the bounding boxes for tracking the corners
def instantiateCorners(corner_bounding_boxes, img):
    # Create corner multi tracker
    corner_multi_tracker = cv.legacy.MultiTracker_create()  
    for bbox in corner_bounding_boxes:
        tracker = cv.legacy.TrackerCSRT_create()
        corner_multi_tracker.add(tracker, img, bbox)

    M, source = getPerspectiveMatrix(corner_bounding_boxes)
    
    return corner_multi_tracker, M, source

# Given locations for the corners of the field, computes a matrix M which is able to translate relative positions from pixel to field coordinates
# Input: 
# corner_bounding_boxes: Bounding boxes associated with the corners of the field of form ["top left", "bottom left", "bottom right", "top right"] where for each corner we record [x, y, width, height]
# Outputs:
# M: Perspective transformation matrix which translates from pixel to field coordinates
# source: 2d array with x and y coordinates of each of the 4 corners (middle location, not bounding boxes)
def getPerspectiveMatrix(corner_bounding_boxes):
    # Initiliaze a source 2d array (will eventually contain corner location information)
    source = np.float32([[0,0],[0,0],[0,0],[0,0]])

    # Right now our destination has the order [bottom-left, top-left, top-right, bottom-right]
    # This 2d array represents the relative locations of the corners when scaled to have an origin at the top-left of the top endzone
    destination = np.float32([[0,20],[0,90],[40,90],[40,20]])

    # Update our source array to contain corner location information
    for i, corner_box in enumerate(corner_bounding_boxes):
        middleCoords = getMiddleCoords(corner_box)
        # Update source array
        source[i][0] = middleCoords[0]
        source[i][1] = middleCoords[1]
    # This opencv function returns a matrix which translates our pixel coordinates into relative field coordinates
    M = cv.getPerspectiveTransform(source, destination)

    return M, source

# use object detection to find players 
def detectionSelection(img, source):
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

def displayInstructions(original_img, text):
    # Make a copy of the original image to preserve the background
    img = original_img.copy()

    # Display instructions on the image window with larger font size and black color
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Increase font size
    font_color = (0, 0, 0)  # Black color
    font_thickness = 2

    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = 50

    cv.rectangle(img, (text_x-5, text_y+5), (text_x+text_size[0]+5, text_y-text_size[1]-5), (255,255,255), -1)
    cv.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return img

def createPlayerNumberImage(img, player_number):
    # Make a copy of the original image to avoid modifying it
    img_copy = img.copy()

    # Display prompt on the image window
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 0, 0)  # Black color
    font_thickness = 2
    text = "What team is player " + str(player_number) + " on? (Enter 1 or 2)"
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = 50

    cv.rectangle(img_copy, (text_x-5, text_y+5), (text_x+text_size[0]+5, text_y-text_size[1]-5), (255,255,255), -1)
    cv.putText(img_copy, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return img_copy

def main():
    global player_bounding_boxes
    global playerMultiTracker
    global kalmanFilters

    # Name of mp4 with frisbee film
    file_name = 'frisbee.mp4'

    # Load the video
    cap = cv.VideoCapture(file_name)

    # This line reads the first frame of our video and returns two values
    # ret: Boolean which is set to TRUE if frame is successfully read, FALSE if not
    # img: First frame from the video
    ret, img = cap.read()

    if not ret:
        print("Failed to read frame from video source. Exiting...")
        exit()

    # cv.namedWindow("Tracking...", cv.WINDOW_NORMAL)
    # cv.namedWindow("Identify teams in the terminal.", cv.WINDOW_NORMAL)
    # cv.namedWindow("Draw a box around any players that don\'t currently have a box.", cv.WINDOW_NORMAL)
    
    # create csv where we output computed player coordinates
    coordinates_filename = 'playercoordinates.csv'
    coordinates_file = open(coordinates_filename, "w", newline='')
    coordinates_file_writer = csv.writer(coordinates_file, delimiter=',')
    # create csv that contains the team (1 or 2) each player is on
    teams_file = open("teams.csv", "w", newline='')
    teams_file_writer = csv.writer(teams_file, delimiter=',')

    # ================= MATH CONVERSION SETUP ==========================================

    # lists for storing information about players and corners 
    player_bounding_boxes = []
    cornerNames = ["top left", "bottom left", "bottom right", "top right"]
    # These are the coordinates of the bounding boxes for the specific test frisbee film we are using (need to be changed depending on the video that is being used)
    corner_bounding_boxes = [(1189, 676, 11, 15), (0, 1739, 26, 30), (3513, 1662, 27, 37), (2294, 676, 21, 17)]

    # manually mark the bounding boxes of corners
    corner_bounding_boxes = []
    cv.namedWindow("Corner MultiTracker", cv.WINDOW_NORMAL)

    for j in range(4):
        instruction_text = 'Draw a box around the ' + cornerNames[j] + ' corner, then press ENTER'
        display_img = displayInstructions(img, instruction_text)
        box = cv.selectROI('Corner MultiTracker', display_img, False, printNotice=False)
        corner_bounding_boxes.append(box)
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, (0, 0, 0), 2, 1)
    
    #player_count = getPlayerCount()
    cv.destroyWindow("Corner MultiTracker")

    # Create a multi tracker for the corners and players 
    corner_multi_tracker, M, source = instantiateCorners(corner_bounding_boxes, img)

    
    #print(player_count)

    playerMultiTracker = cv.legacy.MultiTracker_create()
    player_bounding_boxes = detectionSelection(img, source)

    # Add player trackers to the multitracker
    for bbox in player_bounding_boxes:
        tracker = cv.legacy.TrackerCSRT_create()
        playerMultiTracker.add(tracker, img, bbox)

    # write the boxes on the image 
    for i, box in enumerate(player_bounding_boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
        (w, h), _ = cv.getTextSize(str(i+1), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (int(box[0]), int(box[1])-20), (int(box[0])+w+10, int(box[1])), (0,0,0), -1)
        cv.putText(img, str(i+1), (int(box[0])+5, int(box[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    print("Detection complete -------------------------------------------------------------------------")

    # have user select any players that were not found by object detection 
    cv.namedWindow('Draw a box around any players that don\'t currently have a box.', cv.WINDOW_NORMAL)
    for i in range(len(player_bounding_boxes), 14):
        print("Select player " + str(i+1))
        bbox = cv.selectROI('Draw a box around any players that don\'t currently have a box.', img, False, printNotice=False)
        while (bbox[2] == 0 or bbox[3] == 0):
            bbox = cv.selectROI('Draw a box around any players that don\'t currently have a box.', img, False, printNotice=False)
        player_bounding_boxes.append(bbox)

        tracker = cv.legacy.TrackerCSRT_create()
        playerMultiTracker.add(tracker, img, bbox)

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
        (w, h), _ = cv.getTextSize(str(i+1), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (int(bbox[0]), int(bbox[1])-20), (int(bbox[0])+w+10, int(bbox[1])), (0,0,0), -1)
        cv.putText(img, str(i+1), (int(bbox[0])+5, int(bbox[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv.destroyWindow('Draw a box around any players that don\'t currently have a box.')

    cv.namedWindow("Identify teams.", cv.WINDOW_NORMAL)
    teams = []

    for i in range(1, 15):
        cur_image = createPlayerNumberImage(img, i)
        cv.imshow("Identify teams.", cur_image)

        # Wait for key press
        key = cv.waitKey(0)

        # Record team based on key press
        if key == ord('1'):
            teams.append('1')
        elif key == ord('2'):
            teams.append('2')

    cv.destroyWindow('Identify teams.')
    teams_file_writer.writerows(teams)
    print("Beginning tracking -------------------------------------------------------------------------")

    # ==================== PLAYER/CORNER TRACKING ======================================
    kalmanFilters = []

    def redetectPlayers(redetectAll=False):
        global player_bounding_boxes
        global playerMultiTracker
        global kalmanFilters
        new_player_bounding_boxes = detectionSelection(img, source)

        numDetectedPlayers = len(new_player_bounding_boxes)
        numTrackedPlayers = len(player_bounding_boxes)

        for newPlayerBox in new_player_bounding_boxes:
            p1 = (int(newPlayerBox[0]), int(newPlayerBox[1]))
            p2 = (int(newPlayerBox[0] + newPlayerBox[2]), int(newPlayerBox[1] + newPlayerBox[3]))
            cv.rectangle(img, p1, p2, (255,0,0), 2, 1)
            
        if redetectAll:
            # use detection, but preserve unique IDs
            updatedplayer_bounding_boxes = [None] * numTrackedPlayers
            if numDetectedPlayers < numTrackedPlayers:
                updatedplayer_bounding_boxes = new_player_bounding_boxes
            else:
                for index, detected_player in enumerate(new_player_bounding_boxes):
                    detected_player_location = getBottomMiddleCoords(detected_player)

                    old_locations_dif = []
                    for old_bbox in player_bounding_boxes:
                        if old_bbox != None:
                            old_locations_dif.append(math.dist(getBottomMiddleCoords(old_bbox), detected_player_location))
                        else:
                            old_locations_dif.append(sys.maxint)
                    smallest_dif = min(newLocationsDif)
                    if smallest_dif != sys.maxint:
                        print("replacement")
                        closestIndex = np.argmin(newLocationsDif)
                        updatedplayer_bounding_boxes[closestIndex] = detected_player
                        player_bounding_boxes[closestIndex] = None
                    else:
                        updatedplayer_bounding_boxes.append(detected_player)
                
            player_bounding_boxes = updatedplayer_bounding_boxes

            playerMultiTracker = cv.legacy.MultiTracker_create()

            for bbox in player_bounding_boxes:
                tracker = cv.legacy.TrackerCSRT_create()
                playerMultiTracker.add(tracker, img, bbox)
        else:
            possibleDoubles = []
            detected_player_indices_to_delete = []
            # detected_player_indices_to_delete_matching_index = []
            for index, trackedPlayer in enumerate(player_bounding_boxes):
                oldLocation = getMiddleCoords(trackedPlayer)

                newLocationsDif = []
                for newBbox in new_player_bounding_boxes:
                    newLocationsDif.append(math.dist(getMiddleCoords(newBbox), oldLocation))
                if len(newLocationsDif) == 0:
                    print("what???")
                    print(player_bounding_boxes)
                closestVal = min(newLocationsDif)
                closestIndex = np.argmin(newLocationsDif)

                if closestIndex not in detected_player_indices_to_delete:
                    detected_player_indices_to_delete.append(closestIndex)
                    # detected_player_indices_to_delete_matching_index([index, closestVal])
                else:
                    # possible double
                    # if closestVal < detected_player_indices_to_delete_matching_index
                    possibleDoubles.append(index)

            detected_player_indices_to_delete.sort(reverse=True)
            for index_to_delete in detected_player_indices_to_delete:
                new_player_bounding_boxes.pop(index_to_delete)
            need_new_multitracker = False
            for detectedPlayer in new_player_bounding_boxes:
                if len(possibleDoubles) > 0:
                    need_new_multitracker = True
                    distinct_player_location = getMiddleCoords(detectedPlayer)
                    locationsDif = []
                    for double_pair in possibleDoubles:
                        locationsDif.append(math.dist(getMiddleCoords(player_bounding_boxes[double_pair]), distinct_player_location))
                    
                    bbox_index_to_delete = possibleDoubles[np.argmin(locationsDif)]
                    possibleDoubles.pop(np.argmin(locationsDif))

                    player_bounding_boxes[bbox_index_to_delete] = detectedPlayer
                else:
                    np.append(player_bounding_boxes, detectedPlayer)
                    # player_box_colors.append(randomColor())
                    tracker = cv.legacy.TrackerCSRT_create()
                    playerMultiTracker.add(tracker, img, detectedPlayer)
            if need_new_multitracker:
                playerMultiTracker = cv.legacy.MultiTracker_create()
                for bbox in player_bounding_boxes:
                    tracker = cv.legacy.TrackerCSRT_create()
                    playerMultiTracker.add(tracker, img, bbox)
 
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
            print("Tracking of the corner was lost!")

        M, source = getPerspectiveMatrix(corner_bounding_boxes)

        # ==================== PLAYER TRACKING ======================================
        # re detect players every x frames
        # update tracking for players
        success, updated_player_bounding_boxes = playerMultiTracker.update(img)

        # If tracking was lost, run detection again 
        if not success:
            print("Tracking was lost!")
            redetectPlayers(redetectAll=True)
            success, updated_player_bounding_boxes = playerMultiTracker.update(img)

        player_bounding_boxes = updated_player_bounding_boxes

        csvLine = []

        for i, newPlayerBox in enumerate(player_bounding_boxes):
            p1 = (int(newPlayerBox[0]), int(newPlayerBox[1]))
            p2 = (int(newPlayerBox[0] + newPlayerBox[2]), int(newPlayerBox[1] + newPlayerBox[3]))
            cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
            (w, h), _ = cv.getTextSize(str(i+1), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv.rectangle(img, (int(newPlayerBox[0]), int(newPlayerBox[1])-20), (int(newPlayerBox[0])+w+10, int(newPlayerBox[1])), (0,0,0), -1)
            cv.putText(img, str(i+1), (int(newPlayerBox[0])+5, int(newPlayerBox[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if not newPlayerBox[0] > 0 :
                csvLine.append(-1)
                csvLine.append(-1)
            else:
                bottomMiddleCoords = getBottomMiddleCoords(newPlayerBox)
                # convert field to rectangle and translate to yards
                convertedPlayerCoords = screen2fieldCoordinates(bottomMiddleCoords[0],bottomMiddleCoords[1], M)
                csvLine.append(convertedPlayerCoords[0])
                csvLine.append(convertedPlayerCoords[1])
                        
        coordinates_file_writer.writerow(csvLine)   

        cv.imshow("Tracking...", img)

        # Exit if ESC pressed
        k = cv.waitKey(1) & 0xff
        if k == 27 : break
        
        # check for routine redetection
        if counter >= 16:
            redetectPlayers()
            counter = 0

        # grab every 10th frame to speed up testing
        for i in range(4):
            success, img = cap.read()
            counter += 1
            if not success:
                break

    print("Tracking complete. -------------------------------------------------------------------------")

    # ======================= CLEANUP ==================================================

    cap.release()
    coordinates_file.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()