import cv2 as cv
import csv
import numpy as np
from detection import detect
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animationLib
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")

team1Color = (255,0,54)
team2Color = (70,126,255)

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

# Given locations for the corners of the field, computes a matrix M which is able to translate relative positions from pixel to field coordinates
# Input: 
# corner_bounding_boxes: Bounding boxes associated with the corners of the field of form ["top left", "bottom left", "bottom right", "top right"] where for each corner we record [x, y, width, height]
# Outputs:
# M: Perspective transformation matrix which translates from pixel to field coordinates
# source: 2d array with x and y coordinates of each of the 4 corners (middle location, not bounding boxes)

# use object detection to find players 
def detectionSelection(img, game):

    # Get source array
    source = np.float32([[0,0],[0,0],[0,0],[0,0]])
    for i, corner_box in enumerate(game.corner_bounding_boxes):
        middleCoords = getMiddleCoords(corner_box)
        source[i][0] = middleCoords[0]
        source[i][1] = middleCoords[1]

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

# This function takes a frame and some text and adds the text to the top of the frame so we can convey instructions to the user
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

# This is a function which modifies the image so that it asks the user to input which team a particular player is on (identified by their ID 'player_number')
# Note: This function does not actually retrieve this value, it simply modifies the image to add the desired prompt to the top of the frame
def createPlayerNumberImage(img, player_number):
    # Text we want to display
    text = "What team is player " + str(player_number) + " on? (Enter 1 or 2)"
    # Generate Image
    img_copy = displayInstructions(img, text)

    return img_copy

# This is a function for prompting the user to enter the number of players that are on the field at a particular frame (img)
def getPlayerCount(img):
    # Text we want to display
    text = "Enter the number of players on the field"
    # Generate image
    image_copy = displayInstructions(img, text)

    cv.namedWindow("Player Count", cv.WINDOW_NORMAL)
    cv.imshow("Player Count", image_copy)

    # Wait for input
    player_count_str = ''
    while True:
        key = cv.waitKey(0)
        if key == 13:  # Enter key pressed
            break
        elif key >= 48 and key <= 57:  # Only accept digits 0-9
            player_count_str += chr(key)

    # Destroy the window
    cv.destroyWindow('Player Count')
    player_count = int(player_count_str)

    return player_count

def redetectPlayers(img, game, redetect_all=False):
    new_player_bounding_boxes = detectionSelection(img, game)

    num_detected_players = len(new_player_bounding_boxes)
    num_tracked_players = len(game.players_on_field)
    

    if redetect_all: # We enter this conditional if we lost a player somewhere
        # use detection, but preserve unique IDs

        players_to_update = []
        already_matched_players = []
        for index, detected_player_bbox in enumerate(new_player_bounding_boxes):
            if index >= game.max_players:
                print("Too many players detected")
                break
            detected_player_location = getBottomMiddleCoords(detected_player_bbox)

            old_locations_dif = []
            for old_player_id in game.players_on_field:
                old_player_bbox = game.all_players[old_player_id].getBoundingBox()
                if not old_player_id in already_matched_players:
                    old_locations_dif.append([old_player_id, math.dist(getBottomMiddleCoords(old_player_bbox), detected_player_location)])
                else:
                    old_locations_dif.append([old_player_id, sys.maxsize])
            smallest_dif = min(old_locations_dif, key = lambda p: p[1])
            if smallest_dif[1] != sys.maxsize:
                closest_player_id = smallest_dif[0]
                players_to_update.append([closest_player_id, detected_player_bbox])
            else:
                # add detected_player
                # TO-DO: choose player that is closest
                for player in game.all_players:
                    if not player.id in game.players_on_field:
                        player.updateBoundingBox(detected_player_bbox)
                        game.addPlayerToField(player.id, img)
                        break 
        game.removeAllPlayersFromField(img)
        for player_id, detected_player in players_to_update:
            game.all_players[player_id].updateBoundingBox(detected_player)
            game.addPlayerToField(player_id, img)
        game.updatePlayerMultitracker(img)
    else:
        detected_boxes_to_add = []

        tracked_player_boxes_to_match = []
        for player_id in game.players_on_field:
            tracked_player_boxes_to_match.append([player_id, game.all_players[player_id].getBoundingBox()])
        
        for index, detected_player_bbox in enumerate(new_player_bounding_boxes):
            if len(tracked_player_boxes_to_match) > 0:
                detected_player_location = getMiddleCoords(detected_player_bbox)

                old_locations_dif = []
                for tracked_player in tracked_player_boxes_to_match:
                    tracked_player_id = tracked_player[0]
                    tracked_player_bbox = tracked_player[1]
                    old_locations_dif.append([tracked_player_id, math.dist(getMiddleCoords(tracked_player_bbox), detected_player_location)])
                smallest_dif = min(old_locations_dif, key = lambda p: p[1])
                closest_player_id = smallest_dif[0]
                closest_player_distance = smallest_dif[1]
                if closest_player_distance < detected_player_bbox[3]:
                    for i, tracked_player_info in enumerate(tracked_player_boxes_to_match):
                        if tracked_player_info[0] == closest_player_id:
                            tracked_player_boxes_to_match.pop(i)
                            break
                else:
                    detected_boxes_to_add.append(detected_player_bbox)
        for detected_box_to_add in detected_boxes_to_add:
            if len(tracked_player_boxes_to_match) > 0:
                # find closest remaining tracked player and update bbox to detected bbox
                detected_player_location = getMiddleCoords(detected_box_to_add)

                old_locations_dif = []
                for tracked_player in tracked_player_boxes_to_match:
                    tracked_player_id = tracked_player[0]
                    tracked_player_bbox = tracked_player[1]
                    old_locations_dif.append([tracked_player_id, math.dist(getMiddleCoords(tracked_player_bbox), detected_player_location)])
                smallest_dif = min(old_locations_dif, key = lambda p: p[1])
                closest_player_id = smallest_dif[0]
                
                game.all_players[closest_player_id].updateBoundingBox(detected_box_to_add)

                for i, tracked_player_info in enumerate(tracked_player_boxes_to_match):
                        if tracked_player_info[0] == closest_player_id:
                            tracked_player_boxes_to_match.pop(i)
                            break
            else:
                # assign to player not on field and add to field
                for player in game.all_players.values():
                    if not player.id in game.players_on_field:
                        player.updateBoundingBox(detected_box_to_add)
                        game.addPlayerToField(player.id, img)
                        break 
        # for remaining_tracked_player in tracked_player_boxes_to_match:
        #     player_id = remaining_tracked_player[0]
        #     game.removePlayerFromField(player_id, img)
        game.updatePlayerMultitracker(img)

def writePlayerBoundingBoxes(img, game):
    players_on_field = game.getPlayersOnField()
    for player_id in players_on_field:
        box = game.all_players[player_id].getBoundingBox()
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
        (w, h), _ = cv.getTextSize(str(player_id), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (int(box[0]), int(box[1])-20), (int(box[0])+w+10, int(box[1])), (0,0,0), -1)
        cv.putText(img, str(player_id), (int(box[0])+5, int(box[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def writePlayerBoundingBox(img, game, player_id):
    box = game.all_players[player_id].getBoundingBox()
    print("box:")
    print(box)
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
    (w, h), _ = cv.getTextSize(str(player_id), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(img, (int(box[0]), int(box[1])-20), (int(box[0])+w+10, int(box[1])), (0,0,0), -1)
    cv.putText(img, str(player_id), (int(box[0])+5, int(box[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def writeCornerBoundingBoxes(img, game):
    corner_bboxes = game.getCornerBoundingBoxes()
    for box in corner_bboxes:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
    return img

def writeCornerBoundingBox(img, box):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
    return img


class Game:
    def __init__(self, corner_bounding_boxes, max_players, img, destination_matrix):
        self.max_players = max_players
        self.num_players = 0
        # List of players on field by ID
        self.players_on_field = []
        # Dictionary to keep track of players via their associated ID's
        self.all_players = {}
        self.corner_multi_tracker = cv.legacy.MultiTracker_create()
        self.corner_bounding_boxes = corner_bounding_boxes
        for corner_bbox in corner_bounding_boxes:
            corner_tracker = cv.legacy.TrackerCSRT_create()
            self.corner_multi_tracker.add(corner_tracker, img, corner_bbox)
        self.player_multi_tracker = cv.legacy.MultiTracker_create()
        self.corner_bounding_boxes = corner_bounding_boxes
        self.destination_matrix = destination_matrix
        self.transformation_matrix = self.updateTransformationMatrix()

    # Return the players_on_field array
    def getPlayersOnField(self):
        return self.players_on_field
    
    # Return the corner bounding boxs
    def getCornerBoundingBoxes(self):
        return self.corner_bounding_boxes
           
    # Method to add a new Player to our Game class, takes in a bounding_box of the player and generates and ID to assign to the player
    def addPlayerToGame(self, bounding_box, add_to_field, img):
        self.num_players += 1
        id = self.num_players
        
        new_player = Player(id, None, bounding_box)
        self.all_players[id] = new_player

        if add_to_field == True:
            self.addPlayerToField(id, img)
        return id

    # Method to add a player as being recognized as currently on the field (we still keep track of that players information)
    def addPlayerToField(self, player_id, img):
        if player_id not in self.players_on_field:
            self.players_on_field.append(player_id)
            # add player to multitracker
            player_tracker = cv.legacy.TrackerCSRT_create()
            self.player_multi_tracker.add(player_tracker, img, self.all_players[player_id].getBoundingBox())
    
    # Method to remove a player from being recognized as on the field (we still keep track of that players information)
    def removePlayerFromField(self, player_id, img):
        for i, id in enumerate(self.players_on_field):
            if id == player_id:
                self.players_on_field.pop(i)
                self.updatePlayerMultitracker(img)
                break
    
    # Remove every player from being recognized as on the field and update the multi-tracker with img
    def removeAllPlayersFromField(self, img):
        self.players_on_field = []
        self.updatePlayerMultitracker(img)
    
    # Updates our player multi-tracker with a new image
    def updatePlayerMultitracker(self, img):
        new_player_multi_tracker = cv.legacy.MultiTracker_create()
        for player_id in self.players_on_field:
            player_bbox = self.all_players[player_id].getBoundingBox()
            new_player_tracker = cv.legacy.TrackerCSRT_create()
            new_player_multi_tracker.add(new_player_tracker, img, player_bbox)
        self.player_multi_tracker = new_player_multi_tracker
    
    # Take in image (next frame), generate player bboxs and corners using our mutitrackers (take in an image and array of bboxes)
    def updateCorners(self, img):
        # updates multitracker
        success, updated_corner_bounding_boxes = self.corner_multi_tracker.update(img)
        if success:
            self.corner_bounding_boxes = updated_corner_bounding_boxes
            self.updateTransformationMatrix()

        return success
    
    def updatePlayers(self, img):
        success, updated_player_bounding_boxes = self.player_multi_tracker.update(img)
        if success:
            for i in range(len(updated_player_bounding_boxes)):
                updated_bounding_box = updated_player_bounding_boxes[i]
                player_id = self.players_on_field[i]
                self.all_players[player_id].updateBoundingBox(updated_bounding_box)

        return success
    
    def getAllPlayers(self):
        return self.all_players
    
    def addToAllPlayerCoordinateHistories(self):
        for player in self.all_players.values():
            if player.id in self.players_on_field:
                bottom_middle_coords = getBottomMiddleCoords(player.getBoundingBox())
                x_coord = bottom_middle_coords[0]
                y_coord = bottom_middle_coords[1]
                input_array = np.float32([[[x_coord,y_coord]]])
                output_array = cv.perspectiveTransform(input_array, self.transformation_matrix)
                transformed_coordinates = output_array[0][0]
                transformed_x_value = transformed_coordinates[0]
                transformed_y_value = transformed_coordinates[1]
                player.addToCoordinateHistory([transformed_x_value, transformed_y_value])
            else:
                player.addToCoordinateHistory([])
    
    def updateTransformationMatrix(self):
        # initialize empty source array
        source = np.float32([[0,0],[0,0],[0,0],[0,0]])
        for i, corner_box in enumerate(self.corner_bounding_boxes):
            middleCoords = getMiddleCoords(corner_box)
            # Update source array
            source[i][0] = middleCoords[0]
            source[i][1] = middleCoords[1]
        # This opencv function returns a matrix which translates our pixel coordinates into relative field coordinates
        # More specifically, provides a matrix M such that source*M = destination_matrix
        self.transformation_matrix = cv.getPerspectiveTransform(source, self.destination_matrix)
    
# Player class to store information on 
class Player:
    def __init__(self, id, team, bounding_box):
        self.id = id  # Unique identifier for the player
        self.team = team  # Team number (1 or 2 or None)
        self.bounding_box = bounding_box  # Bounding box coordinates [x, y, width, height]
        self.translated_coordinate_history = []
        self.smoothed_history = []
    
    def addToCoordinateHistory(self, translated_coordinate):
        self.translated_coordinate_history.append(translated_coordinate)

    def getCoordinateHistory(self):
        return self.translated_coordinate_history
    
    def setSmoothedHistory(self, new_smoothed_history):
        self.smoothed_history = new_smoothed_history

    def getSmoothedHistory(self):
        return self.smoothed_history

    def getBoundingBox(self):
        return self.bounding_box

    def updateBoundingBox(self, new_bounding_box):
        self.bounding_box = new_bounding_box
    
    def updateTeam(self, new_team):
        self.team = new_team
        
    def getTeam(self):
        return self.team

def main():
    # Name of mp4 with frisbee film
    file_name = 'frisbee.mp4'

    # Load the video
    cap = cv.VideoCapture(file_name)

    # This line reads the first frame of our video and returns | ret: Boolean which is set to TRUE if frame is successfully read, FALSE if not | img: First frame from the video
    ret, img = cap.read()
    if not ret:
        print("Failed to read frame from video source. Exiting...")
        exit()


    # lists for storing information about players and corners 
    corner_bounding_boxes = []
    
    cornerNames = ["top left", "bottom left", "bottom right", "top right"]
    # # These are the coordinates of the bounding boxes for the specific test frisbee film we are using (need to be changed depending on the video that is being used)
    # corner_bounding_boxes = [(1189, 676, 11, 15), (0, 1739, 26, 30), (3513, 1662, 27, 37), (2294, 676, 21, 17)]

    cv.namedWindow("Corner MultiTracker", cv.WINDOW_NORMAL)
    for j in range(4):
        instruction_text = 'Draw a box around the ' + cornerNames[j] + ' corner, then press ENTER'
        display_img = displayInstructions(img, instruction_text)
        box = cv.selectROI('Corner MultiTracker', display_img, False, printNotice=False)
        corner_bounding_boxes.append(box)
        # this should maybe be a function later
        img = writeCornerBoundingBox(img, box)
    
    player_count = getPlayerCount(img)
    cv.destroyWindow("Corner MultiTracker")

    # specifies the four corners that we located
    destination_matrix = np.float32([[0,20],[0,90],[40,90],[40,20]])

    game = Game(corner_bounding_boxes, player_count, img, destination_matrix)

    detected_players_bounding_boxes = detectionSelection(img, game)
    for detected_player_bbox in detected_players_bounding_boxes:
        # add player to game and field
        game.addPlayerToGame(detected_player_bbox, True, img)

    # write all the boxes for detected players
    img = writePlayerBoundingBoxes(img, game)

    # Have user select any players that were not found by object detection 
    cv.namedWindow('Draw a box around any players that don\'t currently have a box.', cv.WINDOW_NORMAL)

    for i in range(len(game.getPlayersOnField()), game.max_players):
        print("Select player " + str(i+1))
        bbox = cv.selectROI('Draw a box around any players that don\'t currently have a box.', img, False, printNotice=False)
        while (bbox[2] == 0 or bbox[3] == 0):
            bbox = cv.selectROI('Draw a box around any players that don\'t currently have a box.', img, False, printNotice=False)
        # add player to game and field
        player_id = game.addPlayerToGame(bbox, True, img)
        print("id: " + str(player_id))
        writePlayerBoundingBox(img, game, player_id)
    
    cv.destroyWindow('Draw a box around any players that don\'t currently have a box.')

    cv.namedWindow("Identify teams.", cv.WINDOW_NORMAL)

    players_on_field = game.getPlayersOnField()
    for player_id in players_on_field:
        cur_image = createPlayerNumberImage(img, player_id)
        cv.imshow("Identify teams.", cur_image)

        # Wait for key press
        key = cv.waitKey(0)

        # Record team based on key press
        if key == ord('1'):
            game.all_players[player_id].updateTeam(1)
        elif key == ord('2'):
            game.all_players[player_id].updateTeam(2)

    cv.destroyWindow('Identify teams.')
    print("Beginning tracking -------------------------------------------------------------------------")

    # ==================== PLAYER/CORNER TRACKING ======================================
 
    counter = 0
    # Loop through video
    cv.namedWindow("Tracking...", cv.WINDOW_NORMAL)
    while cap.isOpened():
        success, img = cap.read()
        counter += 1
        if not success:
            break

        # update tracking for corners
        success = game.updateCorners(img)
        # If tracking was lost, select new ROI of corner
        if (not success):
            print("Tracking of the corners was lost! :(")

        # ==================== PLAYER TRACKING ======================================
        # re detect players every x frames
        # update tracking for players
        success = game.updatePlayers(img)

        # If tracking was lost, run detection again 
        if not success:
            print("Tracking of player was lost!")
            redetectPlayers(img, game, redetect_all=True)
            success = game.updatePlayers(img)
        if not success:
            print("Full redection failed :(")

        img = writePlayerBoundingBoxes(img, game)

        game.addToAllPlayerCoordinateHistories()  

        cv.imshow("Tracking...", img)

        # Exit if ESC pressed
        k = cv.waitKey(1) & 0xff
        if k == 27 : break
        
        # check for routine redetection
        if counter >= 16:
            redetectPlayers(img, game)
            img = writePlayerBoundingBoxes(img, game)
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
    cv.destroyAllWindows()

    animateGame(game)

if __name__ == "__main__":
    main()

    
#create the frisbee field - 110 x 40
def generate_field() :
    field = patches.Rectangle((0, 0), 110.0, 40.0, linewidth=2, edgecolor='white', facecolor='green', zorder=0)
    #initialize figure and axis data
    fig, ax = plt.subplots(1, figsize=(11, 4))
    ax.add_patch(field)
    #add field lines
    ax.axvline(x=20.0, color="white", zorder=1)
    ax.axvline(x=90.0, color="white",zorder=1)
    #add horizontal lines to give axis context
    ax.axhline(y=0.0, color="white",zorder=1)
    ax.axhline(y=40.0, color="white",zorder=1)
    plt.axis('off')

    #creating scatter plots for the players? Maybe something we want to do
    ax.scatter([], [], c= '#FF0036', label = 'Team 1', zorder=2)
    ax.scatter([], [], c= '#467EFF', label = 'Team 2', zorder=2)
    # ax.scatter([], [], c='white' , label = 'Disc', zorder=2)
    ax.legend(loc='upper right')

    return fig, ax


def animateGame(game):
    players_dictionary = game.getAllPlayers()

    for player_id, player in players_dictionary.items():
        coordinate_history = player.getCoordinateHistory()
        num_frames = len(coordinate_history)
        smoothed_data = [[] for _ in range(num_frames)]
        for frame_index in range(num_frames):
            x_coord, y_coord = coordinate_history[frame_index][0], coordinate_history[frame_index][1]
            smoothed_data[frame_index] = savgol_filter([x_coord, y_coord], 10, 3)
        
        player.setSmoothedHistory(smoothed_data)

    # plot static graph
    fig, ax = generate_field()

    player_lines = {}
    for player_id, player in players_dictionary.items():
        team = player.getTeam()
        if team == 1:
            color = '#FF0036'
        else:
            color = '#467EFF'
        player_lines[player_id], = ax.plot([], [], color=color, marker='o')

    # Animation function
    def update(frame, player_lines):
        for player_id, player_line in player_lines.items():
            smoothed_history = players_dictionary[player_id].getSmoothedHistory()
            if frame < len(smoothed_history):
                x_coord, y_coord = smoothed_history[frame]
                player_line.set_data(x_coord, y_coord)
        return list(player_lines.values())
    
    animation = animationLib.FuncAnimation(fig, update, frames=len(players_dictionary[0].getSmoothedHistory()), interval=50, blit=True)
    writer = animationLib.FFMpegWriter(fps=8, metadata=dict(artist='Jack_and_Ethan'), bitrate=800)
    animation.save("frisbeeAnimation.mp4", writer=writer)

    print("Animation complete.")

    plt.close()
    