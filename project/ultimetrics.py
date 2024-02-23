import cv2 as cv
import numpy as np
import yolov5
import math
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animationLib
import warnings
from scipy.signal import savgol_filter
import roboflow as Roboflow
from api_key import API_KEY_ROBOFLOW, PROJECT_NAME, VERSION
warnings.filterwarnings("ignore")

team1Color = (255,0,54)
team2Color = (70,126,255)
model = None

# Given a bounding box of a player, this computes the bottom-middle coordinates of the box. This is used to represent the location of the 'feet' of the player
# Input:
# box: Bounding box of player with 4 values [x-coordinate, y-coordinate, width, height]
# Output:
# [xCoord, yCoord] where these are the X and Y coordinates of the bottom middle location of the bounding box
def getBottomMiddleCoords(box):
    xCoord = (box[0]+(box[2]/2))
    yCoord = (box[1]+box[3])
    return [xCoord, yCoord]

# Given a bounding box of a player, this computes the middle coordinates of the box
# Input:
# box: Bounding box of player with 4 values [x-coordinate, y-coordinate, width, height]
# Output:
# [xCoord, yCoord] where these are the X and Y coordinates of the middle location of the bounding box
def getMiddleCoords(box):
    xCoord = (box[0]+(box[2]/2))
    yCoord = (box[1]+(box[3]/2))
    return [xCoord, yCoord]

# ============= OBJECT DETECTION ========================================

def load_model():
    global model
    if not model:
        # load model
        rf = Roboflow.Roboflow(api_key=API_KEY_ROBOFLOW)
        project = rf.workspace().project(PROJECT_NAME)
        model = project.version(VERSION).model

def detect(image):
    
    load_model()

    # perform inference
    results = model(image, size=640)

    # inference with test time augmentation
    results = model(image, augment=True)

    # parse results
    bboxes = []
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    boxes.tolist()

    for i in range(0, len(boxes)):
        x1 = round(boxes[i][0].item())
        y1 = round(boxes[i][1].item())
        width = round(boxes[i][2].item()) - x1
        height = round(boxes[i][3].item()) - y1
        bboxes.append((x1, y1, width, height))

    return bboxes

#=======================Pre-trained model + our own data==================================
    # response = model.predict(image, confidence=40, overlap=30).json()

    # bboxes = []
    # for item in response['predictions']:
    #     x = item['x']
    #     y = item['y']
    #     width = item['width']
    #     height = item['height']

    #     x_topleft = round(x - width / 2)
    #     y_topleft = round(y - height / 2)
    #     box_width = round(x + width / 2) - x_topleft
    #     box_height = round(y + height / 2) - y_topleft
    #     bbox = (x_topleft, y_topleft, box_width, box_height)
    #     bboxes.append(bbox)

    # # Convert bounding box to correct coordinates for tracking

    # return bboxes

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

# ================ PRINTING INSTRUCTIONS ================================

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
    text = "What team is player " + str(player_number) + " on? (Press 1 or 2 on your keyboard)"
    # Generate Image
    img_copy = displayInstructions(img, text)

    return img_copy

# This is a function for prompting the user to enter the number of players that are on the field at a particular frame (img)
def getPlayerCount(img):
    # Text we want to display
    text = "Type the number of players on the field, then press ENTER."
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

# =============== REDETECTION ===================================

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
                game.addNewDetectedPlayerToField(detected_player_bbox, img)
        game.removeAllPlayersFromField(img)
        for player_id, detected_player in players_to_update:
            game.all_players[player_id].updateBoundingBox(detected_player)
            game.addPlayerToField(player_id, img)
        game.updatePlayerMultitracker(img)
    else:
        detected_player_boxes_to_add = []

        for detected_player_bbox in new_player_bounding_boxes:
            found_match = False
            detected_player_location = getBottomMiddleCoords(detected_player_bbox)

            for tracked_player in game.all_players.values():
                if tracked_player.id in game.players_on_field:
                    tracked_player_bbox = tracked_player.getBoundingBox()
                    distance = math.dist(getBottomMiddleCoords(tracked_player_bbox), detected_player_location)
                    if distance < detected_player_bbox[2]:
                        # print("matched: " + str(distance))
                        found_match = True
                        break
            if not found_match:
                print("no match found")
                detected_player_boxes_to_add.append(detected_player_bbox)
        
        for detected_bbox in detected_player_boxes_to_add:
            if len(game.players_on_field) >= game.max_players:
                player_id_to_change = game.removeClosestTwoPlayers()
                game.all_players[player_id_to_change].updateBoundingBox(detected_bbox)
                game.addPlayerToField(player_id_to_change, img)
            else:
                game.addNewDetectedPlayerToField(detected_bbox, img)
            # game.addPlayerToGame(detected_bbox, True, img)
        game.updatePlayerMultitracker(img)

# ============== DRAWING BBOXES =======================================

def writePlayerBoundingBoxes(img, game):
    players_on_field = game.getPlayersOnField()
    for player_id in players_on_field:
        current_player = game.all_players[player_id]
        box = current_player.getBoundingBox()
        team = current_player.getTeam()
        if team == 1: 
            color = team1Color
        elif team == 2: 
            color = team2Color
        else: 
            color = (0,0,0)
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, color, 2, 1)
        (w, h), _ = cv.getTextSize(str(player_id), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (int(box[0]), int(box[1])-20), (int(box[0])+w+10, int(box[1])), color, -1)
        cv.putText(img, str(player_id), (int(box[0])+5, int(box[1])-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def writePlayerBoundingBox(img, game, player_id):
    box = game.all_players[player_id].getBoundingBox()
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

def writeCornerBoundingBox(img, box, color=(0,0,0)):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    cv.rectangle(img, p1, p2, color, 2, 1)
    return img

# ======================= ANIMATION ==================================================
 
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

    # Look at this for how we have changed our savgol filter
    for player_id, player in players_dictionary.items():
        coordinate_history = player.getCoordinateHistory()
        print(coordinate_history)
        x_coords = [frame[0] for frame in coordinate_history]
        y_coords = [frame[1] for frame in coordinate_history]
        smoothed_x = savgol_filter(x_coords, 10, 3)
        smoothed_y = savgol_filter(y_coords, 10, 3)
        smoothed_data = list(zip(smoothed_x, smoothed_y))
        player.setSmoothedHistory(smoothed_data)

    # for player_id, player in players_dictionary.items():
    #     coordinate_history = player.getCoordinateHistory()
    #     num_frames = len(coordinate_history)
    #     smoothed_data = [[] for _ in range(num_frames)]
    #     for frame_index in range(num_frames):
    #         x_coord, y_coord = coordinate_history[frame_index][0], coordinate_history[frame_index][1]
    #         smoothed_data[frame_index] = savgol_filter([x_coord, y_coord], 10, 3)
        
    #     player.setSmoothedHistory(smoothed_data)

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
    def update(frame):
        for player_id, player_line in player_lines.items():
            smoothed_history = players_dictionary[player_id].getSmoothedHistory()
            if frame < len(smoothed_history):
                x_coord, y_coord = smoothed_history[frame]
                player_line.set_data(x_coord, y_coord)
        return list(player_lines.values())
    
    animation = animationLib.FuncAnimation(fig, update, frames=len(players_dictionary[1].getSmoothedHistory()), interval=50, blit=True)
    writer = animationLib.FFMpegWriter(fps=8, metadata=dict(artist='Jack_and_Ethan'), bitrate=800)
    animation.save("frisbeeAnimation.mp4", writer=writer)

    print("Animation complete.")

    plt.close()

# ================ GAME CLASS ========================================

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

    def getPlayersOnField(self):
        return self.players_on_field
    
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
    
    def addNewDetectedPlayerToField(self, new_bbox, img):
        # add detected_player to a non-field player
        # TO-DO: choose non-field player that is closest
        final_player_id = None
        for player in self.all_players.values():
            if not player.id in self.players_on_field:
                player.updateBoundingBox(new_bbox)
                self.addPlayerToField(player.id, img)
                final_player_id = player.id
                break 
        return final_player_id
    # Method to remove a player from being recognized as on the field (we still keep track of that players information)
    def removePlayerFromField(self, player_id):
        for i, id in enumerate(self.players_on_field):
            if id == player_id:
                self.players_on_field.pop(i)
                break
                
    def removeAllPlayersFromField(self, img):
        self.players_on_field = []
        self.updatePlayerMultitracker(img)
    
    def removeClosestTwoPlayers(self):
        players = list(self.all_players.items())
        # closest_two_players = [distance, first_player_id, second_player_id]
        closest_two_players = [sys.maxsize, None, None]
        start_index = 1
        for player_id, player in players:
            if player_id in self.players_on_field:
                for i in range(start_index, len(players)):
                    compared_player_id, compared_player = players[i]
                    if compared_player_id in self.players_on_field:
                        distance = math.dist(getBottomMiddleCoords(player.getBoundingBox()), getBottomMiddleCoords(compared_player.getBoundingBox()))
                        if distance < closest_two_players[0] and distance != 0:
                            # found new closest two
                            closest_two_players[0] = distance
                            closest_two_players[1] = player_id
                            closest_two_players[2] = compared_player_id
            start_index += 1
        
        # TO-DO: optimize which one to choose
        # arbitrarily pick first player
        player_id_to_remove = closest_two_players[1]
        self.removePlayerFromField(player_id_to_remove)
        return player_id_to_remove
    
    def updatePlayerMultitracker(self, img):
        new_player_multi_tracker = cv.legacy.MultiTracker_create()
        for player_id in self.players_on_field:
            player_bbox = self.all_players[player_id].getBoundingBox()
            new_player_tracker = cv.legacy.TrackerCSRT_create()
            new_player_multi_tracker.add(new_player_tracker, img, player_bbox)
        self.player_multi_tracker = new_player_multi_tracker
    
    # take in image (next frame), generate player bboxs and corners using our mutitrackers (take in an image and array of bboxes)
    def updateCorners(self, img):
        # updates multitracker
        success, updated_corner_bounding_boxes = self.corner_multi_tracker.update(img)
        if success:
            self.corner_bounding_boxes = updated_corner_bounding_boxes
            self.updateTransformationMatrix()

        return success, img
    
    def updatePlayers(self, img):
        success, updated_player_bounding_boxes = self.player_multi_tracker.update(img)
        if success:
            for i in range(len(updated_player_bounding_boxes)):
                updated_bounding_box = updated_player_bounding_boxes[i]
                player_id = self.players_on_field[i]
                self.all_players[player_id].updateBoundingBox(updated_bounding_box)

        return success, img
    
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
                player.addToCoordinateHistory([None,None])
    
    def updateTransformationMatrix(self):
        # initialize empty source array
        source = np.float32([[0,0],[0,0],[0,0],[0,0]])
        for i, corner_box in enumerate(self.corner_bounding_boxes):
            middleCoords = getMiddleCoords(corner_box)
            # Update source array
            source[i][0] = middleCoords[0]
            source[i][1] = middleCoords[1]
        # This opencv function returns a matrix which translates our pixel coordinates into relative field coordinates
        self.transformation_matrix = cv.getPerspectiveTransform(source, self.destination_matrix)

# ================= PLAYER CLASS ==================================

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
    # file_name = 'huck.mp4'
    # Load the video
    cap = cv.VideoCapture(file_name)

    # This line reads the first frame of our video and returns | ret: Boolean which is set to TRUE if frame is successfully read, FALSE if not | img: First frame from the video
    ret, img = cap.read()
    if not ret:
        print("Failed to read frame from video source. Exiting...")
        exit()

    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    out = cv.VideoWriter(filename='trackedGameVideoFile.mp4', fourcc=fourcc, fps=4.0, frameSize=(img.shape[1], img.shape[0]))


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
        img = writeCornerBoundingBox(img, box)
    
    player_count = getPlayerCount(img)
    cv.destroyWindow("Corner MultiTracker")

    # specifies the four corners that we located
    destination_matrix = np.float32([[0,20],[0,90],[40,90],[40,20]])

    game = Game(corner_bounding_boxes, player_count, img, destination_matrix)

    # object detection
    detected_players_bounding_boxes = detectionSelection(img, game)
    for detected_player_bbox in detected_players_bounding_boxes:
        # add player to game and field
        game.addPlayerToGame(detected_player_bbox, True, img)

    # write all the boxes for detected players
    img = writePlayerBoundingBoxes(img, game)

    # Have user select any players that were not found by object detection 
    cv.namedWindow('Identify missing players.', cv.WINDOW_NORMAL)
    for i in range(len(game.getPlayersOnField()), game.max_players):
        bbox = cv.selectROI('Identify missing players.', img, False, printNotice=False)
        display_img = displayInstructions(img, "Draw a box around any players that don\'t currently have a box around them.")
        while (bbox[2] == 0 or bbox[3] == 0):
            bbox = cv.selectROI('Identify missing players.', display_img, False, printNotice=False)
        # add player to game and field
        player_id = game.addPlayerToGame(bbox, True, img)
        writePlayerBoundingBox(img, game, player_id)
    cv.destroyWindow('Identify missing players.')

    # have user identify which team each player is on
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
        success, img = game.updatePlayers(img)

        # If tracking was lost, run detection again 
        if not success:
            print("Tracking of player was lost!")
            redetectPlayers(img, game, redetect_all=True)
            success, img = game.updatePlayers(img)
        if not success:
            print("Full redection failed :(")

        img = writePlayerBoundingBoxes(img, game)

        out.write(img)

        cv.imshow("Tracking...", img)

        # Exit if ESC pressed
        k = cv.waitKey(1) & 0xff
        if k == 27 : break
        
        # check for routine redetection
        if counter >= 8:
            redetectPlayers(img, game)
            img = writePlayerBoundingBoxes(img, game)
            counter = 0
        
        game.addToAllPlayerCoordinateHistories()

        # grab every 10th frame to speed up testing
        for i in range(4):
            success, img = cap.read()
            counter += 1
            if not success:
                break

    print("Tracking complete.")

    cap.release()
    out.release()
    cv.destroyAllWindows()

    animateGame(game)

if __name__ == "__main__":
    main()
    