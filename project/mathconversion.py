import numpy as np
import cv2

#converts pixel coordinates to field coordinates in yards from top left
def screen2fieldCoordinates(x,y, transformation_matrix):
    inputArray = np.float32([[[x,y]]])
    outputArray = cv2.perspectiveTransform(inputArray, transformation_matrix)
    outputArray = outputArray[0][0]
    output = []
    output.append(round(outputArray[0]/30.85,2))
    output.append(round(outputArray[1]/30.85,2))
    return output

src = np.float32([
    [1194,681],
    [14,1751],
    [3525,1683],
    [2306,681]
])

dst = np.float32([
    [1320,30],
    [1320,2130],
    [2520,2130],
    [2520,30]
])
M = cv2.getPerspectiveTransform(src,dst)
oldcornerfile = open("corners.txt", "r")
newcornerfile = open("mathedCorners.txt", "w")
oldplayerfile = open("playercoordinates.txt", "r")
newplayerfile = open("mathedPlayers.txt", "w")

print("Beginning corner coordinate math...")
for line in oldcornerfile: 
    line = line[2:-2]
    corners = line.split("), (")
    newcorners = []
    for corner in corners:
        corner = corner[1:-1]
        corner = corner.split(", ")
        x = float(corner[0])
        y = float(corner[1])
        converted_coordinates = screen2fieldCoordinates(x, y, M)
        newcorners.append(converted_coordinates)
    newcornerfile.write(str(newcorners) + "\n")

print("Corner coordinate math complete.")
print("Beginning player coordinate math...")
for line in oldplayerfile: 
    line = line[2:-2]
    players = line.split("), (")
    newplayers = []
    for player in players:
        player = player[1:-1]
        player = player.split(", ")
        x = float(player[0])
        y = float(player[1])
        converted_coordinates = screen2fieldCoordinates(x, y, M)
        newplayers.append(converted_coordinates)
    newplayerfile.write(str(newplayers) + "\n")
print("Player coordinate math complete.")

oldcornerfile.close()
newcornerfile.close()
oldplayerfile.close()
newplayerfile.close()
