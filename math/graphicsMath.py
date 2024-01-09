import numpy as np
from numpy.linalg import inv
import math

# Sequence of Events
# Begin with Frame of video and accompanying data (lat, long, elevation, pitch, yaw)
# We want a function which takes in these inputs and a pixel data and outputs the world coordinates of that pixel

# Should we convert from lat/long to meters before/after this function, thinking after
# Create a funciton which test all 8 possibilities of PY matrices
# Add more test points on the ultimate field which are in the frame

# To go from screen_coords to world_coords we need to have screen_z (third component of vector) is this something we just need to calculate once?


# ----------------------------------------------------------------------------------------------------------------------------
# Helper Functions Section

# Computes cosine, make sure x is in radians NOT degrees
def cos(x):
    # takes in angle in degrees, outputs cos(x)
    return math.cos(math.radians(x))

# Computes sine, make sure x is in radians NOT degrees
def sin(x):
    # takes in angle in degrees, outputs cos(x)
    return math.sin(math.radians(x))

# Computes tangent, make sure x is in radians NOT degrees
def tan(x):
    # takes in angle in degrees, outputs cos(x)
    return math.tan(math.radians(x))

# Executes homogeneous division based off third component
def homogeneous_division(vec):
    # Ensure the fourth component (w) is not zero to avoid division by zero
    if vec[3] != 0:
        newVec = np.array([vec[0]/vec[3], vec[1]/vec[3], vec[2]/vec[3], 1])
        return newVec
    else:
        raise ValueError("Homogeneous division by zero is undefined.")

# Prints the values of a vector
def print_vector(vector):
    print("Vector values:")
    for value in vector:
        print(value)

# THESE two functions relLat and relLong are attempting to scale our latitude and longitude relative to the top left corner of the field
# NOTE field runs east/west so the to top left corner is actually the farthest north and east corner
# Top left corner has a 
# Increase in latitude is a northward movement
def rel_Lat(lat):
    topLeftLat = 44.46475110139
    return (topLeftLat-lat)*111139

# increase in long means a eastward movement
def rel_Long(long, lat):
    topLeftLong = -93.146163
    relLong = topLeftLong-long
    sinInput = 3.1415926/2 - lat
    sinOutput = sin(sinInput)
    return relLong*sinOutput

# Calculate how far off the predicted pixel was from the true pixel
def pixel_diff(prediction, truth):
    predictedX = prediction[0]
    predictedY = prediction[1]
    trueX = truth[0]
    trueY = truth[1]
    xDiff = predictedX - trueX
    yDiff = predictedY - trueY
    return xDiff, yDiff

def matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3):
    invR = inv(R)
    predictedScreenCoords1 = V @ homogeneous_division(P @ invR @ invT @ worldCoords1)  # Predicted pixel data for Top Left Corner
    predictedScreenCoords2 = V @ homogeneous_division(P @ invR @ invT @ worldCoords2)  # Predicted pixel data for Top Right Corner
    predictedScreenCoords3 = V @ homogeneous_division(P @ invR @ invT @ worldCoords3)  # Predicted pixel data for Yellow Corner

    xDiffLeft, yDiffLeft = pixel_diff(predictedScreenCoords1, topLeftTruePixels)
    xDiffRight, yDiffRight = pixel_diff(predictedScreenCoords2, topRightTruePixels)
    xDiffYellow, yDiffYellow = pixel_diff(predictedScreenCoords3, yellowCornerTruePixels)

    print("\nTesting: " + testName + "\n")
    print("Top Left Corner: \n")
    print("Predicted X Pixel Diff: " + str(xDiffLeft) + " Value: " + str(predictedScreenCoords1[0]) + "\n")
    print("Predicted Y Pixel Diff: " + str(yDiffLeft) + " Value: " + str(predictedScreenCoords1[1]) + "\n")
    print("Top Right Corner: \n")
    print("Predicted X Pixel Diff: " + str(xDiffRight) + " Value: " + str(predictedScreenCoords2[0]) + "\n")
    print("Predicted Y Pixel Diff: " + str(yDiffRight) + " Value: " + str(predictedScreenCoords2[1]) + "\n")
    print("Yellow Corner: \n")
    print("Predicted X Pixel Diff: " + str(xDiffYellow) + " Value: " + str(predictedScreenCoords3[0]) + "\n")
    print("Predicted Y Pixel Diff: " + str(yDiffYellow) + " Value: " + str(predictedScreenCoords3[1]) + "\n")
    print("\n ------------------------------------------- \n")



# ----------------------------------------------------------------------------------------------------------------------------
# Code testing section

# Computed the conversion for 40 yards to meters and found that distance represented in latitude. Added that to latitude for top right corner measurement
# Top left corner of playing field
topLeftLat = 44.46475110139
topLeftLong = -93.146163
topLeftRelX = rel_Lat(topLeftLat)
topLeftRelY = rel_Long(topLeftLong, topLeftLat)
topLeftPixelX = 1018
topleftPixelY = 534
topLeftTruePixels = np.array([topLeftPixelX, topleftPixelY, 0, 0])

# Top right corner of playing field
topRightLat = 44.464422
topRightLong = -93.146163
topRightRelX = rel_Lat(topRightLat)
topRightRelY = rel_Long(topRightLong, topRightLat)
topRightPixelX = 2007
topRightPixelY = 527
topRightTruePixels = np.array([topRightPixelX, topRightPixelY, 0, 0])

# Bottom Right instersection of yellow lines
yellowCornerLat = 44.464400
yellowCornerLong = -93.146936
yellowCornerRelX = rel_Lat(yellowCornerLat)
yellowCornerRelY = rel_Long(yellowCornerLong, yellowCornerLat)
yellowCornerPixelX = 2716
yellowCornerPixelY = 1445
yellowCornerTruePixels = np.array([yellowCornerPixelX, yellowCornerPixelY, 0, 0])


# Drone information from the first frame of video
droneLat = 44.4645524473822
droneLong = -93.1474988221592
droneRelX = rel_Lat(droneLat)
droneRelY = rel_Long(droneLong, droneLat)
droneHeightFeet = 74.1 # Feet
droneHeight = droneHeightFeet*0.3048 # Meters

degree = math.pi/180.0
inputGimbalPitchDeg = -24.5 # Degrees
inputGimbalYawDeg = 18 # Degrees
dronePitch = inputGimbalPitchDeg*degree # Radians
droneYaw = inputGimbalYawDeg*degree # Radians


# ----------------------------------------------------------------------------------------------------------------------------
# VIEWPORT MATRIX INFORMATION

r = 3840 # x value of frame
t = 2160 # y value of frame

V = np.array([
    [r/2, 0, 0, r/2],
    [0, t/2, 0, t/2],
    [0, 0, 1/2, 1/2],
    [0, 0, 0, 1]
])

invV = inv(V)


# ----------------------------------------------------------------------------------------------------------------------------
# PERSPECTIVE PROJECTION MATRIX INFORMATION

# Camera information
fov = 83
fovy = fov/2
# Might want to try with fov not fovy as well

far = -16/9
# -focal length * ratio, ratio is not aspect ratio of screen it should be 10?
# near is related to the aspect ratio
near = -9/16
# top view of camera
top = -near * tan(fovy)
# bottom view of camera
bottom = -top
# right view of camera
right = top * 16/9
# left view of camera
left = -right

P = np.array([
    [(2*near)/(right-1), 0, (right + left)/(right - left), 0],
    [0, (2*near)/(top - bottom), (top + bottom)/(top - bottom), 0],
    [0, 0, -(far + near)/(far - near), (-2*far*near)/(far-near)],
    [0, 0, -1, 0]
])

invP = inv(P)

# ----------------------------------------------------------------------------------------------------------------------------

# Translation Matrix Information

# X is drone position relative to top left corner
x = droneRelX

# Y is drone position relative to top left corner
y = droneRelY

# Elevation value recorded on drone
z = droneHeight


T = np.array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
])

invT = inv(T)
# Hard code the inverse for production code

# ----------------------------------------------------------------------------------------------------------------------------

# ROTATIONAL MATRIX INFORMATION

# yaw is the yaw of the gimbal
# pitch is the pitch of the gimbal
# These are defined at the top of our code

yaw = droneYaw
pitch = dronePitch

# Yaw Matrix, accounts for 1/2 of rotational matrix
yawMatrix = np.array([
    [cos(-yaw), -1*sin(-yaw), 0, 0],
    [sin(-yaw), cos(-yaw), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Pitch Matrix, accounts for 1/2 of rotational matrix
pitchMatrix = np.array([
    [1, 0, 0, 0],
    [0, cos(math.pi/2 + pitch), -1*sin(math.pi/2 + pitch), 0],
    [0, sin(math.pi/2 + pitch), cos(math.pi/2 + pitch), 0],
    [0, 0, 0, 1]
])


# ----------------------------------------------------------------------------------------------------------------------------
# Computational Section

def test_function():
    worldCoords1 = np.array([topLeftRelX, topLeftRelY, 0, 1]) # Top Left Corner
    worldCoords2 = np.array([topRightRelX, topRightRelY, 0, 1]) # Top Right Corner
    worldCoords3 = np.array([yellowCornerRelX, yellowCornerRelY, 0, 1]) # Yellow Corner

    # 8 Combinations we need to try
    # yaw @ pitch
    # inv(yaw) @ pitch
    # yaw @ inv(pitch)
    # inv(yaw) @ inv(pitch)
    # pitch @ yaw
    # inv(pitch) @ yaw
    # pitch @ inv(yaw)
    # inv(pitch) @ inv(yaw)

    # yaw @ pitch
    R = yawMatrix @ pitchMatrix
    testName = "yaw @ pitch"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)

    # inv(yaw) @ pitch
    R = inv(yawMatrix) @ pitchMatrix
    testName = "inv(yaw) @ pitch"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)

    # yaw @ inv(pitch)
    R = yawMatrix @ inv(pitchMatrix)
    testName = "yaw @ inv(pitch)"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)

    # inv(yaw) @ inv(pitch)
    R = inv(yawMatrix) @ inv(pitchMatrix)
    testName = "inv(yaw) @ inv(pitch)"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)

    # pitch @ yaw
    R = pitchMatrix @ yawMatrix
    testName = "pitch @ yaw"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)
    
    # inv(pitch) @ yaw
    R = inv(pitchMatrix) @ yawMatrix
    testName = "pitch @ yaw"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)
    
    # pitch @ inv(yaw)
    R = pitchMatrix @ inv(yawMatrix)
    testName = "pitch @ yaw"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)
    
    # inv(pitch) @ inv(yaw)
    R = inv(pitchMatrix) @ inv(yawMatrix)
    testName = "inv(pitch) @ inv(yaw)"
    matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3)


test_function()


# screen_coords = V @ homogeneous_division(P @ invR @ invT @ world_coords)
