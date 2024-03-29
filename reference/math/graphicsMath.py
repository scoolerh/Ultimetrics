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
    # return math.cos(math.radians(x))
    return math.cos(x)

# Computes sine, make sure x is in radians NOT degrees
def sin(x):
    # takes in angle in degrees, outputs cos(x)
    # return math.sin(math.radians(x))
    return math.sin(x)

# Computes tangent, make sure x is in radians NOT degrees
def tan(x):
    # takes in angle in degrees, outputs cos(x)
    # return math.tan(math.radians(x))
    return math.tan(x)

# Executes homogeneous division based off third component
def homogeneous_division(vec):
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
    return (lat - topLeftLat)*111132.954

# increase in long means a eastward movement
def rel_Long(long, lat):
    topLeftLong = -93.146163
    relLong = long-topLeftLong
    sinOutput = sin(math.pi/2 - lat)
    longScalar = sinOutput*111132.954
    return relLong*longScalar

def test_coordinate_data():
    print("Top Left Corner Data:")
    print(topLeftRelX)
    print(topLeftRelY)
    print("Top Right Corner Data:")
    print(topRightRelX)
    print(topRightRelY)
    print("Yellow Corner Data:")
    print(yellowCornerRelX)
    print(yellowCornerRelY)
    print("Drone Data:")
    print(droneRelX)
    print(droneRelY)

# Calculate how far off the predicted pixel was from the true pixel
def pixel_diff(prediction, truth):
    predictedX = prediction[0]
    predictedY = prediction[1]
    trueX = truth[0]
    trueY = truth[1]
    xDiff = predictedX - trueX
    yDiff = predictedY - trueY
    return xDiff, yDiff

# def matrix_test(R, testName, worldCoords1, worldCoords2, worldCoords3):
#     invR = inv(R)
    
#     predictedScreenCoords1 = homogeneous_division(V @ P @ invR @ invT @ worldCoords1)  # Predicted pixel data for Top Left Corner
#     predictedScreenCoords2 = homogeneous_division(V @ P @ invR @ invT @ worldCoords2)  # Predicted pixel data for Top Right Corner
#     predictedScreenCoords3 = homogeneous_division(V @ P @ invR @ invT @ worldCoords3)  # Predicted pixel data for Yellow Corner

#     xDiffLeft, yDiffLeft = pixel_diff(predictedScreenCoords1, topLeftTruePixels)
#     xDiffRight, yDiffRight = pixel_diff(predictedScreenCoords2, topRightTruePixels)
#     xDiffYellow, yDiffYellow = pixel_diff(predictedScreenCoords3, yellowCornerTruePixels)

#     print("\nTesting: " + testName + "\n")
#     print("Top Left Corner: \n")
#     print("Predicted X Pixel Diff: " + str(xDiffLeft) + " Value: " + str(predictedScreenCoords1[0]) + "\n")
#     print("Predicted Y Pixel Diff: " + str(yDiffLeft) + " Value: " + str(predictedScreenCoords1[1]) + "\n")
#     print("Top Right Corner: \n")
#     print("Predicted X Pixel Diff: " + str(xDiffRight) + " Value: " + str(predictedScreenCoords2[0]) + "\n")
#     print("Predicted Y Pixel Diff: " + str(yDiffRight) + " Value: " + str(predictedScreenCoords2[1]) + "\n")
#     print("Yellow Corner: \n")
#     print("Predicted X Pixel Diff: " + str(xDiffYellow) + " Value: " + str(predictedScreenCoords3[0]) + "\n")
#     print("Predicted Y Pixel Diff: " + str(yDiffYellow) + " Value: " + str(predictedScreenCoords3[1]) + "\n")
#     print("\n ------------------------------------------- \n")


def matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3):
    try:
        invR = np.transpose(R) # inverting and doing the transpose are equivalent here
        
        predictedScreenCoords1 = homogeneous_division(V @ P @ invR @ invT @ worldCoords1)  # Predicted pixel data for Top Left Corner
        predictedScreenCoords2 = homogeneous_division(V @ P @ invR @ invT @ worldCoords2)  # Predicted pixel data for Top Right Corner
        predictedScreenCoords3 = homogeneous_division(V @ P @ invR @ invT @ worldCoords3)  # Predicted pixel data for Yellow Corner

        invP = inv(P)

        predictedScreenCoords1 = homogeneous_division(V @ P @ R @ invT @ worldCoords1)  # Predicted pixel data for Top Left Corner
        predictedScreenCoords2 = homogeneous_division(V @ P @ R @ invT @ worldCoords2)  # Predicted pixel data for Top Right Corner
        predictedScreenCoords3 = homogeneous_division(V @ P @ R @ invT @ worldCoords3)  # Predicted pixel data for Yellow Corner

        xDiffLeft, yDiffLeft = pixel_diff(predictedScreenCoords1, topLeftTruePixels)
        xDiffRight, yDiffRight = pixel_diff(predictedScreenCoords2, topRightTruePixels)
        xDiffYellow, yDiffYellow = pixel_diff(predictedScreenCoords3, yellowCornerTruePixels)

        total_diff = abs(xDiffLeft) + abs(yDiffLeft) + abs(xDiffRight) + abs(yDiffRight) + abs(xDiffYellow) + abs(yDiffYellow)

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
        print("\nTotal Sum of Differences: " + str(total_diff))
        print("\n ------------------------------------------- \n")

        return max([abs(xDiffLeft), abs(yDiffLeft), abs(xDiffRight), abs(yDiffRight), abs(xDiffYellow), abs(yDiffYellow)])
        # return total_diff
    except Exception as e:
        print(f"An error occurred in test: {testName}\nError: {e}")
        return float('inf')  # Return a large value for failed cases




# ----------------------------------------------------------------------------------------------------------------------------
# Code testing section

# Computed the conversion for 40 yards to meters and found that distance represented in latitude. Added that to latitude for top right corner measurement
# Top left corner of playing field
topLeftLat = 44.46475110139
topLeftLong = -93.146163
topLeftRelX = rel_Long(topLeftLong, topLeftLat)
topLeftRelY = rel_Lat(topLeftLat)
# Where origin is top left 
topLeftPixelX = 1018
topleftPixelY = 534
# Where origin is bottom left (built for graphics methods)
topLeftGraphicsX = topLeftPixelX
topLeftGraphicsY = 2160 - topleftPixelY
# topLeftGraphicsY = topleftPixelY
topLeftTruePixels = np.array([topLeftGraphicsX, topLeftGraphicsY, 0, 0])

# Top right corner of playing field
topRightLat = 44.464422
topRightLong = -93.146163
topRightRelX = rel_Long(topRightLong, topRightLat)
topRightRelY = rel_Lat(topRightLat)
# Where origin is top left
topRightPixelX = 2007
topRightPixelY = 527
# Where origin is bottom left
topRightGraphicsX = topRightPixelX
topRightGraphicsY = 2160 - topRightPixelY
# topRightGraphicsY = topRightPixelY
topRightTruePixels = np.array([topRightGraphicsX, topRightGraphicsY, 0, 0])

# Bottom Right instersection of yellow lines
yellowCornerLat = 44.464400
yellowCornerLong = -93.146936
yellowCornerRelX = rel_Long(yellowCornerLong, yellowCornerLat) 
yellowCornerRelY = rel_Lat(yellowCornerLat)
# Where origin is top left
yellowCornerPixelX = 2716
yellowCornerPixelY = 1445
# Where origin is bottom left
yellowCornerGraphicsX = yellowCornerPixelX
yellowCornerGraphicsY = 2160 - yellowCornerPixelY
# yellowCornerGraphicsY = yellowCornerPixelY
yellowCornerTruePixels = np.array([yellowCornerGraphicsX, yellowCornerGraphicsY, 0, 0])


# Drone information from the first frame of video
# Go measure accuracy of the drone in real world
droneLat = 44.46457025
droneLong = -93.14746161
droneRelX = rel_Long(droneLong, droneLat)
droneRelY = rel_Lat(droneLat)
droneHeightFeet = 90.5 # Feet
droneHeight = droneHeightFeet*0.3048 # Meters


# Do we have to worry about the yaw of the drone itself? Are we assuming pitch of drone itself is 0?
inputGimbalPitchDeg = -23.3 # Degrees measured relative to horizontal (parallel to ground) negative is looking down positive is looking up, might also have to incorporate the pitch of the drone itself which was at -1.2 degrees
inputGimbalPitchDegWOSD = -24.5 # Pitch degree with the -1.2 degree of OSD.pitch incorporated
newPitchDeg = 90 + inputGimbalPitchDeg
inputGimbalYawDeg = 101.5 # Degrees (relative to north going clockwise)
dronePitch = math.radians(inputGimbalPitchDeg) # Radians
# dronePitch = math.radians(inputGimbalPitchDeg)
droneYaw = math.radians(inputGimbalYawDeg) # Radians
droneRoll = math.radians(-3)

# dronePitch = inputGimbalPitchDeg
# droneYaw = inputGimbalYawDeg


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


# V = inv(V)
invV = inv(V)


# ----------------------------------------------------------------------------------------------------------------------------
# PERSPECTIVE PROJECTION MATRIX INFORMATION

# Camera information
fov = math.radians(82)
# fovy = fov/2

# Test I ran said that Focal = 1, Ratio = 2, Yaw = 315, Pitch = 45 was the best combo
focal = 15
# focal = 1000.0


# far = -focal * ratio
# near = -focal / ratio
far = 1.0
near = -1.0

top = focal*tan(fov * 0.5)
bottom = -top


# right view of camera
right = top * 1/16
# left view of camera
left = -right

# Assumes frustrum is centered along z axis, this is the P from josh's material
P = np.array([
    [(-2*near)/(right-left), 0, (right + left)/(right - left), 0],
    [0, (-2*near)/(top - bottom), (top + bottom)/(top - bottom), 0],
    [0, 0, (near + far)/(near - far), (-2*far*near)/(near - far)],
    [0, 0, -1, 0]
])

# # This is the persepctive projection matrxi from the youtube video https://www.youtube.com/watch?v=U0_ONQQ5ZNM
# P = np.array([
#     [(2*near)/(right-left), 0, -(right + left)/(right - left), 0],
#     [0, (2*near)/(bottom - top), -(bottom + top)/(bottom - top), 0],
#     [0, 0, far/(far - near), (-far*near)/(far - near)],
#     [0, 0, 1, 0]
# ])

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

# yaw is the yaw of the gimbal measured in 360 degrees relative to true north (we already converted to radians)
# pitch is the pitch of the gimbal measured relative to the horizontal plane of the drone (already convereted to radians, may need to incorporate the pitch of the drone itself)
# roll is the roll of the drone measured (we already converted to radians)
# These are defined at the top of our code

yaw = droneYaw
pitch = dronePitch
roll = droneRoll

# # Rotation about the z-axis (turns left and right)
# yawMatrix = np.array([
#     [cos(yaw), -sin(yaw), 0, 0],
#     [sin(yaw), cos(yaw), 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])

# # Rotation about the y-axis (pitcing up and down)
# pitchMatrix = np.array([
#     [cos(pitch), 0, sin(pitch), 0],
#     [0, 1, 0, 0],
#     [-sin(pitch), 0, cos(pitch), 0],
#     [0, 0, 0, 1]
# ])

# # Rotation about the x-axis (rolling side to side)
# rollMatrix = np.array([
#     [1, 0, 0, 0],
#     [0, cos(roll), -sin(roll), 0],
#     [0, sin(roll), cos(roll), 0],
#     [0, 0, 0, 1]
# ])

# # Matrix used to rotate the cameras persepctive to be relative to the coordinate system we are using
# R_x = np.array([
#     [1, 0, 0, 0],
#     [0, cos(math.pi), 0],
#     [-sin(-math.pi/2), 0, cos(-math.pi/2), 0],
#     [0, 0, 0, 1]
# ])

# # Matrix used to rotate the cameras persepctive to be relative to the coordinate system we are using
# R_y = np.array([
#     [cos(-math.pi/2), 0, sin(-math.pi/2), 0],
#     [0, 1, 0, 0],
#     [-sin(-math.pi/2), 0, cos(-math.pi/2), 0],
#     [0, 0, 0, 1]
# ])

# # Matrix used to rotate the cameras perspective to be relative to the coordinate system we are using
# R_z = np.array([
#     [cos(-math.pi/2), -sin(-math.pi/2), 0, 0],
#     [sin(-math.pi/2), cos(-math.pi/2), 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ])


stdNegZ = np.array([0, 0, -1])
stdY = np.array([0, 1, 0])
defaultNegZ = np.array([0, 1, 0])
defaultY = np.array([0, 0, 1])


yawMatrix = np.array([
    [cos(-yaw), -sin(-yaw), 0],
    [sin(-yaw), cos(-yaw), 0],
    [0, 0, 1],
])

pitchMatrix = np.array([
    [1, 0, 0],
    [0, cos(pitch), -sin(pitch)],
    [0, sin(pitch), cos(pitch)],
])

rollMatrix = np.array([
    [cos(roll), 0, sin(roll)],
    [0, 1, 0],
    [-sin(roll), 0, cos(roll)],
])

# R_yp = yawMatrix @ pitchMatrix
# R_py = pitchMatrix @ yawMatrix

currentNegZ = yawMatrix @ pitchMatrix @ defaultNegZ
currentY = yawMatrix @ pitchMatrix @ defaultY


def basisRotation(stdY, stdNegZ, inputY, inputNegZ):
    # Calculate cross product of stdY and stdNegZ
    cross_product_R = np.cross(stdY, stdNegZ)
    
    # Create matrix R
    R = np.array([stdY, stdNegZ, cross_product_R])
    
    # Calculate cross product of inputY and inputNegZ
    cross_product_S = np.cross(inputY, inputNegZ)
    
    # Create matrix S
    S = np.array([inputY, inputNegZ, cross_product_S])
    
    Rtranspose = np.transpose(R)
    f = S @ Rtranspose
    matrix4d = np.zeros((4, 4))
    matrix4d[:3, :3] = f
    matrix4d[3, 3] = 1
    print(matrix4d)
    return matrix4d

currentNegZ = yawMatrix @ pitchMatrix @ defaultNegZ
currentY = yawMatrix @ pitchMatrix @ defaultY

R = basisRotation(stdY, stdNegZ, currentY, currentNegZ)

currentNegZ = pitchMatrix @ yawMatrix @ defaultNegZ
currentY = pitchMatrix @ yawMatrix @ defaultY

R2 = basisRotation(stdY, stdNegZ, currentY, currentNegZ)

# ----------------------------------------------------------------------------------------------------------------------------
# Computational Section

def test_function4():
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

    results = []
    # Basis Rotation 1 (pitch then yaw)
    R = basisRotation(stdY, stdNegZ, currentY, currentNegZ)
    testName = "basisRotation1 pitch then yaw"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # Basis Rotation 2 (yaw then p)
    R2
    testName = "basisRotation2 yaw then pitch"
    results.append(matrix_test(R2, P, testName, worldCoords1, worldCoords2, worldCoords3))


    print(results)
    # print(min(results))
    # print(np.argmin(results))




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

    results = []
    # yaw @ pitch
    R = yawMatrix @ pitchMatrix
    testName = "yaw @ pitch"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # inv(yaw) @ pitch
    R = inv(yawMatrix) @ pitchMatrix
    testName = "inv(yaw) @ pitch"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # yaw @ inv(pitch)
    R = yawMatrix @ inv(pitchMatrix)
    testName = "yaw @ inv(pitch)"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # inv(yaw) @ inv(pitch)
    R = inv(yawMatrix) @ inv(pitchMatrix)
    testName = "inv(yaw) @ inv(pitch)"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # pitch @ yaw
    R = pitchMatrix @ yawMatrix
    testName = "pitch @ yaw"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))
    
    # inv(pitch) @ yaw
    R = inv(pitchMatrix) @ yawMatrix
    testName = "pitch @ yaw"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))
    
    # pitch @ inv(yaw)
    R = pitchMatrix @ inv(yawMatrix)
    testName = "pitch @ yaw"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))
    
    # inv(pitch) @ inv(yaw)
    R = inv(pitchMatrix) @ inv(yawMatrix)
    testName = "inv(pitch) @ inv(yaw)"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    print(results)
    print(min(results))
    print(np.argmin(results))

# Same as test_function but just with the roll matrixa added onto the end
def test_function2():
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

    results = []
    # pitch then yaw then roll (rollMatrix @ yawMatrix @ pitchMatrix) with camera rotation first
    R = rollMatrix @ yawMatrix @ pitchMatrix @ R_y @ R_z
    testName = "rollMatrix @ yawMatrix @ pitchMatrix @ R_y @ R_z"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # pitch then yaw then roll (rollMatrix @ yawMatrix @ pitchMatrix) with camera rotation last
    R = R_y @ R_z @ rollMatrix @ yawMatrix @ pitchMatrix
    testName = "R = R_y @ R_z @ rollMatrix @ yawMatrix @ pitchMatrix"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # inv(roll) then inv(yaw) then inv(pitch) with camera rotation first
    R = inv(pitchMatrix) @ inv(yawMatrix) @ inv(rollMatrix) @ R_y @ R_z
    testName = "inv(pitchMatrix) @ inv(yawMatrix) @ inv(rollMatrix) @ R_y @ R_z"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # inv(roll) then inv(yaw) then inv(pitch) with camera rotation last
    R = R_y @ R_z @ inv(pitchMatrix) @ inv(yawMatrix) @ inv(rollMatrix)
    testName = "R_y @ R_z @ inv(pitchMatrix) @ inv(yawMatrix) @ inv(rollMatrix)"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))

    # inv(roll) then inv(yaw) then inv(pitch) with camera rotation last
    R = R_y @ R_z @ pitchMatrix @ yawMatrix @ rollMatrix
    testName = "R_y @ R_z @ inv(pitchMatrix) @ inv(yawMatrix) @ inv(rollMatrix)"
    results.append(matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3))


    print(results)
    print(min(results))
    print(np.argmin(results))

def test_function3():
    worldCoords1 = np.array([topLeftRelX, topLeftRelY, 0, 1])  # Top Left Corner
    worldCoords2 = np.array([topRightRelX, topRightRelY, 0, 1])  # Top Right Corner
    worldCoords3 = np.array([yellowCornerRelX, yellowCornerRelY, 0, 1])  # Yellow Corner

    best_performance = float('inf')  # Initialize with a large value
    best_parameters = None

    # Best appears to be the highest focal possible and a ratio of 18
    for focal in range(1, 101):
        for ratio in range(1, 19):
            fov = math.radians(83)
            
            far = -focal * ratio
            near = -focal / ratio

            top = focal*tan(fov * 0.5)
            bottom = -top

            right = top * 16/9
            left = -right

            # Check for division by zero
            if far == near:
                continue

            P = np.array([
                [(2*near)/(right-1), 0, (right + left)/(right - left), 0],
                [0, (2*near)/(top - bottom), (top + bottom)/(top - bottom), 0],
                [0, 0, -(far + near)/(far - near), (-2*far*near)/(far-near)],
                [0, 0, -1, 0]
            ])

            # Test all 8 combinations
            matrices = [
                yawMatrix @ pitchMatrix,
                inv(yawMatrix) @ pitchMatrix,
                yawMatrix @ inv(pitchMatrix),
                inv(yawMatrix) @ inv(pitchMatrix),
                pitchMatrix @ yawMatrix,
                inv(pitchMatrix) @ yawMatrix,
                pitchMatrix @ inv(yawMatrix),
                inv(pitchMatrix) @ inv(yawMatrix),
            ]

            for R in matrices:
                testName = f"Focal: {focal}, Ratio: {ratio}"
                total_diff = matrix_test(R, P, testName, worldCoords1, worldCoords2, worldCoords3)

                # Update best performance if the current combination is better
                if total_diff < best_performance:
                    best_performance = total_diff
                    best_parameters = (focal, ratio, R)

        print(f"Best performing parameters: Focal = {best_parameters[0]}, Ratio = {best_parameters[1]}")
        print(f"Best performing rotation matrix:\n {best_parameters[2]}")
        
# test_function()
test_function4()
# test_coordinate_data()