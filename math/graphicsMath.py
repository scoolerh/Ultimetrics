import numpy as np
from numpy.linalg import inv
import math

# Sequence of Events
# Begin with Frame of video and accompanying data (lat, long, elevation, pitch, yaw)
# We want a function which takes in these inputs and a pixel data and outputs the world coordinates of that pixel

# Should we convert from lat/long to meters before/after this function, thinking after





# THESE two functions relLat and relLong are attempting to scale our latitude and longitude relative to the top left corner of the field
# NOTE field runs east/west so the to top left corner is actually the farthest north and east corner
# increase in latitude is a northward movement
def relLat(lat):
    topLeftLat = 44.464654
    return (topLeftLat-lat)*111139

# increase in long means a eastward movement
def relLong(long):
    topLeftLong = -93.146314
    return (topLeftLong-long)*111139


# Test inputs, this is all information related to the drone itself
inputLat = relLat(44.4645524473822)
inputLong = relLong(-93.1474988221592)
print("input relative lat: " + str(inputLat))
print("input relative long: " + str(inputLong))
inputHeightFeet = 74.1 # feet
inputHeight = inputHeightFeet*0.3048 # meters
inputGimbalPitch = -24.5
inputGimbalYaw = 18
# inputGimbalRoll = 0

#Top-right corner of field test information (farthest south and east corner)
pixel_lat = relLat(44.464395)
pixel_long = relLong(-93.146314)
print("top right relative lat: " + str(pixel_lat))
print("top right relative long: " + str(pixel_long) + "\n")
pixel_elevation = 0

# This is the pixel coordinates of the top right corner of the field (our test point)
# pixel (0,0) is the top left of the film
pixel_x = 2007
pixel_y = 532


# ----------------------------------------------------------------------------------------------------------------------------

# Computes cosine
def cos(x):
    # takes in angle in degrees, outputs cos(x)
    return math.cos(math.radians(x))

# Computes sine
def sin(x):
    # takes in angle in degrees, outputs cos(x)
    return math.sin(math.radians(x))

# Computes tangent
def tan(x):
    # takes in angle in degrees, outputs cos(x)
    return math.tan(math.radians(x))

# Executes homogeneous divisio
def homogeneous_division(vec):
    # Ensure the fourth component (w) is not zero to avoid division by zero
    if vec[3] != 0:
        newVec = np.array([vec[0]/vec[2], vec[1]/vec[2], 1, vec[3]/vec[2]])
        return newVec
    else:
        raise ValueError("Homogeneous division by zero is undefined.")

# Prints the values of a vector
def print_vector(vector):
    print("Vector values:")
    for value in vector:
        print(value)

# np.set_printoptions(precision = 2, suppress = True)

# Camera information
fov = 83
fovy = fov/2

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

far = -16/9
# near is related to the aspect ratio
near = -9/16
# top view of camera
top = near * tan(fovy)
# bottom view of camera
bottom = -top
# right view of camera
right = top * 16/9
# left view of camera
left = -right

P = np.array([
    [(2*near)/(right-1), 0, (right + left)/(right - left), 0],
    [0, (2*near)/(top - bottom), (top + bottom)/(top - bottom), 0],
    [0, 0, -(far + near)/(far - near), -2*far*near],
    [0, 0, -1, 0]
])

invP = inv(P)

# ----------------------------------------------------------------------------------------------------------------------------

# Translation Matrix Information

# X is latitude
x = inputLat
# Y is longitude
y = inputLong

# Elevation value recorded on drone
z = inputHeight


T = np.array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
])

invT = inv(T)

# ----------------------------------------------------------------------------------------------------------------------------

# ROTATIONAL MATRIX INFORMATION

# yaw is the yaw of the gimbal
yaw = inputGimbalYaw
# pitch is the pitch of the gimbal
pitch = inputGimbalPitch


# yaw is r1
r1 = np.array([
    [cos(-yaw), -1*sin(-yaw), 0, 0],
    [sin(-yaw), cos(yaw), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# pitch is r2
r2 = np.array([
    [1, 0, 0, 0],
    [0, cos(90+pitch), -1*sin(90+pitch), 0],
    [0, sin(90+pitch), cos(90+pitch), 0],
    [0, 0, 0, 1]
])

# four options for the matrix: R, R^T, R', R'^T
# R = np.dot(r1,r2)
R = np.dot(r2, r1)
R = inv(R)
# R = np.matrix.transpose(R)
invR = inv(R)

# ----------------------------------------------------------------------------------------------------------------------------

# Computational Section
# Right now we are attempting to see if we can use known latitude and longitudes of a known pixel in a test frame to test our computations
world_x = pixel_lat
world_y = pixel_long
world_z = pixel_elevation
world_w = 1

# Placeholders
screen_x = 0
screen_y = 0
screen_z = 0
screen_w = 1

world_coords = np.array([world_x, world_y, world_z, world_w])

# screen_coords = np.array([screen_x, screen_y, screen_z, screen_w])


# world_coords = T @ R @ homogeneous_division(invP @ invV @ screen_coords)
# T * R * homo(invP * invV * screenCords)

screen_coords = V @ homogeneous_division(P @ invR @ invT @ world_coords)


print_vector(screen_coords)



