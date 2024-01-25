import numpy as np
from numpy.linalg import inv
import math
import cv2
import matplotlib.pyplot as plt

def screen2fieldCoordinates(x,y, transformation_matrix):
    #converts pixel coordinates to field coordinates in yards from top left
    inputArray = np.float32([[[x,y]]])
    # outputArray = np.float32([[[0, 0]]])
    outputArray = cv2.perspectiveTransform(inputArray, transformation_matrix)
    outputArray = outputArray[0][0]
    outputArray[0] = outputArray[0]/30.85
    outputArray[1] = outputArray[1]/30.85
    return outputArray


img = cv2.imread("./testFilmImage4Points.jpg") 

src = np.float32([
    [1194,681],
    [14,1751],
    [3525,1683],
    [2306,681]
])
# dst = np.float32([
#     [0,0],
#     [0,2160],
#     [1234,2160],
#     [1234,0]
# ])

dst = np.float32([
    [1320,30],
    [1320,2130],
    [2520,2130],
    [2520,30]
])


M = cv2.getPerspectiveTransform(src,dst)

x = 1922
# x = 3525
y = 1116
# y= 1683
converted_coordinates = screen2fieldCoordinates(x, y, M)
print("Pixel coordinates (" + str(x) +  ", " + str(y) + ")" + " converts to:")
print("(" + str(converted_coordinates[0]) + ", " + str(converted_coordinates[1]) + ")")
print("in yard coordinates")

# out = cv2.warpPerspective(img,M,(1234,2160),flags=cv2.INTER_LINEAR)
out = cv2.warpPerspective(img,M,(3840, 2160),flags=cv2.INTER_LINEAR)


cv2.imwrite("warpedImage.jpg", out)
cv2.imshow("Image", out)

cv2.waitKey()
