import cv2 as cv
import numpy as np
import yolov5
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animationLib
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")

file_name = "UltimetricsTester.png"

img = cv.imread(file_name)

cv.namedWindow("Select box for colors", cv.WINDOW_NORMAL)
box = cv.selectROI("Select box for colors", img, False)
#debug
#print(img.shape)
#print(box)
boxCoords = (box[0], box[1], (box[0] + box[2]), (box[1] + box[3]))
#iterate over and get the pixels
# mask = cv.inRange(img, (20, 20, 20), (40, 255, 40))
# cv.imshow("masked", img)
# cv.waitKey(0)
# inv_mask = cv.bitwise_not(mask)
# no_green = cv.bitwise_and(img, inv_mask)
# cv.imshow("no green", no_green)
#we are BGR coordinates, so we want to limit by getting rid of all green pixels
#safe pixels are pixels outside of the green range
safePixels = []
colorVal = 0
for i in range(boxCoords[0], boxCoords[2]) :
    for j in range(boxCoords[1], boxCoords[3]) :
        #debug
        #print(i,j)
        b = img[j,i,0]
        g = img[j,i,1]
        r = img[j,i,2]
        if not ((b < 40 and 20 < g < 255 and r < 40) or (((g - 15) > r) and ((g-15) > b))) :
            safePixels.append(img[j,i])
#print(safePixels)
total = 0
sumB = 0
sumG = 0
sumR = 0
for pixel in safePixels :
    total += 1
    sumB += pixel[0]
    sumG += pixel[1]
    sumR += pixel[2]
print(sumB/total)
print(sumG/total)
print(sumR/total)


