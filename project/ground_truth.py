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
# from api_key import API_KEY_ROBOFLOW, PROJECT_NAME, VERSION
warnings.filterwarnings("ignore")

output_file = open('groundTruth.txt', 'w')
# Name of mp4 with frisbee film
file_name = 'frisbee.mp4'
# file_name = 'huck.mp4'

# Load the video
cap = cv.VideoCapture(file_name)

# ==================== PLAYER/CORNER TRACKING ======================================

# Loop through video
cv.namedWindow("Tracking...", cv.WINDOW_NORMAL)
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    cv.namedWindow("Player Selector", cv.WINDOW_NORMAL)
    for j in range(14):
        box = cv.selectROI("Player Selector", img, False, printNotice=False)
        output_file.write(str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(box[3]) + "|")
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(img, p1, p2, (0,0,0), 2, 1)
    output_file.write("\n")
    
    cv.destroyWindow("Player Selector")

    # Exit if ESC pressed
    k = cv.waitKey(1) & 0xff
    if k == 27 : break

    for i in range(20):
        success, img = cap.read()
        if not success:
            break

print("Tracking complete.")

cap.release()
cv.destroyAllWindows()