# get the coordinates of the corners of the field 
import cv2
import numpy as np

corners = []
f = open("corners.txt", "w")

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append(f"({x},{y})")

# Load the first frame of the video
cap = cv2.VideoCapture('frisbee.mp4')
ret, img = cap.read()

# Create a window and set the callback function for mouse events
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse)

while len(corners) < 4:
    cv2.imshow("Image", img)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

f.write(str(corners))
f.close()
cv2.destroyAllWindows()