import cv2
import numpy as np

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Selected pixel coordinates: x = {x}, y = {y}")

# Load your image
image = cv2.imread("firstFrame0647.jpg")  # Replace with the path to your image

# Create a window and set the callback function for mouse events
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse)

while True:
    cv2.imshow("Image", image)
    
    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()