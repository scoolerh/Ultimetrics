import cv2 as cv
import numpy as np

# takes a little longer than the video length to run 

cap = cv.VideoCapture('Videos/full.mp4')

ret, img = cap.read()

fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

# if doing grayscale
#out = cv.VideoWriter("Videos/full_grayscale.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720), isColor=False)
# otherwise
out = cv.VideoWriter("Videos/full_sharpen.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720))

while True:
    success, img = cap.read()
    if not success:
        break

    #grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #avg_5x5 = np.ones((5, 5), np.float32) / 25.0
    #blur = cv.filter2D(grayscale, -1, avg_5x5)

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv.filter2D(img, -1, kernel)

    out.write(sharp)

cap.release()
out.release()
cv.destroyAllWindows()