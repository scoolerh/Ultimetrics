import cv2 as cv

cap = cv.VideoCapture('Videos/layoutish.mp4')

#412, 130, 17, 8 box coordinates from CroppedThrow frisbee box selection 
#455, 139, 16, 12 box coordinates from flowThrow frisbee box selection

tracker = cv.legacy.TrackerBoosting_create()
tracker2 = cv.legacy.TrackerCSRT_create()
tracker3 = cv.legacy.TrackerMIL_create()

ret, img = cap.read()

bbox = cv.selectROI(img, False)

#croppedThrow box for frisbee
#bbox = 412, 130, 17, 8

#flowThrow box
#bbox = 455, 139, 16, 12

print("Bounding box coordinates: " + str(bbox))
tracker.init(img, bbox)
tracker2.init(img, bbox)
tracker3.init(img, bbox)

#setup for rudimentary center tracking
# we will start with 1/5 frames checking the center of the box
counter = 0
trackedCoordsBoosting = open("coordsListBoosting.txt", "w")
trackedCoordsCSRT = open("coordsListCSRT.txt", "w")
trackedCoordsMIL = open("coordsListMIL.txt", "w")
trueCoords = open("groundTruth.txt", "w")
xcordB = 0
ycordB = 0
xcordC = 0
ycordC = 0
xcordM = 0
ycordM = 0
xcordt = 0
ycordt = 0

while True:
    success, img = cap.read()
    if not success:
        break

    success, bboxB = tracker.update(img)
    success, bboxC = tracker2.update(img)
    success, bboxM = tracker3.update(img)

    if counter == 4 :
        xcordB = bboxB[0] + (bboxB[2] / 2)
        ycordB = bboxB[1] + (bboxB[3] / 2)
        xcordC = bboxC[0] + (bboxC[2] / 2)
        ycordC = bboxC[1] + (bboxC[3] / 2)
        xcordM = bboxM[0] + (bboxM[2] / 2)
        ycordM = bboxM[1] + (bboxM[3] / 2)
        trackedCoordsBoosting.write(str(xcordB) + ',' + str(ycordB) + '\n')
        trackedCoordsCSRT.write(str(xcordC) + ',' + str(ycordC) + '\n')
        trackedCoordsMIL.write(str(xcordM) + ',' + str(ycordM) + '\n')

        counter = -1

        #get the HITL to input the center box coordinates
        bboxt = cv.selectROI(img, False)
        xcordt = bboxt[0] + (bboxt[2] / 2)
        ycordt = bboxt[1] + (bboxt[3] / 2)
        trueCoords.write(str(xcordt) + ',' + str(ycordt) + '\n')

    counter += 1

trackedCoordsBoosting.close()
trackedCoordsCSRT.close()
trackedCoordsMIL.close()
trueCoords.close()
cap.release()
cv.destroyAllWindows()
