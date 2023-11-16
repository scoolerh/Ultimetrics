import cv2 as cv

cap = cv.VideoCapture('Videos/frisbee.mp4')

tracker = cv.legacy.TrackerBoosting_create()
tracker2 = cv.legacy.TrackerCSRT_create()
tracker3 = cv.legacy.TrackerMIL_create()

ret, img = cap.read()

bbox = cv.selectROI(img, False)

tracker.init(img, bbox)
tracker2.init(img, bbox)
tracker3.init(img, bbox)

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

    if counter == 9:
        #get coords of the algos' bboxes 
        xcordB = bboxB[0] + (bboxB[2] / 2)
        ycordB = bboxB[1] + (bboxB[3] / 2)
        trackedCoordsBoosting.write(str(xcordB) + ',' + str(ycordB) + '\n')
        xcordC = bboxC[0] + (bboxC[2] / 2)
        ycordC = bboxC[1] + (bboxC[3] / 2)
        trackedCoordsCSRT.write(str(xcordC) + ',' + str(ycordC) + '\n')
        xcordM = bboxM[0] + (bboxM[2] / 2)
        ycordM = bboxM[1] + (bboxM[3] / 2)
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
