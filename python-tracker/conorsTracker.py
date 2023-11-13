import cv2 as cv
import numpy as np

#takes about 30 min to run on a 10 min video

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (255,0,0), 3,1)
    cv.putText(img, "Tracking", (120,75), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)

cap = cv.VideoCapture('Videos/CroppedThrow.mp4')

#412, 130, 17, 8 box coordinates from CroppedThrow frisbee box selection 

#455, 139, 16, 12 box coordinates from flowThrow frisbee box selection

#tracker = cv.legacy.TrackerBoosting_create()
tracker = cv.legacy.TrackerCSRT_create()
#tracker = cv.legacy.TrackerMIL_create()
#tracker = cv.legacy.TrackerMOSSE_create()

ret, img = cap.read()

#bbox = cv.selectROI(img, False)

#croppedThrow box for frisbee
bbox = 412, 130, 17, 8

#flowThrow box
#bbox = 455, 139, 16, 12

print(bbox)
tracker.init(img, bbox)
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv.VideoWriter("full_CSRT_sharpen.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720))

#setup for rudimentary center tracking
# we will start with 1/3 frames checking the center of the box
counter = 0
trackedCoords = open("coordsList.txt", "w")
trueCoords = open("groundTruth.txt", "w")
#I think we actually want txt files, not csv files
#trackedCoords = open('coordsList.csv', 'w')
#trueCoords = open('groundTruth.csv', 'w')
xcord = 0
ycord = 0
xcord2 = 0
ycord2 = 0

while True:
    #testers
    #print(bbox)
    #print(counter)
    success, img = cap.read()
    if not success:
        break

    success, bbox = tracker.update(img)

    if counter == 2 :
        #xcord of the center is x plus one half of the width
        xcord = bbox[0] + (bbox[2] / 2)
        #ycord of the center is y plus one half the height
        ycord = bbox[1] + (bbox[3] / 2)
        trackedCoords.write(str(xcord) + ',' + str(ycord) + '\n')
        
        counter = -1

        #get the HITL to input the center box coordinates
        bbox2 = cv.selectROI(img, False)
        #testing
        print(bbox2)
        xcord2 = bbox2[0] + (bbox2[2] / 2)
        ycord2 = bbox2[1] + (bbox2[3] / 2)
        trueCoords.write(str(xcord2) + ',' + str(ycord2) + '\n')
        




    counter += 1
    
    #couldn't find a way to combine preprocessing + box detection
    #although i just tried it quickly 

    if success:
        drawBox(img,bbox)

    out.write(img)

trackedCoords.close()
trueCoords.close()
cap.release()
out.release()
cv.destroyAllWindows()
