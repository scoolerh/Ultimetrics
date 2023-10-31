import cv2 as cv

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (255,0,0), 3,1)
    cv.putText(img, "Tracking", (120,75), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)

'''
Select which video to use
'''
# cap = cv.VideoCapture("Videos/CroppedFinalDance.mp4") 
# cap = cv.VideoCapture('Videos/cross.mp4')
# cap = cv.VideoCapture('Videos/running.mp4')
cap = cv.VideoCapture('Videos/throw.mp4')


'''
Select which algorithm to use
'''
# tracker = cv.legacy.TrackerBoosting_create()
tracker = cv.legacy.TrackerMIL_create()
# tracker = cv.legacy.TrackerCSRT_create()
# tracker = cv.legacy.TrackerKCF_create()
# tracker = cv.legacy.TrackerTLD_create()
# tracker = cv.legacy.TrackerMedianFlow_create()
# tracker = cv.legacy.TrackerMOSSE_create()

ret, img = cap.read()


'''
Select which bounding box to use
'''
bbox = cv.selectROI(img, False)
# print(bbox)
# bbox = (676, 254, 17, 19)
# running_bbox = (553, 151, 56, 80)
# cross_bbox = (324, 111, 38, 86)
# throw_bbox = (672, 287, 53, 102)
tracker.init(img, bbox)



fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv.VideoWriter("running_frisbee_MIL.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720))

while True:
    success, img = cap.read()
    if not success:
        break

    success, bbox = tracker.update(img)
    timer = cv.getTickCount()

    if success:
        drawBox(img,bbox)
    else:
        cv.putText(img, "Tracking lost", (120,75), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
      
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
    cv.putText(img, str(int(fps)), (120,100), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    out.write(img)

cap.release()
out.release()
cv.destroyAllWindows()