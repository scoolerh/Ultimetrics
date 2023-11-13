import cv2 as cv

trackerList = []

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (255,0,0), 3,1)

tracker1 = cv.legacy.TrackerBoosting_create()
tracker2 = cv.legacy.TrackerBoosting_create()
tracker3 = cv.legacy.TrackerBoosting_create()
tracker4 = cv.legacy.TrackerBoosting_create()
tracker5 = cv.legacy.TrackerBoosting_create()
tracker6 = cv.legacy.TrackerBoosting_create()
# tracker7 = cv.legacy.TrackerCSRT_create()
# tracker8 = cv.legacy.TrackerCSRT_create()
# tracker9 = cv.legacy.TrackerCSRT_create()
# tracker10 = cv.legacy.TrackerCSRT_create()
# tracker11 = cv.legacy.TrackerCSRT_create()
# tracker12 = cv.legacy.TrackerCSRT_create()
# tracker13 = cv.legacy.TrackerCSRT_create()
# tracker15 = cv.legacy.TrackerCSRT_create()
# tracker14 = cv.legacy.TrackerCSRT_create()

trackerList.append(tracker1)
trackerList.append(tracker2)
trackerList.append(tracker3)
trackerList.append(tracker4)
trackerList.append(tracker5)
trackerList.append(tracker6)
# trackerList.append(tracker7)
# trackerList.append(tracker8)
# trackerList.append(tracker9)
# trackerList.append(tracker10)
# trackerList.append(tracker11)
# trackerList.append(tracker12)
# trackerList.append(tracker13)
# trackerList.append(tracker14)
# trackerList.append(tracker15)

# cap = cv.VideoCapture('Videos/cross.mp4')
# cap = cv.VideoCapture('Videos/throw.mp4')
cap = cv.VideoCapture('Videos/running.mp4')

ret, img = cap.read()

# Different coordinates used for each video
bboxes = []
# throw_bboxes = [(439, 238, 42, 95), (513, 269, 35, 90), (504, 386, 50, 60), (685, 288, 34, 100), (688, 209, 36, 85), (670, 216, 24, 68)]
# cross_bboxes = [(327, 111, 31, 83), (366, 135, 34, 82), (507, 158, 30, 74), (467, 192, 32, 70), (464, 257, 35, 103), (692, 262, 34, 93)]
# running_bboxes = [(399, 144, 40, 76), (471, 110, 29, 73), (567, 151, 33, 80), (571, 278, 44, 94), (487, 336, 57, 119), (427, 401, 45, 110)]

for i in range(0, len(trackerList)):
    bbox = cv.selectROI(img, False)
    bboxes.append(bbox)
    trackerList[i].init(img, bboxes[i])


# initial_bboxes = bboxes
# print(bboxes)
# quit()


fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv.VideoWriter("FRISBEE_RUNNING_BOOSTING.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720))

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    # Loop through each bounding box
    for i in range(0, len(bboxes) - 1):
        # If tracking was lost go to next box
        if (bboxes[i] == 0 or trackerList[i] == 0):
            continue

        else:
            # Update box
            success, bboxes[i] = trackerList[i].update(img)

            # Update successful
            if success:
                drawBox(img, bboxes[i])
                
            # Unsuccessful
            else:
                bboxes[i] = 0
                trackerList[i] = 0
        
        print('bboxes: ', bboxes)
        print('trackerList: ', trackerList)
    out.write(img)


cap.release()
out.release()
cv.destroyAllWindows()