import cv2 as cv
from robo import object_detection
trackerList = []

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (255,0,0), 3,1)

tracker1 = cv.legacy.TrackerCSRT_create()
tracker2 = cv.legacy.TrackerCSRT_create()
tracker3 = cv.legacy.TrackerCSRT_create()
tracker4 = cv.legacy.TrackerCSRT_create()
tracker5 = cv.legacy.TrackerCSRT_create()
# tracker6 = cv.legacy.TrackerCSRT_create()
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
# trackerList.append(tracker6)
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
# cap = cv.VideoCapture('Videos/running.mp4')
# static_cap = cv.VideoCapture('Videos/running.mp4')
cap = cv.VideoCapture('Videos/cross.mp4')
static_cap = cv.VideoCapture('Videos/cross.mp4')

ret, img = cap.read()

# Keeping track of players
ret, static_image = static_cap.read()

# Different coordinates used for each video
bboxes = []
player_images = []
# throw_bboxes = [(439, 238, 42, 95), (513, 269, 35, 90), (504, 386, 50, 60), (685, 288, 34, 100), (688, 209, 36, 85), (670, 216, 24, 68)]
# cross_bboxes = [(327, 111, 31, 83), (366, 135, 34, 82), (507, 158, 30, 74), (467, 192, 32, 70), (464, 257, 35, 103), (692, 262, 34, 93)]
# running_bboxes = [(399, 144, 40, 76), (471, 110, 29, 73), (567, 151, 33, 80), (571, 278, 44, 94), (487, 336, 57, 119), (427, 401, 45, 110)]


initial_image = "initial_image.jpg"
cv.imwrite("initial_image.jpg", img)

json = object_detection(initial_image)

x = []
y = []
width = []
height = []
object = []
for item in json['predictions']:
    x.append(item['x'])
    y.append(item['y'])
    width.append(item['width'])
    height.append(item['height'])
    object.append(item['class'])


# cv.rectangle(img, (round(x[0] - width[0] / 2), round(y[0] - height[0] / 2)), (round(x[0] + width[0] / 2), round(y[0] + height[0] / 2)), (255, 0, 0))

for i in range(0, len(x)):
    x_topleft = round(x[i] - width[i] / 2)
    y_topleft = round(y[i] - height[i] / 2)
    box_width = round(x[i] + width[i] / 2) - x_topleft
    box_height = round(y[i] + height[i] / 2) - y_topleft
    bbox = (x_topleft, y_topleft, box_width, box_height)
    # cv.rectangle(img, (round(x[i] - width[i] / 2), round(y[i] - height[i] / 2)), (round(x[i] + width[i] / 2), round(y[i] + height[i] / 2)), (0, 0, 255))
    bboxes.append(bbox)
    trackerList[i].init(img, bboxes[i])

# test_tracker = cv.legacy.TrackerCSRT_create()
# test_tracker.init(img, bboxes[0])
# roi_bbox = cv.selectROI(img, False)
# print("roi_bbox:", roi_bbox)
# print(bboxes)
# cv.imshow("image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# quit()
    

# Manual Object Detection
# for i in range(0, len(trackerList)):
#     bbox = cv.selectROI(img, False)
#     bboxes.append(bbox)
    

#     # Captures images of each player
#     cropped = static_image[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    
#     player_images.append(cropped)

#     trackerList[i].init(img, bboxes[i])
#     print('bbox index: ', bboxes[i])


fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv.VideoWriter("TESTINGTESTING.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720))

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    # Loop through each bounding box
    for i in range(0, len(bboxes)):

        # If tracking was lost, select new ROI of player
        # if (bboxes[i] == 0 or trackerList[i] == 0):
        #     print("PLAYER LOST!!!!!!!!")
        #     # cv.imshow('Lost Player', player_images[i])
        #     # bbox = cv.selectROI(img, False)
        #     # print('BBOX: ', bbox)

        #     # If no coordinates were selected
        #     if bbox == (0, 0, 0, 0):
        #         continue

        #     # Insert new bbox into list and continue tracking
        #     else:
        #         bboxes[i] = bbox
        #         new_tracker = cv.legacy.TrackerCSRT_create()
        #         trackerList[i] = new_tracker
        #         trackerList[i].init(img, bboxes[i])

        # else:
        # Update box
        success, bboxes[i] = trackerList[i].update(img)

        # Update successful
        if success:
            drawBox(img, bboxes[i])
            
        # Unsuccessful
        else:
            bboxes[i] = 0
            # trackerList[i] = 0
        
        print('bboxes: ', bboxes)
        print('trackerList: ', trackerList)
    out.write(img)

cap.release()
static_cap.release()
out.release()
cv.destroyAllWindows()