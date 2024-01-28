import cv2 as cv
import random

trackerList = []

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    r = random.randint(0,256)
    g = random.randint(0,256)
    b = random.randint(0,256)
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

for i in range(2): 
    tracker = cv.legacy.TrackerCSRT_create()
    trackerList.append(tracker)

cap = cv.VideoCapture('frisbee.mp4')
static_cap = cv.VideoCapture('frisbee.mp4')

ret, img = cap.read()

""" fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv.VideoWriter("multiplayer.mp4", fourcc, cap.get(cv.CAP_PROP_FPS), (1280,720)) """

# Keeping track of players
ret, static_image = static_cap.read()

# Different coordinates used for each video
bboxes = []
player_images = []
f = open("playercoordinates.txt", "w")

for i in range(0, len(trackerList)):
    player = i + 1
    print('Select player ' + str(player))
    bbox = cv.selectROI('Select Players', img, False)
    bboxes.append(bbox)
    drawBox(img, bbox)

    # Captures images of each player
    cropped = static_image[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    
    # Stores images in list
    player_images.append(cropped)

    trackerList[i].init(img, bboxes[i])
    print('bbox index: ', bboxes[i])
    cv.destroyWindow('Select Players')


# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    # Loop through each bounding box
    for i in range(0, len(bboxes)):

        # If tracking was lost, select new ROI of player
        if (bboxes[i] == 0 or trackerList[i] == 0):
            print("PLAYER LOST!!!!!!!!")
            cv.imshow('Lost Player', player_images[i])
            bbox = cv.selectROI('Reselect Lost Player', img, False)
            print('BBOX: ', bbox)

            # If no coordinates were selected
            if bbox == (0, 0, 0, 0):
                continue

            # Insert new bbox into list and continue tracking
            else:
                bboxes[i] = bbox
                new_tracker = cv.legacy.TrackerCSRT_create()
                trackerList[i] = new_tracker
                trackerList[i].init(img, bboxes[i])
            cv.destroyWindow('Reselect Lost Player')
            cv.destroyWindow('Lost Player')

        else:
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
        f.write(str(bboxes) + "\n")
    #out.write(img)

cap.release()
static_cap.release()
#out.release()
f.close()
cv.destroyAllWindows()