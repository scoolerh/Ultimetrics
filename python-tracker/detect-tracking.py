import cv2 as cv
import random
from detection import detect
trackerList = []

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    r = random.randint(0,256)
    g = random.randint(0,256)
    b = random.randint(0,256)
    cv.rectangle (img,(x,y), ((x+w), (y+h)), (r,g,b), 3,1)

cap = cv.VideoCapture('frisbee.mp4')
static_cap = cv.VideoCapture('frisbee.mp4')

ret, img = cap.read()

# Keeping track of players
ret, static_image = static_cap.read()

# Different coordinates used for each video
bboxes = detect(img)
player_images = []
f = open("results/playercoordinates.txt", "w")

for i in range(0, len(bboxes)):
    tracker = cv.legacy.TrackerCSRT_create()
    trackerList.append(tracker)
    drawBox(img, bboxes[i])

    # Captures images of each player
    cropped = static_image[bboxes[i][1]:(bboxes[i][1] + bboxes[i][3]), bboxes[i][0]:(bboxes[i][0] + bboxes[i][2])]
    player_images.append(cropped)

    trackerList[i].init(img, bboxes[i])

print("Currently tracking the players...")

# Loop through video
while True:
    success, img = cap.read()
    if not success:
        break

    for i in range(0, len(bboxes)):

        # If tracking was lost, select new ROI of player
        if (bboxes[i] == 0 or trackerList[i] == 0):
            print("Player was lost!")
            cv.imshow('Lost Player', player_images[i])
            bbox = cv.selectROI('Reselect Lost Player', img, False)

            if bbox == (0, 0, 0, 0):
                continue
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

            if success:
                drawBox(img, bboxes[i])
            else:
                bboxes[i] = -1
        
    positions = []
    for box in bboxes: 
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        xcord = x + (w/2)
        center = (xcord, y)
        positions.append(center)
    f.write(str(positions) + "\n")

print("Player tracking complete!")

cap.release()
static_cap.release()
f.close()
cv.destroyAllWindows()