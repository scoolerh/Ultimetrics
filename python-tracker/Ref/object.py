import cv2 as cv
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv.dnn_DetectionModel(frozen_model, config_file)

print("Model: ", model)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)

# Video

# cap = cv.VideoCapture('Videos/CroppedFinalDance.mp4')
cap = cv.VideoCapture('Videos/test.mp4')
if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cant open video')

ClassIndex = []
tracker = cv.legacy.TrackerMOSSE_create()

def detect_objects(frame):

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)

    if(len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if(ClassInd <= 80):
                print("Frame1: ", frame)
                print("Boxes1: ", boxes)
                cv.rectangle(frame, boxes, (255, 0, 0), 2)

    return frame, bbox

def track_objects(frame, bbox):
    for box in bbox:
        ok = tracker.init(frame, box)
        ok, box = tracker.update(frame)
        cv.rectangle(frame, box, (255, 0, 0), 2)
    return frame, bbox

test = 0
box = (10,10,10,10)
while True:
    ret, frame = cap.read()
    # ok = tracker.init(frame, box)
    if test > 0: 
        frame, bbox = track_objects(frame, bbox)
        # print("Tracker")
        # ok, bbox = tracker.update(frame)
        # print("Bbox before zip: ", type(bbox))
        # for boxes in zip(bbox):
        #     print("Frame2: ", frame)
        #     print("Boxes2: ", boxes)
        #     cv.rectangle(frame, boxes, (255, 0, 0), 2)
    else:
        print("Detection")
        frame, bbox = detect_objects(frame)
        test += 1
        print("Frame: ", frame)
        print("Bbox: ", bbox)

    cv.imshow('Object detection', frame)

    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()