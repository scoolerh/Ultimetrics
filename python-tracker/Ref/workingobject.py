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

cap = cv.VideoCapture('Videos/CroppedFinalDance.mp4')
if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cant open video')


while True:
    ret, frame = cap.read()
    print(frame)
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)

    # print("Class Index: ", ClassIndex)
    # print("Confidence: ", confidence)
    print("Bbox: ", bbox)


    if(len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if(ClassInd <= 80):
                print(boxes)
                cv.rectangle(frame, boxes, (255, 0, 0), 2)

    cv.imshow('Object detection', frame)

    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()