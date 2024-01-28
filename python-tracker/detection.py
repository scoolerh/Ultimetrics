import yolov5
import cv2 as cv

def detect(image):
    # load model
    model = yolov5.load('keremberke/yolov5m-football')
    
    # set model parameters
    model.conf = 0.45  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image


    # perform inference
    results = model(image, size=640)

    # inference with test time augmentation
    results = model(image, augment=True)

    # parse results
    bboxes = []
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    boxes.tolist()

    for i in range(0, len(boxes)):
        x1 = round(boxes[i][0].item())
        y1 = round(boxes[i][1].item())
        width = round(boxes[i][2].item()) - x1
        height = round(boxes[i][3].item()) - y1
        bboxes.append((x1, y1, width, height))

    scores = predictions[:, 4]

    categories = predictions[:, 5]

    return bboxes