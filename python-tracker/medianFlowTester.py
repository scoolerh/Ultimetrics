#import the requisite libraries
import cv2
import sys
print("dingdong")
#not sure what this does
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def main():
    #set up tracker
    print("ding")
    
    tracker = cv2.Tracker_create('MEDIANFLOW')
    tracker = cv2.Tracker.init()
    #read in the video
    video = cv2.VideoCapture("/Users/cslab/Desktop/frisbee.MP4")

    #exit if video not opened
    if not video.isOpened() :
        print("Couldn't open the video")
        sys.exit()

    #read first frame
    ok, frame = video.read()
    if not ok:
        print('unable to read file')
        sys.exit()

    #define initial bounding box
    bbox = (287, 23, 86, 320)
    bbox = cv2.selectROI(frame, False)

    #initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True :
        ok,frame = video.read()
        if not ok:
                break
        timer = cv2.getTickCount()

        ok, bbox = tracker.update(frame)
        #calculate FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if ok: 
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0),2,1)
        else :
            cv2.putText(frame, "trackign failure", (100,80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255),2)
        k = cv2.waitKey(1) & 0xff
        if k == 22 : break

if __name__ == "__main__" :
     main()
