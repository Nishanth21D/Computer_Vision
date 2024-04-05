""" YOLO V8 _ Detection and Tracking Count """
""" Program Scope:
Uses YoloV8 model for people's detection and tracking-id. 
Then uses the tracking-id to compute the entry and exit count.
Return (Frame_status, Frame, [(entry_length, exit_length)])
"""
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

model_1 = YOLO('yolov8s.pt')
model_1.info()
# class_file = open('coco.names','r')
# classes = class_file.read().split("\n")
# print(classes)
device = 'CUDA:0' if torch.cuda.is_available() else 'CPU'
print("Using Device: ", device)
ent_list = []
ext_list = []


def people_counter(cap1):
    has_frame1, frame1 = cap1.read()
    if not has_frame1:
        return False, 0, [0,0]
    frame1 = cv2.resize(frame1, (720, 480))
    # roi = cv2.selectROI('ROI',frame1)                 # Extracting the roi co-ord manually on screen
    # print(roi)
    roi = frame1[131:475,3:399]
    roi_poly = [(210, 369), (215, 152), (380, 122), (369, 466)]
    # results = model_1(roi,stream=True)
    # results = model_1.predict(roi,stream=True,conf=0.7,classes=[0],show=True,device='cpu',verbose=False) # Prediction
    results = model_1.track(roi,stream=True,conf=0.8,iou=0.2,classes=[0],device=device,persist=True,
                            verbose=False,show=False)   # Prediction with Tracking
    bbox_list = []
    for res in results:
        boxes = res.boxes
        for box in boxes:
            # print(box.xywh[0])
            x1, y1, x2, y2 = box.xywh[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if box.id is not None:
                bbox_list.append([x1, y1, x2, y2, int(box.id)])
            # cv2.rectangle(frame1,(x1, y1),(x1+x2, y1+y2),(255,0,255),3)
        # print(bbox_list)
        for bbid in bbox_list:
            x, y, w, h, bid = bbid
            # cv2.putText(frame1, str(bid), (x, y), 1, 2, (0,255, 255), 2)
            results = cv2.pointPolygonTest(np.array(roi_poly, np.int32), (x, y), False)
            if (results > 0) and (bid not in ent_list):
                ent_list.append(bid)
            if results < 0:
                if (bid not in ext_list) and (bid in ent_list):
                    ext_list.append(bid)
                    ent_list.remove(bid)
            # print(ent_list,ext_list)

    # cv2.imshow('roi', roi)
    cv2.polylines(frame1,[np.array(roi_poly, np.int32)],True,(0,255,0),4)
    cv2.rectangle(frame1,(0,35),(150,75),(200,200,200),-1)
    cv2.putText(frame1,"Entry Count: "+str(len(ent_list)), (5,50), 2, 0.55, (255,0,0), 1)
    cv2.putText(frame1, "Exit  Count: "+str(len(ext_list)), (5,70), 2, 0.55, (0,0,255), 1)
    cv2.putText(frame1,'Using Device: '+device,(300,20),2,0.55,(0,255,255),1)

    cv2.rectangle(frame1, (0, 440), (130, 480), (255, 255, 255), -1)
    cv2.putText(frame1, "Feed: " + 'Outside', (0, 465), 2, 0.6, (0, 0, 0), 1)
    # cv2.imshow('frame_1', frame1)

    return True, frame1, ([len(ent_list),len(ext_list)])        # Returns Frame_status, frame, [Length of entry & exit]


if __name__ == "__main__":
    cap_1 = cv2.VideoCapture('Fire Detection/people.mp4')
    while cv2.waitKey(1) != 27:
        cap1_sts, frame_1, cnt = people_counter(cap_1)
        if not cap1_sts:
            break
        print(cnt)
        cv2.imshow('frame', frame_1)
    cap_1.release()
    cv2.destroyAllWindows()
