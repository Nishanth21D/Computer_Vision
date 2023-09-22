""" Script to detect objects using YOLO_V5 and tracks the object using DeepSort, then counts it. """
import cv2
import os
import matplotlib.pyplot as plt
import time
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from custom_ObjDetection import yoloV5_detector

# Calling the Yolo_V5 detector class
detector = yoloV5_detector(conf=0.6,spcfc_cls=[2,7],model_name='yolov5x')
tracker = DeepSort(max_age=1000,n_init=1000,max_cosine_distance=1.3,max_iou_distance=0.7)               # Tracker Initialization
src = os.path.join("highway","h8.mp4")
cap = cv2.VideoCapture(src)
fps = int(cap.get(cv2.CAP_PROP_FPS))
border = 110            # BorderLine - Downsized it as it will be validated on the resized frame
divider = 650           # Divider - Downsized it as it will be validated on the resized frame
out_list = [[],[]]      # Tracks the Exit vehicle counter of [car][truck]
in_list = [[],[]]       # Tracks the Entry vehicle counter of [car][truck]
out_list_1 = []         # Exit vehicle track-id counter
in_list_1 = []          # Entry vehicle track-id counter

# print(fps)
output = cv2.VideoWriter("vehicle_multi_track_out.mp4", cv2.VideoWriter_fourcc(*"XVID"),12,(1280,720))

while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame:
        break
    # print(frame.shape)
    roi = frame[170:570, 0:1250]
    start = time.perf_counter()
    _, result = detector.yolo_v5(roi)
    tracked_objs = tracker.update_tracks(raw_detections=result, frame=roi)
    for track in tracked_objs:
        if track.is_confirmed():
            continue
        # print(track.track_id)
        trid = track.track_id               # Extracts tracked objects unique id
        cls = track.get_det_class()         # Extract tracked class
        score = track.get_det_conf()        # Extract tracked conf_score
        (x,y,w,h) = track.to_ltrb()      # Extract bbox(l,w,r,h)
        (x, y, w, h) = (int(x), int(y), int(w), int(h))
        # print(x,y)
        cx, cy = int(x+(w-x)/2), int(y+(h-y)/2)
        # print(cx, cy,trid)        # centre points
        if border >= cy >= (border-5):
            if trid not in out_list_1 and (cx >= divider):
                out_list_1.append(trid)
                if cls == 'car':
                    out_list[0].append([trid,cls])
                else:
                    out_list[1].append([trid,cls])
        elif border <= cy <= (border+5):
            if trid not in in_list_1 and (cx <= divider):
                in_list_1.append(trid)
                if cls == 'car':
                    in_list[0].append([trid, cls])
                else:
                    in_list[1].append([trid, cls])

        cv2.circle(roi, (cx, cy), 2, (0, 255, 255), -1)
        cv2.rectangle(roi, pt1=(x, y), pt2=(w, h), color=(0, 255, 0), thickness=2)
        cv2.putText(roi, cls, (x, y - 7), 2, 0.6, (255,0,0), 2)
        # cv2.putText(roi, str(trid), (x+60, y-3), 2, 0.7, (0,0, 255), 1)
        # cv2.putText(roi, str(round(score, 2)), (x, y + 15), 2, 0.5, (255, 255, 0), 1)

    # print("Out_Car: ",len(out_list[0]),"Out_Truck: ",len(out_list[1]))
    # print("In_Car: ",len(in_list[0]),"In_Truck: ",len(in_list[1]))

    cv2.line(frame,(425,280),(920,280),(100,255,255),1)             # Detecting Border line
    cv2.rectangle(frame,(40,260), (180,350),(100,220,220),-1)                                 # Entry Box Highlighter
    cv2.rectangle(frame, (1020, 260), (1160, 350), (100, 220, 220), -1)                       # Exit Box Highlighter
    cv2.putText(frame,'Exit Counter:',(1025,274),2,0.6,(0,0,0),1)
    cv2.putText(frame, 'Entry Counter:', (43, 274), 2, 0.6, (0, 0, 0), 1)
    cv2.putText(frame, 'Car Count   :'+str(len(out_list[0])), (1025, 310), 2, 0.55, (0, 0, 255), 1)
    cv2.putText(frame, 'Truck Count :' + str(len(out_list[1])), (1025, 330), 2, 0.55, (0, 0, 255), 1)
    cv2.putText(frame, 'Car Count   :' + str(len(in_list[0])), (43, 310), 2, 0.55, (255, 0, 0), 1)
    cv2.putText(frame, 'Truck Count :' + str(len(in_list[1])), (43, 330), 2, 0.55, (255, 0, 0), 1)

    end = time.perf_counter()
    fps = int(1 / (end - start))
    cv2.putText(frame, "FPS: " + str(fps), (20, 50), 1, 1.5, (0, 0, 0), 2)

    # cv2.imshow("ROI", roi)
    cv2.imshow("Main_Win", frame)
    frame = cv2.resize(frame, (1280, 720))
    output.write(frame)
    # cv2.waitKey(0)

cap.release()
output.release()
print("done")
cv2.destroyAllWindows()