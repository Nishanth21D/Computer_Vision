import math
import cv2
import os
import time
import matplotlib.pyplot as plt

from custom_ObjDetection import yolo_detector

detector = yolo_detector(0.8,0.5)
classes = detector.load_class(cls_path="coco.names")
pTime = 0
center_pts = {}
count = 0
to_enter = []
enter_id = []
exit_id = []
ent_ext = {}
enter_area = 200
exit_area = 390
dist = 20

src = os.path.join("highway","highway_2.mp4")
vids_output = cv2.VideoWriter("tracking_result.mp4", fourcc=cv2.VideoWriter_fourcc(*"XVID"),fps=25, frameSize=(960,540))
cap = cv2.VideoCapture(src)
while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame: break
    height, width = int(frame.shape[0] / 2), int(frame.shape[1] / 2)
    # print(w,h)
    frame = cv2.resize(frame,(width,height))
    # plt.imshow(frame); plt.show()
    clsid,score,bbox = detector.yolo_v4(frame)
    detections = []
    for (cls,scr,box) in zip (clsid,score,bbox):
        if classes[cls] in ['bicycle', 'car', 'motorbike', 'bus', 'truck']:
            # print(cls)
            x,y,w,h = box
            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            # print(cx, cy)        # centre points
            cv2.circle(frame, (cx,cy),3,(0,255,255),-1)
            cv2.putText(frame, str(classes[cls]), (cx+15,cy), 2, 0.7, (255,0,255),2)
            same_obj = False
            for id,pt in center_pts.items():
                dist = math.hypot(cx-pt[0], cy-pt[1])
                if dist < 20:
                    center_pts[id] = (cx,cy)
                    # print(center_pts)
                    detections.append([x,y,w,h,id])
                    same_obj = True
                    break
            if same_obj is False:
                center_pts[count] = (cx,cy)
                detections.append([x,y,w,h,count])
                count += 1

    # Ignore the Below loop as it created for calculating the vehicle's Speed. Needed little modification.
    # Commenting the below loop will also the code works.
    for id,pt in center_pts.items():
        if pt[1] < enter_area:
            if id not in to_enter:
                to_enter.append(id)
        elif pt[1] > enter_area and (id in to_enter):
            ent_time = time.time()
            enter_id.append(id)
            ent_ext[id] = [ent_time]
            to_enter.remove(id)
        elif pt[1] >= exit_area and (id in enter_id):
            elapsed = time.time() - ent_ext[id][0]
            # exit_id.append([id,ext_time,pt])
            speed_ms = dist/elapsed
            distance = speed_ms * 3.6                   # calc km/h
            ent_ext[id] = [ent_ext[id][0],distance,pt]
            if pt[1] > 462:
                enter_id.remove(id)
                ent_ext.pop(id)
    # print(center_pts)
    # print(to_enter)
    # print(enter_id)
    # print(exit_id)
    # print(ent_ext)
    # print(detections)
    # bbid = tracker.trackobj(detections)
    bbid = detections
    for id in bbid:
        x,y,w,h,bb = id
        cv2.rectangle(frame,pt1=(x, y), pt2=(x+w, y+h), color=(0,255,0), thickness=2)
        cv2.putText(frame, str(bb), (x, y-10), 1, 1.5, (255,0,255), 2)

    # cv2.line(frame,(180,200),(960,200), (0,0,255),1,cv2.LINE_4)
    # cv2.line(frame, (220, 390), (960, 390), (0, 0, 255), 1, cv2.LINE_4)
    # print(len(curr_pnts))
    # cTime = time.time()
    # fps = int(1/ (cTime - pTime))
    # cv2.putText(frame,"FPS: "+str(fps), (20,50), 1,2,(255,0,0),2)
    # pTime = cTime
    # cv2.imshow("Image",frame)
    vids_output.write(frame)
    # cv2.waitKey(0)

print("done")
cap.release()
vids_output.release()
cv2.destroyAllWindows()
