""" Fire Detection based on HSV & Tracking People's Count using YOLO_V8 """
""" Program Scope
Fire Detection _ HSV
People Entry & Exit Count _ Yolo V8
Sound Alarm after Detection
Intimation of Fire and People Count & map coord after Detection thru mail/whatsapp to corresponding Recipients
"""

import cv2
import threading
import numpy as np
import playsound
from logo_add import logo_add
from yolov8_tracker import people_counter
from comms_func import send_email, send_whatsapp, map_func

# Output Write
output = cv2.VideoWriter("fire_detection_out.mp4", cv2.VideoWriter_fourcc(*"XVID"),45,(1440,640))

# Indicators
possibility_fire = False
fire = False
alarm_sts = False
email_sts = False
whatsapp_sts = False
map_sts = False
density_chck = False
frame_snap = None
mask_snap = None
i = 0

# Alarm Activation
def sound_alarm():
    while True:
        playsound.playsound('alarm_2.mp3',True)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('fire.mp4')
cap_1 = cv2.VideoCapture('people.mp4')

while True:
    has_frame, frame = cap.read()
    if not has_frame:
        threading.Event().set()           # Killing the thread
        break

    cap1_sts, frame_1, cnt_1 = people_counter(cap_1)
    if cap1_sts:
        frame_1 = cv2.resize(frame_1,(720,320))
        # cv2.imshow('frame_1',frame_1)
        # print(cnt_1)

    frame = cv2.resize(frame,(720,640))

    blur = cv2.GaussianBlur(frame,(21,21),0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)

    ## Tuned HSV values
    # lower = np.array([0, 74, 200], dtype=np.uint8)
    # upper = np.array([18, 166, 255], dtype=np.uint8)
    lower = np.array([5, 40, 220], dtype=np.uint8)
    upper = np.array([50, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv,lower,upper)
    mask_output = cv2.bitwise_and(frame,hsv,mask=mask)
    # cv2.imshow('masked_output',mask_output_re)

    area_red = int(cv2.countNonZero(mask))
    # print(area_red)
    if area_red > 5 and not fire:
        possibility_fire = True
        # cv2.waitKey(0)
        if area_red > 50:
            fire = True
            possibility_fire = False

    if possibility_fire:
        cv2.putText(frame,"Possibility of Fire",(250, 30),2,0.7,(255,255,0),1)

    if fire:
        cv2.rectangle(frame,(245,10),(400,35),(0,0,0),-1)
        cv2.putText(frame,"Fire Detected",(250, 30),2,0.7,(255,0,0),1)
        if (i % 8) == 0:
            cv2.rectangle(frame, (245, 10), (400, 35), (0, 0, 0), -1)
            cv2.putText(frame, "Fire Detected", (250, 30), 2, 0.7, (0, 0, 255), 1)

        cv2.rectangle(frame, (0, 35), (150, 75), (200, 200, 200), -1)
        cv2.putText(frame, "Entry Count: " + str(cnt_1[0]), (5, 50), 2, 0.55, (255, 0, 0), 1)
        cv2.putText(frame, "Exit  Count: " + str(cnt_1[1]), (5, 70), 2, 0.55, (0, 0, 255), 1)
        cv2.rectangle(frame, (525, 35), (710, 95), (216,191,216), -1)           # For Notifications

        if not alarm_sts:
            thread_1 = threading.Thread(target=sound_alarm,name='alarm',daemon=True).start()
            alarm_sts = True
        if not email_sts and density_chck and map_sts:
            thread_2 = threading.Thread(target=send_email,args=(cnt_1[0],), name='email', daemon=True).start()
            email_sts = True
        if not whatsapp_sts:
            thread_3 = threading.Thread(target=send_whatsapp, args=(cnt_1[0],), name='whatsapp', daemon=True).start()
            whatsapp_sts = True
        if not map_sts and density_chck:
            thread_4 = threading.Thread(target=map_func,args=(cnt_1[0],), name='map', daemon=True).start()
            map_sts = True

    if alarm_sts:
        cv2.putText(frame,"Alarm Activated",(530,50),2,0.60,(0,0,0),1)
    if whatsapp_sts:
        cv2.putText(frame, "Whatsapp Notified", (530, 70), 2, 0.60, (0,0,128), 1)
    if email_sts:
        cv2.putText(frame, "Email Sent", (530, 90), 2, 0.60,(139,0,0), 1)

    # # LinkedIn Profile Logo Add
    frame = logo_add(frame,(540,470))  # Mention the coordinates (h,w)
    # cv2.rectangle(frame, (500, 610), (720, 640), (170, 232, 238), -1)
    # cv2.putText(frame, "In: nishanth-deva", (505, 630), 2, 0.7, (128, 0, 0), 1)

    # Labelling the Frame
    cv2.rectangle(frame, (0, 610), (120, 640), (255, 255, 255), -1)
    cv2.putText(frame, "Feed: " + 'Inside', (0, 630), 2, 0.6, (0, 0, 0), 1)
    cv2.rectangle(mask_output, (0, 610), (130, 640), (255, 255, 255), -1)
    cv2.putText(mask_output, "Feed: " + 'Masked', (0, 630), 2, 0.6, (0, 0, 0), 1)

    # Minimum Fire Density check for Fire Snap
    if (area_red > 5400) and not density_chck:
        frame_snap = frame
        mask_snap = mask_output
        density_chck = True
        stacked = np.hstack([mask_snap,frame_snap])
        cv2.imwrite('fire_mask_snap.jpg',stacked)
        # cv2.waitKey(0)

    mask_output_re = cv2.resize(mask_output, (720, 320))
    vstack = np.vstack([frame_1, mask_output_re])
    # cv2.imshow("vstack", vstack)
    firedet_output = np.hstack([vstack,frame])
    # print(firedet_output.shape)
    cv2.imshow("Firedet_output", firedet_output)
    output.write(firedet_output)
    # cv2.imshow('frame', frame)

    i += 1
    if i == 100: i = 1

    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'): break

cap.release()
cap_1.release()
output.release()
print("Output Rendered")
cv2.destroyAllWindows()
