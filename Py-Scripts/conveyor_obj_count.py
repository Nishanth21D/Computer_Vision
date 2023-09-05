""" Simple Conveyor Script to track and count the Objects """
import os
import cv2
import time

pTime = 0
li = []

src = os.path.join("conveyor_videos","conveyor_4.mp4")
cap = cv2.VideoCapture(src)
vids_output  = cv2.VideoWriter("Coveyor_Obj_Count.mp4", fourcc=cv2.VideoWriter_fourcc(*"XVID"),fps=30, frameSize=(1280,720))
while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame: break
    h,w = int(frame.shape[0]/3), int(frame.shape[1]/3)
    frame=cv2.resize(frame, (w,h))
    # plt.imshow(frame),plt.show()
    # x = 300, y = 14
    # x2 = 938, y2 = 186
    belt = frame[14:186, 300:938] # Always [y:y2,x:x2]
    gray_belt = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_belt, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (area > 7000):
            # print(area)
            cv2.rectangle(belt,(x,y), (x+w, y+h), (0,255,0), 2, -1)     # bbox around the object
            # cv2.putText(belt, str(area), (x,y), 1,2,(0,0,255), 2)   # specify area around the object
            li.append(area)
    # cv2.imshow("Belt", belt)
    # cv2.imshow("Gray_Belt", threshold)
    cTime = time.time()
    fps = int(1 / (cTime - pTime))  # Calculating FPS
    pTime = cTime

    cv2.putText(frame, "FPS: "+str(fps), (20,50),1,1.5,(255,0,0),2)
    cv2.putText(frame, "Count: " + str(len(li)), (20, 80), 1, 1.5, (0, 0, 255), 2)
    cv2.imshow("Conveyor", frame)
    vids_output.write(frame)

# print(len(li))
cap.release()
vids_output.release()
cv2.destroyAllWindows()
