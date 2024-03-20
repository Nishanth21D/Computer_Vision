# PoseEstimate using MediaPipe Library
import cv2
import os
import time
import mediapipe as mp

"""The 33 pose landmarks."""
"""
NOSE = 0, LEFT_EYE_INNER = 1,LEFT_EYE = 2, LEFT_EYE_OUTER = 3, RIGHT_EYE_INNER = 4, RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6, LEFT_EAR = 7, RIGHT_EAR = 8, MOUTH_LEFT = 9, MOUTH_RIGHT = 10, LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12, LEFT_ELBOW = 13, RIGHT_ELBOW = 14, LEFT_WRIST = 15, RIGHT_WRIST = 16, LEFT_PINKY = 17
RIGHT_PINKY = 18, LEFT_INDEX = 19, RIGHT_INDEX = 20, LEFT_THUMB = 21, RIGHT_THUMB = 22, LEFT_HIP = 23
RIGHT_HIP = 24, LEFT_KNEE = 25, RIGHT_KNEE = 26, LEFT_ANKLE = 27, RIGHT_ANKLE = 28, LEFT_HEEL = 29
RIGHT_HEEL = 30, LEFT_FOOT_INDEX = 31, RIGHT_FOOT_INDEX = 32
"""
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

print(os.listdir("pose_videos"))
s = os.path.join("pose_videos","martial_2.mp4")
cap = cv2.VideoCapture(s if s else 0)

p_time = 0
win_name = "Pose Estimate"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame: break
    # print(frame.shape)
    h, w = int(frame.shape[0]/3), int(frame.shape[1]/3)
    frame = cv2.resize(frame, (w, h))
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)          # Pass the RGB_Image input to the pose
    # print(results.pose_landmarks)           # .pose_landmarks contains x,y,z points
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)
        #print(results.pose_landmarks.landmark)      ## .pose_landmarks.landmark contains each points x,y,z and visibility_confidence
        lm_list = []                             # Store pose_point,x,y
        for i, l_mark in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(l_mark.x * w), int(l_mark.y * h)   # scaling
            lm_list.append([i,cx,cy])
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)    # Draw all the pose_points
            # cv2.circle(frame, (lm_list[5][1], lm_list[5][2]), 4, (0, 0, 255), -1)
            #print(lm_list)
            roi = 14                                 # Specify the  interested pose_point
            if len(lm_list) > roi:                  # To draw the particular the pose_point
                roi -= 1
                cv2.circle(frame, (lm_list[roi][1], lm_list[roi][2]), 12, (0, 0, 255), -1)


    c_time = time.time()
    fps = int(1 / (c_time - p_time))    # Calculating FPS
    p_time = c_time
    cv2.putText(frame,str(fps),(50,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)

    cv2.imshow(win_name, frame)

cap.release()
cv2.destroyAllWindows()
