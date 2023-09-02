""""" Custom Pose Estimation Module to be used """""

""" The 33 pose landmarks."""
"""
NOSE = 0, LEFT_EYE_INNER = 1,LEFT_EYE = 2, LEFT_EYE_OUTER = 3, RIGHT_EYE_INNER = 4, RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6, LEFT_EAR = 7, RIGHT_EAR = 8, MOUTH_LEFT = 9, MOUTH_RIGHT = 10, LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12, LEFT_ELBOW = 13, RIGHT_ELBOW = 14, LEFT_WRIST = 15, RIGHT_WRIST = 16, LEFT_PINKY = 17
RIGHT_PINKY = 18, LEFT_INDEX = 19, RIGHT_INDEX = 20, LEFT_THUMB = 21, RIGHT_THUMB = 22, LEFT_HIP = 23
RIGHT_HIP = 24, LEFT_KNEE = 25, RIGHT_KNEE = 26, LEFT_ANKLE = 27, RIGHT_ANKLE = 28, LEFT_HEEL = 29
RIGHT_HEEL = 30, LEFT_FOOT_INDEX = 31, RIGHT_FOOT_INDEX = 32
"""

# Libraries
import cv2
import mediapipe as mp
import os
import time
import math

class PoseDetector():
    def __init__(self, mode=False, complexity=1, smoothness=True,
                 segmentation=False, smooth_seg=True, detconf=0.5, trackconf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smoothness = smoothness
        self.segmentation = segmentation
        self.smooth_seg = smooth_seg
        self.detconf = detconf
        self.trackconf = trackconf
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothness, self.segmentation,
                                     self.smooth_seg, self.detconf, self.trackconf)

    def findPose(self,frame,resize=False):
        if resize:
            h, w = int(frame.shape[0]/3),int(frame.shape[1]/3)
            frame = cv2.resize(frame,(w,h))
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(frame_rgb)       # Pass the RGB_Image input to the pose
        # print(result.pose_landmarks)              # .pose_landmarks contains x,y,z points
        if self.result.pose_landmarks:
            self.mpDraw.draw_landmarks(image=frame, landmark_list=self.result.pose_landmarks, connections=self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPosition(self, frame, pos=None, draw=False):
        self.lmark_list = []
        if pos == None: pos = 100

        if self.result.pose_landmarks:
            # print(self.result.pose_landmarks.landmark)       # .pose_landmarks.landmark contains each points x,y,z and visibility_confidence
            lmark = self.result.pose_landmarks.landmark

            for cnt,l_mark in enumerate(lmark):
                h, w, c = frame.shape
                cx, cy = int(l_mark.x * w), int(l_mark.y * h)       # Scaling
                self.lmark_list.append([cnt, cx, cy])
                if draw:
                    cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
                    # print(len(self.lmark_list))
                if draw and (len(self.lmark_list) > pos):
                    # cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.circle(frame, (self.lmark_list[pos][1], self.lmark_list[pos][2]), 10, (0, 0, 255), -1)
                #    cv2.putText(frame, "Found the position " + str(int(pos)), (190, 70),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # else:
                #     cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                #     cv2.putText(frame, "Couldn't find the position "+str(int(pos)), (190,120),
                #                 cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,0), 2)
        # return frame
def main():
    print(os.listdir("pose_videos"))
    s = os.path.join("pose_videos","gym_1.mp4")
    cap = cv2.VideoCapture(s)
    pTime = 0
    detector = PoseDetector()
    while cv2.waitKey(1) != 27:
        has_frame,frame = cap.read()
        if not has_frame: break
        img = detector.findPose(frame,resize=True)
        detector.findPosition(img,pos=15, draw=True)

        cTime = time.time()
        fps = int( 1/ (cTime - pTime))
        pTime = cTime
        cv2.putText(img,"FPS: "+str(int(fps)), (50,70), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,0,0), 2)
        cv2.imshow("Pose", img)

if __name__ == "__main__":
    main()

