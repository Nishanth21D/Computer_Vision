""" Mediapipe FaceMesh 478 Landmark for Video """
import time
import cv2
import numpy as np
import torch
import mediapipe as mp


device = "CUDA" if torch.cuda.is_available() else "CPU"
print("Device: ", device)

# FaceMesh Configuration
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,
                                  min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(720,480))
    orig_frame = frame.copy()
    start = time.perf_counter()

    # To improve performance, making the image as not writable
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame)  # Applying Facemesh

    # Making the image as writable now
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        cnt_pnt_idx = []
        for cnt in mp_face_mesh.FACEMESH_CONTOURS:
            cnt_pnt_idx.append(cnt)
        for lmark in results.multi_face_landmarks:
            for idx, l_mark in enumerate(lmark.landmark):
                h, w = frame.shape[:2]
                cx, cy = int(l_mark.x * w), int(l_mark.y * h)
                cv2.circle(frame,(cx,cy),1,(0,0,255),-1)
                cv2.putText(frame,str(idx),(cx,cy),1,0.5,(255,0,0),1)
                if idx == 151:
                    cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

    end = time.perf_counter()
    fps = int(1 / (end - start))
    cv2.putText(frame, "FPS: " + str(fps), (20, 30), 3, 0.4, (0, 0, 0))
    cv2.putText(frame, "Device: " + str(device), (20, 45), 3, 0.4, (255, 0, 0))
    stack = np.hstack([orig_frame,frame])
    cv2.imshow('FaceMesh',stack)
    # cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()