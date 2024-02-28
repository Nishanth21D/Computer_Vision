""" Mediapipe FaceMesh 478 Landmark for Image """
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

img = cv2.imread("sg_demo/nish8.jpg",1)
img = cv2.resize(img,(360,360))

img.flags.writeable = False # To improve performance
result = face_mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

tesselation_img = img.copy()
contour_img = img.copy()
irises_img = img.copy()
cnt_list = []
lmark_list = []
for lmark in result.multi_face_landmarks:
    print(lmark)
    mp_drawing.draw_landmarks(image=tesselation_img,landmark_list=lmark,connections=mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_style.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(image=contour_img,landmark_list=lmark,connections=mp_face_mesh.FACEMESH_CONTOURS,
                              landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_style.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image=irises_img,landmark_list=lmark,connections=mp_face_mesh.FACEMESH_IRISES,
                              landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_style.get_default_face_mesh_iris_connections_style())

    # Landmarks Points
    for idx, l_mark in enumerate(lmark.landmark):
        h, w = img.shape[:2]
        point = (int(l_mark.x * w), int(l_mark.y * h))
        cv2.circle(img, point, 1, (255, 0, 0), -1)
        cv2.putText(img, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
        lmark_list.append(point)

    # Contours Points
    for pnt in mp_face_mesh.FACEMESH_CONTOURS:
        cnt_list.append(pnt[0])
print(cnt_list)
print(lmark_list)
print(len(lmark_list))

stack_ig = np.hstack([img,tesselation_img,contour_img,irises_img])
cv2.imshow('Orig_FACEMESH[Tesselation_Contour_Irises]',stack_ig)
cv2.imshow('FaceMesh',img)
cv2.waitKey(0)

cv2.destroyAllWindows()