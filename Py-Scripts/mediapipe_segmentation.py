import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmen = mp.solutions.selfie_segmentation

selfie_segmen = mp_selfie_segmen.SelfieSegmentation(model_selection=0)

bg_color = (212, 255, 127)
bg_img = cv2.imread('bg/bg3.jpg')
cap = cv2.VideoCapture(0)
while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame: break

    frame = cv2.cvtColor((cv2.flip(frame,1)), cv2.COLOR_BGR2RGB)

    frame.flags.writeable = False
    results = selfie_segmen.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    condition = np.stack((results.segmentation_mask, )*3, axis= -1) > 0.6

    # bg_image = np.zeros(frame.shape, dtype=np.uint8)
    # bg_image[:] = bg_color
    bg_img = cv2.resize(bg_img, frame.shape[1::-1])
    output_frame = np.where(condition,frame, bg_img)
    cv2.imshow('selfie_frame', output_frame)

cap.release()
cv2.destroyAllWindows()
