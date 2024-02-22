""" Face Blurring using Haar Cascade & Gaussian Blur"""

import cv2
import time
import torch

device = "CUDA" if torch.cuda.is_available() else "CPU"
print("Device: ", device)

# Loading the pre-trained face detection Haar Cascade classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)

while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    img = cv2.flip(frame,1)
    start = time.perf_counter()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       # Converting to gray_scale

    # Detecting faces on the grayscale image
    faces = cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(5,5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x-40,y-40),(x+w+40, y+h+40), (255,255,0),2)
        face = frame[y:y+h, x:x+w]                                          # Extract ROI
        bg_blur = cv2.GaussianBlur(face,ksize=(99,99),sigmaX=40)            # Apply blur to ROI
        frame[y:y + h, x:x + w] = bg_blur                                   # Merging the blurred face to frame

    end = time.perf_counter()
    fps = int(1 / (end - start))
    cv2.putText(frame, "FPS: " + str(fps), (20, 30), 2, 0.5, (0, 0, 0))
    cv2.putText(frame, "Device: " + str(device), (20,60), 2, 0.5, (255,0,0))
    cv2.imshow("Blur", frame+img)
    cv2.imshow('Orig', img)
    # cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()