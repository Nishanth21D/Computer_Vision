""" Background Subtraction using MOG2 and KNN"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,360)
cap.set(4,240)

bgsub_MOG2 = cv2.createBackgroundSubtractorMOG2()    # Background Subtraction using MOG2

bgsub_KNN = cv2.createBackgroundSubtractorKNN()      # Background Subtraction using KNN

while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    if not has_frame: break

    orig_frame = frame

    # Applying the background subtractor to the frame
    mog2_frame = bgsub_MOG2.apply(frame)
    # Applying the background subtractor to the frame
    knn_frame = bgsub_KNN.apply(frame)

    # Converting the mask to 3 channels (Not mandatory to mask it, just doing it for stacking purpose)
    mog2_frame = cv2.cvtColor(mog2_frame, cv2.COLOR_GRAY2BGR)
    knn_frame = cv2.cvtColor(knn_frame, cv2.COLOR_GRAY2BGR)

    # Stacking all the Frame (Mog2, Knn, Original)
    stack_frame = np.hstack((mog2_frame, knn_frame,orig_frame))
    cv2.imshow("MOG2 & KNN & Original", stack_frame)

cap.release()
cv2.destroyAllWindows()