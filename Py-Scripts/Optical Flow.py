""" Optical Flow (Sparse & Dense) """
import cv2
import numpy as np

# Sparse Optical Flow _ Lucas-Kanade algorithm
def lucas_kanade(src):
    cap = cv2.VideoCapture(src)
    # Parameter _ ShiThomas Corner Detection
    shi_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameter _ Pyramid Lucas Kanade
    lk_params = dict(winSize=(20,20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0,255,(100,3))
    _, old_frame = cap.read()
    old_frame = cv2.resize(old_frame,(480,360))
    old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(old_gray,mask=None, **shi_params)
    # print(prev)
    mask = np.zeros_like(old_frame)
    while cv2.waitKey(1) != 27:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(480,360))
        frame1 = frame.copy()
        curr_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Calculate Optical FLow
        next, sts, err = cv2.calcOpticalFlowPyrLK(old_gray, curr_gray, prev, None, **lk_params)

        # Selecting good points from prev and curr frame
        pts_new = next[sts == 1]
        pts_old = prev[sts == 1]

        for i , (new,old) in enumerate(zip(pts_new,pts_old)):
            x, y = new.ravel()
            w, h = old.ravel()
            print(int(x),int(y))
            mask = cv2.line(mask, (int(x),int(y)), (int(w),int(h)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(x),int(y)),5, color[i].tolist(), -1)
        img = cv2.add(frame,mask)
        stack_frame = np.hstack([frame1,mask,img])
        cv2.imshow('Frame', stack_frame)

        if cv2.waitKey(1) == ord('c'):
            mask = np.zeros_like(old_frame)
        # Updating prev frame & points
        old_gray = curr_gray.copy()
        prev = pts_new.reshape(-1,1,2)

    cap.release()
    cv2.destroyAllWindows()

def dense_opticalflow(opts=1, src=0, gray=False):
    cap = cv2.VideoCapture(src)
    params = []
    _, old_frame = cap.read()
    old_frame = cv2.resize(old_frame,(480,360))

    # Creating an HSV & make a constant value (255)
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    if gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    if opts == 1:
        # Lucas Kanade Dense Optical Flow
        algo = cv2.optflow.calcOpticalFlowSparseToDense
    elif opts == 2:
        algo = cv2.optflow.calcOpticalFlowDenseRLOF
    elif opts == 3:
        # Farneback
        params = [0.5, 3, 15, 3, 5, 1.2, 0]  # Farneback's algorithm parameters
        algo = cv2.calcOpticalFlowFarneback


    while cv2.waitKey(1) != 27:
        has_frame, new_frame = cap.read()
        new_frame = cv2.resize(new_frame,(480,360))
        frame_cpy = new_frame.copy()
        if gray:
            new_frame = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)

        # Calculating dense optical flow
        optflow = algo(old_frame,new_frame, None, *params)

        # Converting the algo's output to polar coordinates
        magnitude, angle = cv2.cartToPolar(x=optflow[..., 0], y=optflow[..., 1])

        # Use Hue & Saturation to encode the optical flow
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(src=magnitude, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Converting HSV to BGR
        conv_frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        stacked = np.hstack([frame_cpy,conv_frame])
        cv2.imshow("Dense Flow", stacked)
        old_frame = new_frame

## Calling Main Function
# lucas_kanade("pose_videos/gym_1.mp4")

## Calling Various Dense OpticalFlow
# dense_opticalflow(opts=1,src="pose_videos/gym_1.mp4",gray=True)
# dense_opticalflow(opts=2,src="pose_videos/gym_1.mp4")
dense_opticalflow(opts=3,src="pose_videos/dance_2.mp4",gray=True)