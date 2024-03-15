""" Custom Snapchat Filter"""
import csv

import math

""" To-DO
1. Blur background & Image Background- Done
2. swiping & pressing on the screen
3. filters annotating
4. main code implementation
5. method to save and store the current frame in local
6. gradio implementation
"""
import cv2
import numpy as np
import time
import torch
import os
import mediapipe as mp
from Custom_HandTrackingModule import HandTracker
import faceBlendCommon as fbc


device = "CUDA" if torch.cuda.is_available() else "CPU"
print("Device: ", device)

# FaceMesh Configuration
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,
                                  min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

## Info of available filters
filter_config = {
    'anonymous':
        [{'path': "filters/GrassVillage_Anonymous_Halloween.png",
          'anno_path': "filters/GrassVillage_Anonymous_Halloween_Annotation.csv",
          'morph': True}],
    'joker':
        [{'path': "filters/joker.png",
          'anno_path': "filters/joker_annotation.csv",
          'morph': True}],
    'carnival':
        [{'path': "filters/carnival.png",
          'anno_path': "filters/carnival_annotation.csv",
          'morph': True}]
}
# os.listdir("Github Repos/Create-AR-filters-using-Mediapipe/filters/anonymous_annotations.csv")
# filter_config = {
#     'anonymous':
#         [{'path': "Github Repos/Create-AR-filters-using-Mediapipe/filters/anonymous.png",
#           'anno_path': "Github Repos/Create-AR-filters-using-Mediapipe/filters/anonymous_annotations.csv",
#           'morph': True}]
# }
# Selected Keypoints from 478 landmarks points
# selected_keypoints = [10, 109, 338, 103, 332, 54, 284, 21, 251, 127, 356, 93, 323, 58, 288, 172, 397, 150, 379, 176, 400,
#                       148, 377, 152, 199, 18, 164, 1, 4, 5, 195, 6, 8, 9, 151, 69, 299, 68, 298, 143, 372, 123, 352,
#                       213, 456, 135, 364, 170, 395, 175, 107, 336, 105, 334, 70, 390, 46, 52, 55, 285, 282, 276, 33, 133,
#                       159, 145, 362, 263, 386, 374, 130, 243, 463, 359, 253, 257, 23, 27, 468, 473, 119, 347, 50, 280,
#                       187, 427, 214, 434, 210, 430, 32, 262, 101, 330, 49, 279, 64, 294, 220, 440, 44, 274, 236, 456,
#                       122, 465, 218, 438, 134, 363, 203, 428, 60, 290, 97, 326, 165, 391,186, 410, 57, 287, 216, 436,
#                       212, 432, 207, 427, 0, 14, 39, 269, 185, 409, 61, 291,91, 321, 181, 405, 40, 270, 78, 308, 81, 178,
#                       13, 14, 311, 402, 11, 16, 73, 180, 303, 404, 194, 418, 182, 406, 454, 234]

selected_keypoints = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]
print(len(selected_keypoints))


# Constrain point to be inside the boundary
def constrainpoint(p, w, h):
    p = (min(max(p[0], 0), w-1), min(max(p[1],0), h-1))
    return p

# Function for Loading Filter Image
def load_filter_img(img):
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    b, g, r, alpha = cv2.split(img)
    img = cv2.merge((b, g, r))
    return img, alpha

# Function to load the filter's landmark
def load_filter_landmarks(anno_file):
    with open(anno_file) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        points = {}
        for _, row in enumerate(reader):
            x, y = int(row[1]), int(row[2])
            points[row[0]] = (x, y)
        return points

def find_convex_hull(points):
    hull = []
    hull_indx = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    hull_pts = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]]  # Eyebrows
    hull_indx = np.concatenate((hull_indx, hull_pts))
    for i in range(0, len(hull_indx)):
        hull.append(points[str(hull_indx[i][0])])

    return hull, hull_indx


def get_landmarks(frame):
    results = face_mesh.process(frame)  # Applying Facemesh
    if not results.multi_face_landmarks:
        print("Face Not Detected")
        return 0,0,0
    if results.multi_face_landmarks:
        for lmark in results.multi_face_landmarks:
            xlist, ylist = [], []                       # For BBOX
            face_keypoints = []                         # Contains all face points
            for idx, l_mark in enumerate(lmark.landmark):
                h, w = frame.shape[:2]
                px, py = int(l_mark.x * w), int(l_mark.y * h)
                xlist.append(px)
                ylist.append(py)
                face_keypoints.append((px,py))
                if idx in selected_keypoints:
                    cv2.circle(frame,(px,py),1,(0,0,255),-1)    # Landmark points
                # cv2.putText(frame,str(idx),(cx,cy),1,0.5,(255,0,0),1)        # Landmark labels
                # if idx == 151:
                #     cv2.circle(frame, (px, py), 4, (255, 255, 0), -1)

            relevant_keypoints = []
            for i in selected_keypoints:
                relevant_keypoints.append(face_keypoints[i])  # Extract the coordinates of selected keypoints
            return relevant_keypoints, xlist, ylist
    return 0,0,0


def load_filter(filter_name='anonymous'):
    filters = filter_config[filter_name]
    filter_runtime = []
    for filt in filters:
        temp_dict = {}
        img, img_alpha = load_filter_img(filt['path'])
        points = load_filter_landmarks(filt['anno_path'])

        temp_dict['img'] = img
        temp_dict['img_alpha'] = img_alpha
        temp_dict['points'] = points

        if filt['morph']:
            # Find Convex Hull for Delaunay Triangulation
            hull, hull_indx = find_convex_hull(points)

            # Find Delaunay Triangulation
            img_shape = img.shape
            rect = (0,0, img_shape[1], img_shape[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hull_indx'] = hull_indx
            temp_dict['dt'] = dt
            if len(dt) == 0:
                continue

        filter_runtime.append(temp_dict)
    return filters, filter_runtime


## Pyramid Lucas Kanade _ OpticalFlow Parameters
lk_params = dict(winSize=(101,101), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

# Flags
br_indx = False     # Blur Index
frst_frame = True
flt_indx = False
sigma = 50

filter_keys = [key for key in filter_config.keys()]
key_length = len(filter_keys)
index = 0

filters, multi_filter_runtime = load_filter(filter_keys[index])
# filters, multi_filter_runtime = load_filter('anonymous')
cap = cv2.VideoCapture(0)

# Main Loop
while cv2.waitKey(1) != 27:
    # print(indx)
    has_frame, frame = cap.read()
    if not has_frame:
        break
    start = time.perf_counter()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(720,480))
    orig_frame = frame.copy()
    mask = np.ones_like(orig_frame, dtype=np.uint8) * 255  # White mask (clear face regions)
    if cv2.waitKey(1) == ord('b'): br_indx = True

    # To improve performance, making the image as not writable
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


    # Making the image as writable now
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get Facial Landmarks from Mediapipe Facemesh
    relevant_keypoints, xlist, ylist = get_landmarks(frame)

    if not relevant_keypoints or (len(relevant_keypoints) < (len(relevant_keypoints)-30)):
        continue
    ###### ----- Optical flow & Stabilization ------- #######
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if frst_frame:
        prev_points = np.array(relevant_keypoints, np.float32)
        # print(prev_points)
        frame_gray_prev = np.copy(frame_gray)
        frst_frame = False

    # Calculating Optical Flow
    next_points, err, sts = cv2.calcOpticalFlowPyrLK(frame_gray_prev,frame_gray, prev_points,
                                                     np.array(relevant_keypoints, np.float32), **lk_params)
    for k in range(0, len(relevant_keypoints)):
        d = cv2.norm(np.array(relevant_keypoints[k]- next_points[k]))
        alpha = math.exp(-d * d / sigma)
        relevant_keypoints[k] = (1 - alpha) * np.array(relevant_keypoints[k]) + alpha * next_points[k]
        relevant_keypoints[k] = constrainpoint(relevant_keypoints[k], frame.shape[1], frame.shape[0])
        relevant_keypoints[k] = (int(relevant_keypoints[k][0]), int(relevant_keypoints[k][1]))

    # update variable for next pass
    prev_points = np.array(relevant_keypoints, np.float32)
    frame_gray_prev = frame_gray
    ###### ----- End of Optical flow & Stabilization ------- #######

    if flt_indx:
        # print("Inside Filter Loop")
        for indx, filt in enumerate(filters):
            filter_runtime = multi_filter_runtime[indx]
            img = filter_runtime['img']
            img_alpha = filter_runtime['img_alpha']
            points = filter_runtime['points']

            if filt['morph']:
                # print("Inside Morph Loop")
                hull1 = filter_runtime['hull']
                hull_indx = filter_runtime['hull_indx']
                dt = filter_runtime['dt']

                warped_img = np.copy(frame)
                hull2 = []
                for i in range(0, len(hull_indx)):
                    hull2.append(relevant_keypoints[hull_indx[i][0]])
                mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                mask1 = cv2.merge((mask1,mask1,mask1))
                img_alpha_mask = cv2.merge((img_alpha,img_alpha,img_alpha))

                # Warp the Triangle
                for i in range(0, len(dt)):
                    t1, t2 = [], []
                    for j in range(0,3):
                        t1.append(hull1[dt[i][j]])
                        t2.append(hull2[dt[i][j]])
                    fbc.warpTriangle(img, warped_img, t1, t2)
                    fbc.warpTriangle(img_alpha_mask, mask1, t1, t2)

                # Blur the mask before blending
                mask1 = cv2.GaussianBlur(mask1,(3,3), 10)
                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images
                temp1 = np.multiply(warped_img, (mask1 * (1.0/255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2
                cv2.imshow('output', temp1)
                frame = output = np.uint8(output)
            else:
                pass
    # Face BBOX
    xmin, xmax = min(xlist), max(xlist)
    ymin, ymax = min(ylist), max(ylist)
    boxW, boxH = xmax - xmin, ymax - ymin
    bbox = xmin, ymin, boxW, boxH
    cv2.rectangle(frame, pt1=(bbox[0] -50, bbox[1]-90), pt2=(bbox[0] + bbox[2] + 60, bbox[1] + bbox[3] + 60),
                  color=(0, 255, 255), thickness=2)

    ## Blur Option
    if br_indx:
        face = frame[bbox[1]-90:bbox[1] + bbox[3] + 60,bbox[0] -50:bbox[0] + bbox[2] + 60]  # Extracting ROI (y:w+h,x:x+w)
        blur = cv2.GaussianBlur(frame,(99,99),20)
        blur[bbox[1]-90:bbox[1] + bbox[3] + 60, bbox[0] -50:bbox[0] + bbox[2] + 60] = face
        frame = blur
        # frame = cv2.bitwise_and(blur, mask) + cv2.bitwise_and(frame, cv2.bitwise_not(mask)) # Merge ROI into the Blurred background
    if cv2.waitKey(1) == ord('f'):
        flt_indx = True

    if cv2.waitKey(1) == ord('n'):
        index = (index + 1) % key_length
        filters, multi_filter_runtime = load_filter(filter_keys[index])

    if cv2.waitKey(1) == ord('p'):
        index = (index - 1) % key_length
        filters, multi_filter_runtime = load_filter(filter_keys[index])

    if cv2.waitKey(1) == ord('r'):
        br_indx = False
        flt_indx = False

    end = time.perf_counter()
    fps = int(1 / (end - start))
    cv2.putText(frame, "FPS: " + str(fps), (20, 30), 3, 0.4, (0, 0, 0))
    cv2.putText(frame, "Device: " + str(device), (20, 45), 3, 0.4, (255, 0, 0))
    stack = np.hstack([orig_frame,frame])
    cv2.imshow('FaceMesh',stack)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
