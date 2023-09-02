import cv2
import os
import sys

import matplotlib.pyplot as plt

model_file = os.path.join("model_weights","ssd_mobilenet_v2_coco_2018_03_29","frozen_inference_graph.pb")
config_file = os.path.join("model_weights","ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
# Network
net = cv2.dnn.readNetFromTensorflow(model_file,config_file)

with open("coco_class_labels.txt") as fp:
    class_id = fp.read().split('\n')
#print(class_id)

cv2.namedWindow("Object Det", cv2.WINDOW_NORMAL)

def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img,1,(300,300),(0,0,0),False,False)
    net.setInput(blob=blob)
    object = net.forward()
    return object

def display_text(img,txt,t_w,t_h):
    size = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.7,1)
    dim, baseline = size[0], size[1]
    cv2.rectangle(img, (t_w, t_h-dim[1]-baseline), (t_w+dim[0], t_h+baseline), (0,0,0), cv2.FILLED)
    cv2.putText(img, txt, (t_w, t_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)

def display_objects(img,thres=0.6):
    obj = detect_objects(img)   # Calling detect_object function

    height, width = img.shape[0], img.shape[1]
    for i in range(obj.shape[2]):
        t_w = int(obj[0,0,i,3] * width)
        t_h = int(obj[0,0,i,4] * height)
        b_w = int(obj[0,0,i,5] * width-t_w)
        b_h = int(obj[0,0,i,6] * height-t_h)
        if obj[0,0,i,2] > thres:
            cv2.rectangle(img,(t_w,t_h),(b_w+t_w,b_h+t_h),(0,255,0),2)
            lab = class_id[int(obj[0, 0, i, 1])]
            txt = "{}".format(lab)
            display_text(img, txt, t_w, t_h)
    cv2.imshow("Object Det",img)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #plt.imshow(img); plt.show()
# Run
img = os.path.join("tf_images","street.jpg")
frame = cv2.imread(img)
s=0
if len(sys.argv) > 1:  s = sys.argv[1]
cap = cv2.VideoCapture(s)
#while cv2.waitKey(1) != 27:
while cv2.waitKey(1) < 0:
    sts, frame = cap.read()
    if not sts:
        cv2.waitKey()
        break
    frame = cv2.flip(frame,1)
    display_objects(frame,thres=0.60) # Calling all functions
#display_objects(frame,thres=0.25) # Calling all functions
cap.release()
cv2.destroyWindow("Object Det")

