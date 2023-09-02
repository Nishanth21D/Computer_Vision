import cv2
import sys
"""
URL:https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
Github Repository URL:https://github.com/opencv/opencv/tree/4.x/samples/dnn
Model Cofigs URL: https://github.com/opencv/opencv/blob/4.x/samples/dnn/models.yml
"""
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

#s = "my Graduation video_alone 480p.mp4"
source = cv2.VideoCapture(s)
win_name="Camera_Preview"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

# Model
net = cv2.dnn.readNetFromCaffe(prototxt="deploy.prototxt",caffeModel="res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model Parameters
in_width = 300
in_height = 300
in_mean = [104, 177, 123]
conf_thres = 0.7
scale = 1.0

while cv2.waitKey(1) != 27:
    has_frame,frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame,1)
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame,scale,(in_width,in_height),in_mean,swapRB=False,crop=False)
    # Run a Model
    net.setInput(blob)
    detections = net.forward()
    """
    detections[0,0,i,j], i-> iterates over the detected face, j-> contains info about bbox and conf_score
    detections[0,0,i,2] -> conf_score , detections[0,0,i,3:6] -> bbox(b_width,b_height,t_width,t_height)
    """
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_thres:
            b_width = int(detections[0,0,i,3] * frame_w)          # Bottom Right Width
            b_height = int(detections[0, 0, i, 4] * frame_h)      # Bottom Right Height
            t_width = int(detections[0, 0, i, 5] * frame_w)       # Top Left Width
            t_height = int(detections[0, 0, i, 6] * frame_h)      # Top Left Height

            cv2.rectangle(frame,(b_width,b_height),(t_width,t_height),(0,255,0),2)
            label = "Label: "+str(round(confidence,4))
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            cv2.rectangle(frame,(b_width, b_height - label_size[1]),(b_width + label_size[0], b_height + base_line),
                (255, 255, 255),cv2.FILLED)
            cv2.putText(frame,label,(b_width,b_height),cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,255,0),2)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.imshow(win_name,frame)

source.release()
cv2.destroyWindow(win_name)
