"""Custom Tracker for YOLO Models"""
import os
import cv2

# YOLO V4 Model
y4_config = os.path.join("model_weights","yolo_models","yolov4.cfg")
y4_weight = os.path.join("model_weights","weights","yolov4.weights")
# cls_path="coco.names"

class yolo_detector():
    print("___ DNN Model: YOLO V4 __")
    def __init__(self,conf=0.6,thres=0.4,image_size = 608):
        self.conf = conf
        self.thres = thres
        self.image_size = image_size
        # self.load_class()
        self.y4_config = os.path.join("model_weights", "yolo_models", "yolov4.cfg")
        self.y4_weight = os.path.join("model_weights", "weights", "yolov4.weights")
        net = cv2.dnn.readNetFromDarknet(cfgFile=self.y4_config, darknetModel=self.y4_weight)
        # net = cv2.dnn.readNet(model=self.y4_weight, config=self.y4_config)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn.DetectionModel(net)
        self.model.setInputParams(scale=1 / 255, size=(self.image_size, self.image_size), swapRB=True)

    def yolo_v4(self,frame):
        # print("___ DNN Model: YOLO V4 __")
        return self.model.detect(frame, confThreshold=self.conf, nmsThreshold=self.thres)
    def load_class(self,cls_path="coco.names"):
        with open(cls_path, 'r') as f:
            classes = f.read().splitlines()
        print(classes)
        return classes

def main():
    print("Main Class")
    src = os.path.join("soccer.jpg")
    detector = yolo_tracker(0.4, 0.2)
    classes = detector.load_class(cls_path="coco.names")
    cap = cv2.VideoCapture(0)
    while cv2.waitKey(1) != 27:
        has_frame, frame = cap.read()
        if not has_frame:
            cv2.waitKey(0)
            break
        clasid, score, bbox = detector.yolo_v4(frame)
        # print(clasid, score, bbox)
        for (cls, scr, box) in zip(clasid, score, bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=1)
            text = classes[cls] + ":" + str(round(scr, 2))
            cv2.putText(frame, text, (box[0], box[1] - 7), 1, 1.3, (255, 255, 0), 1)

        cv2.imshow("image", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


