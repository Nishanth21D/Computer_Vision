import sys
import cv2
import re
import matplotlib.pyplot as plt

# Load Video
video_file = "race_car.mp4"

def drawRectangle(frame,bbox):
    pt1 = (int(bbox[0]),int(bbox[1]))
    pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame,pt1,pt2,color=(255,0,0), thickness=3,lineType=1)

def displayRectangle(frame,bbox):
    plt.figure(figsize=(10,6))
    frame_copy = frame.copy()
    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    drawRectangle(frame_copy, bbox)
    plt.imshow(frame_copy); plt.axis("off")
    plt.show()

def writeText(frame, txt, location, color=(50,170,50)):
    cv2.putText(frame,txt,org=location,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=color,thickness=3)

# Tracker Types
tracker_types = ["BOOLEAN", "MIL", "KCF", "TLD", "MEDIANFLOW", "CSRT", "GOTURN", "MOOSE"]

# Default Tracker
tracker_types = tracker_types[4]

# Trackers Choosing
if tracker_types == "BOOSTING":
    tracker = cv2.legacy.TrackerBoosting.create()
elif tracker_types == "MIL":
    tracker = cv2.TrackerMIL.create()
elif tracker_types == "KCF":
    tracker = cv2.TrackerKCF.create()
elif tracker_types == "TLD":
    tracker = cv2.legacy.TrackerTLD.create()
elif tracker_types == "MEDIANFLOW":
    tracker = cv2.legacy.TrackerMedianFlow.create()
elif tracker_types == "CSRT":
    tracker = cv2.TrackerCSRT.create()
elif tracker_types == "GOTURN":
    tracker = cv2.TrackerGOTURN.create()
else:
    tracker = cv2.legacy.TrackerMOSSE.create()

# Read Video
vids = cv2.VideoCapture(video_file)

width = int(vids.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vids.get(cv2.CAP_PROP_FRAME_HEIGHT))
dim = (width,height)

# Output Format
out_format = video_file+str(tracker_types)+".mp4"
vids_output = cv2.VideoWriter(out_format, cv2.VideoWriter_fourcc(*"XVID"), 20, dim)

sts, frame = vids.read()
#frame_w, frame_h = frame.shape[:2]
if not sts:
    print("Read Error..!!")
    sys.exit()

win_name = "Playing Video"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name,720, 500)

bbox = [1300,400,170,120]
#bbox = cv2.selectROI(frame,False) #Manual setting of bbox on spot

# Initializing Tracker
ret = tracker.init(frame,bbox)

while True and cv2.waitKey(1) != 27:
    sts, frame = vids.read()
    if not sts:
        print("Error")
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)       # Update Tracker
    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))    # Calculating FPS
    if ret:
        drawRectangle(frame, bbox)
    else:
        writeText(frame,"Tracking Failure",(80,140),(0,0,255))

    #Display Tracker Info
    writeText(frame,txt="Tracker: "+str(tracker_types),location=(80,60))
    writeText(frame,"FPS: " + str(fps), (80,100))
    cv2.imshow(win_name,frame)
    vids_output.write(frame)
    #k = cv2.waitKey(1) & 0xff
    #if k == 27: break

print("Loop Done")
print("Tracker: ",out_format)
vids.release()
vids_output.release()
# playfile(out_format)
cv2.destroyWindow(win_name)
