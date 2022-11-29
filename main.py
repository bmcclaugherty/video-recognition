import cv2
import time
#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(300,300)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,
                            classNames[classId-1].lower() + str(round(confidence*100)),
                            (box[0]+box[2],box[1]+box[3]),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    classNames[classId-1].lower()
    return img,objectInfo

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("/home/pi/Desktop/Object_Detection_Files/video_low_res.mp4")
    cap.set(3,640)
    cap.set(4,480)
    cap.set( cv2.CAP_PROP_FPS, 15 )

    #cap.set(10,70)

    count = 0
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.6,0.2)#,objects=["car"])
        #print(objectInfo)

        cv2.imshow("Output",img)

        for o in objectInfo:
            print(o[1])
            if o[1] == 'car':
                count+=45
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            else:
                cap.release()
        cv2.waitKey(1)
