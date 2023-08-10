import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyttsx3
import threading


engine = pyttsx3.init()

def speak(text):
    try:
        voices = engine.getProperty('voices')
        engine.setProperty('voice', 'voices[0].id')
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        xy = 'true'


t = threading.Thread(target=speak,args=("Hello",))
t.start()
def process(text1):
    t = threading.Thread(target=speak,args=(text1,))
    t.start()

def ObjectDetection():


    def calculateIntersection(a0, a1, b0, b1):
        checkFREEBOX = False
        if a0 >= b0 and a1 <= b1:  # Contained
            intersection = a1 - a0
            checkFREEBOX = True
        elif a0 < b0 and a1 > b1:  # Contains
            intersection = b1 - b0
            checkFREEBOX = True
        elif a0 < b0 and a1 > b0:  # Intersects right
            intersection = a1 - b0
            checkFREEBOX = True
        elif a1 > b1 and a0 < b1:  # Intersects left
            intersection = b1 - a0
            checkFREEBOX = True
        else:  # No intersection (either side)
            intersection = 0
            checkFREEBOX = False

        return checkFREEBOX

    thres = 0.45  # Threshold to detect object
    nms_threshold = 0.2
    cap = cv2.VideoCapture('./video1.mp4')
    # cap = cv2.VideoCapture(1)
    # Capture = cv2.imread('./TestImage.jpg')
    # cap.set(3,1280)
    # cap.set(4,720)
    # cap.set(10,150)
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # print(classNames)
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #SSD Algorithm Single Shot Detector
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    abs_location = -1
    mover = 0
    start = True
    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        # print(type(confs[0]))
        # print(confs)

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        # print(indices)
        maxWidth = img.shape[1]
        j=0
        rectangles = []
        intersecting = []
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=0)
            X0, Y0, X1, Y1, = [x, y, (x + w), (h + y)]
            # AREA = float((X1 - X0) * (Y1 - Y0))
            while j < maxWidth:
                cv2.rectangle(img, (j, 550), (j+100, 400), color=(255, 255, 255), thickness=-1)
                rectangles.append([j, 650, j+100, 400])
                j += 110

            i = 0
            Z = []
            count = 0
            for x0, y0, x1, y1 in rectangles:
                width = calculateIntersection(x0, x1, X0, X1)
                height = calculateIntersection(y0, y1, Y0, Y1)

                if width and height:
                    cv2.rectangle(img, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=-1)
                    Z.append(-1)
                    if(count > 2):
                        process('Stop')
                    count += 1
                else:
                    count = 0
                    Z.append(0)
            # print (Z)
            if(start == True):
                abs_location = int(len(Z)/2) # To Start the Algorithm
                start = False
            # if(Z[int(len(Z)/2)] == -5):
            #     abs_location = int(len(Z)/2)
            else:
                mover = 0
                # print(mid)
                if (Z[abs_location] != 0):
                    l = abs_location - 1
                    countL = 0
                    while l >= 0:
                        if (Z[l] == 0):
                            countL += 1
                        l = l - 1
                    l = abs_location + 1
                    countR = 0
                    while l < len(Z):
                        if (Z[l] == 0):
                            countR += 1
                        l = l + 1
                    if (countL >= countR):
                        l = abs_location - 1
                        countL = 0
                        while l >= 0:
                            mover -= 1
                            if (Z[l] == 0):
                                break
                            l = l - 1
                    else:
                        l = abs_location + 1
                        countR = 0
                        while l < len(Z):
                            mover += 1
                            if (Z[l] == 0):
                                break
                            l = l + 1
                abs_location = abs_location + mover
                if (mover > 0 and mover <= 1):
                    process("Right")
                elif (mover < 0 and mover >= -1):
                    process("Left")
                elif (mover > 1):
                    process("Sharp Right")
                elif mover < -1:
                    process("Sharp Left")

            x0, y0, x1, y1 = rectangles[abs_location]
            cv2.rectangle(img, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=-1)






        cv2.imshow("Output", img)

        cv2.waitKey(1)



# ObjectDetection()

x=threading.Thread(target=ObjectDetection())
x.start()
