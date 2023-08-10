import speech_recognition as sr
import pyttsx3
import datetime
import cv2
import easyocr
import numpy as np
import time
import requests

#**********************************************   GLOBAL CODE   ***********************************************


#code for speech recognition engine setup
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', 'voices[0].id')
engine.setProperty('rate', 145)


#text recognition initialization
reader = easyocr.Reader(['en'])


#object detection global code
thres = 0.45  # Threshold to detect object
nms_threshold = 0.2
#classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


#code for weather info
BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
API_KEY = "7689490016fe71f16db586b3cf825313"
name = 0
city = 0


#**************************************************   FUNCTIONS   *****************************************************


# to convert to speech !!! no change in this !!!
def speak(text):
    engine.say(text)
    engine.runAndWait()



# to wish
def wishMe(namecall):
    name1 = namecall
    hour = datetime.datetime.now().hour
    if hour >= 0 and hour < 12 :
        speak(f"Hello,Good Morning {name1}")
    elif hour>=12 and hour<18:
        speak(f"Hello,Good Afternoon {name1}")
    else:
        speak(f"Hello,Good Evening {name1}")


# weather info
def weather(city):
    URL = BASE_URL + "q=" + city + "&appid=" + API_KEY
    # HTTP request
    response = requests.get(URL)
    # checking the status code of the request
    if response.status_code == 200:
        # getting data in the json format
        data = response.json()
        # getting the main dict block
        main = data['main']
        # getting temperature
        temperature = main['temp']
        # getting the humidity
        humidity = main['humidity']
        humid=int(humidity)
        # weather report
        report = data['weather']
        #print(f"{CITY:-^30}")
        # print(f"Temperature: {temperature}")
        # print(f"Humidity: {humidity}")
        # print(f"Weather Report: {report[0]['description']}")
        celsius = int(temperature-273.15)
        report = f"Temperature in your city is {celsius} degree celcius and humidity is {humid} so in short" \
                 f" today it is {report[0]['description']} climate"
        return report
    else:
        # showing the error message
        return "Error in the HTTP request"



# def takeCommand():
#     r=sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         speak("listening")
#         audio=r.listen(source)
#
#         try:
#             statement=r.recognize_google(audio,language='en-in')
#             print(f"user said:{statement}\n")
#
#         except Exception as e:
#             speak("Pardon me, please say that again")
#             return "None"
#         return statement


# text recognition based on video:
def text_reco():
    cap = cv2.VideoCapture(0)  # video start
    _, img = cap.read()  # video read
    result = reader.readtext(img, detail=0, paragraph=True)

    print(result)
    if result != '':
        cap.release()  # video close
        cv2.destroyAllWindows()
        return result
    else:
        speak('no text found')
        print('no text found')


#object detection based on video
def object_detect():
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    s = set()
    #s=[]
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    # print(type(confs[0]))
    # print(confs)
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    # print(indices)
    for i in indices:
        i = i[0]
        s.add(classNames[classIds[i][0] - 1])
        #s.append(classNames[classIds[i][0] - 1])
    #cv2.imshow("Output", img)
    print('objects in front of you are',s)
    speak('objects in front of you are')
    speak(s)
    time.sleep(3)
    if s!='':
        cap.release()  # video close
        cv2.destroyAllWindows()


# Navigation function
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
    start = True

    t_end = time.time() + 60 * 1

    while True:
        if time.time()>t_end:
            break
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
                rectangles.append([j, 550, j+100, 400])
                j += 110

            i = 0
            Z = []
            for x0, y0, x1, y1 in rectangles:
                width = calculateIntersection(x0, x1, X0, X1)
                height = calculateIntersection(y0, y1, Y0, Y1)

                if width and height:
                    cv2.rectangle(img, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=-1)
                    Z.append(-1)
                else:
                    Z.append(0)
            # print (Z)
            if(start == True):
                abs_location = int(len(Z)/2) # To Start the Algorithm
                start = False

            mover = 0
            # print(mid)
            if(Z[abs_location] != 0):
                l = abs_location-1
                countL = 0
                while l>=0:
                    if(Z[l] == 0):
                        countL += 1
                    l = l - 1
                l = abs_location + 1
                countR = 0
                while l<len(Z):
                    if(Z[l] == 0):
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
            # print(abs_location)
            x0, y0, x1, y1 = rectangles[abs_location]
            cv2.rectangle(img, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=-1)
            if(mover > 0 and mover <=1):
                speak("Go Slight Right")
                print("Go Slight Right")
                time.sleep(3)
            elif (mover < 0 and mover >= -1):
                speak("Go Slight Left")
                print("Go Slight Left")
                time.sleep(3)
            elif (mover >1):
                speak("Go Heavy Right")
                print("Go Heavy Right")
                time.sleep(3)
            elif mover < -1:

                speak("Go Heavy Left")
                print("Go Heavy Left")
                time.sleep(3)
            # cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Output", img)
        #time.sleep(4)

        cv2.waitKey(1)




#-----------------------------------------------    MAIN FUNCTION    --------------------------------------------------

if __name__=='__main__':
    speak("initiating Speech mode")

    #speech code to take input from microphone
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.energy_threshold=800
            #r.dynamic_energy_threshold=False
            r.adjust_for_ambient_noise(source, duration=0.8)
            r.dynamic_energy_threshold = True
            r.pause_threshold = 0.8
            statement = 0


            try:
                if name == 0:
                    speak("Please tell me your name")
                    print("Listening...")
                    speak("listening")
                    name_listen = r.listen(source)
                    name = r.recognize_google(name_listen, language='en-in')  # statement contains what we said !! USE THAT !!
                    print(f"name is:{name}\n")

                if city == 0:
                    speak("Please tell me your city")
                    print("Listening...")
                    speak("listening")
                    city_listen = r.listen(source)
                    city = r.recognize_google(city_listen, language='en-in')  # statement contains what we said !! USE THAT !!
                    print(f"city:{city}\n")

                speak("command me")
                print("Listening...")
                speak("listening")
                audio = r.listen(source)
                statement = r.recognize_google(audio, language='en-in')   # statement contains what we said !! USE THAT !!
                print(f"user said:{statement}\n")

            except Exception as e:
                speak("please say that again")

        # no audio or command
        if statement == 0:
            continue


        # to stop or shut down system
        elif "goodbye" in statement or "ok bye" in statement or "stop" in statement:
            speak(f'your personal assistant is shutting down,Good bye{name}')
            break


        # general code to wish hello
        elif 'hello' in statement:
            wishMe(name)
            statement=0
            continue

        elif 'wait' in statement:
            speak(f"ok {name} I am waiting for 3 seconds")
            time.sleep(3)
            continue

        # to know time
        elif 'time' in statement:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"{name} the time is {strTime}")
            statement = 0
            continue


        # general code
        elif 'how are you' in statement:
            speak(f"I am cool what about you{name}")
            statement = 0
            continue



        # general code
        elif 'fine' in statement:
            speak('ok how can i help you sir')
            statement = 0
            continue


        # general code
        elif 'navigate me' in statement:
            speak("I will assist you for 30 seconds after that ask me again I will help you")
            ObjectDetection()
            continue


        # weather info
        elif 'climate' in statement:
            speak("checking climate")
            weather_result = weather(city)
            speak(weather_result)
            statement = 0
            continue


        # to read code
        elif 'text mode' in statement:
            speak('reading please wait')
            text_res=text_reco()
            speak(text_res)
            statement = 0
            continue


        # to know objects in vision

        elif 'around' or 'surroundings objects' or 'objects near me' or 'objects' or 'objects again' in statement:
            speak('detecting objects please wait')
            object_detect()
            statement = 0

            # to know climate info
        elif "weather" or "climate" in statement:
            speak("checking climate")
            weather_result = weather(city)
            speak(weather_result)
            statement = 0
            continue









