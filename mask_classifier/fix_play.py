import cv2
import os
import numpy as np
from pygame import mixer
from label_detect import classify_face
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import serial
import cv2
import os
import random
import sys
import mysql.connector
import win32api
try:
    import pkg_resources.py2_warn
except ImportError:
    pass
from datetime import datetime


def detect_face(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    try:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # initialize our list of faces, their corresponding locations,
        # # and the list of predictions from our face mask network
        faces = []
        locs = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # # the detection
            confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face_2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_2 = cv2.resize(face, (224, 224))
                face_2 = img_to_array(face)
                face_2 = preprocess_input(face)
                face_2 = np.expand_dims(face, axis=0)
                    # add the face and bounding boxes to their respective
                    # lists
                faces.append(face_2)
                locs.append((startX, startY, endX, endY))
    except:
        pass
                    
    return (locs, faces) 

def logical_conditions(preds, termal, detect_face):
    if (preds== "with_mask" and termal < 37 and len(detect_face) >0 ): #MASKER
        flag_conds = True #jadiin boolean agar efisien True False
        print("No Beep")
        
    elif (preds== "with_mask" and termal > 37 and len(detect_face) >0 ): #MASKER SUHU TINGGI
        flag_conds = False
        sound.play()
        print("Beep")

    elif (preds == "without_mask" and termal > 37 and len(detect_face) >0): #GAK MASKER SUHU TINGGI
        flag_conds = False
        sound.play()
        print("Beep")

    elif (preds == "without_mask" and termal < 37 and len(detect_face) >0): #GAK MASKER SUHU RENDAH
        flag_conds = False
        sound.play()
        print("Beep")
        
    elif (preds == "without_mask"  and len(detect_face) >0): #GAK MASKER 
        flag_conds = False
        sound.play()
        print("Beep")

    else:
        flag_conds= False

    return flag_conds

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

def write_db(photo, temp):
    cursor = db.cursor()
    sql = "INSERT INTO tb_history (date, photo, temperature) VALUES (%s, %s, %s)"
    time = datetime.now()
    empPicture = convertToBinaryData(photo)
    val = (time, empPicture, temp)
    cursor.execute(sql, val)
    db.commit()

def save_pict(frame,temp):
    path = r".\database"
    cv2.imwrite(os.path.join(path,'data.jpg'),frame)
    write_db(r".\database\data.jpg",temp)
    cam_sound.play()

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
path = r"./face_detector/"
prototxtPath = os.path.sep.join([path,"deploy.prototxt"])
weightsPath = os.path.sep.join([path,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# cap = cv2.VideoCapture(1)
#variable global
color_dict={False:(0,0,255),True:(0,255,0)}
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
mixer.init()
sound = mixer.Sound('alarms.wav')
cam_sound = mixer.Sound('camera.wav')
#ini sialisasi program save dan counter
already_saved = False
saveCount = 0
nSecond = 0
totalSec = 3
strSec = '321'
keyPressTime = 0.0
startTime = 0.0
timeElapsed = 0.0
startCounter = False
endCounter = False
flag_starttime = False
# dummy termal
# data_arduino = serial.Serial('COM3', 9600)
#database init
db = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="db_mask"
)

cur = db.cursor()
cur.execute("SELECT * FROM tb_setting")
for row in cur.fetchall():
    serial_port = row[2]

try:
  data_arduino = serial.Serial(serial_port, 9600)

except serial.serialutil.SerialException:
#   print ('Serial Port not open, please check your serial port and change from database')
  win32api.MessageBox(0, 'Serial Port not open, please check your serial port and change from database', 'Error')
  exit()


# program real-time
while True:
    x = vs.read()
    myData = (data_arduino.readline().strip())
    result = (myData.decode('utf-8'))
    if result == "Setting On...." :
        dummy = 0
    else :
        dummy = result
    print(type(dummy))
    dummy = float(dummy)
    dummy_2 = str(dummy) + " C"
    x = imutils.resize(x, width=480)
    cv2.rectangle(x, (155, 38), (335, 308), (255,255,255), 2)
    # cv.rec x1,y1 x2,y2
    #jika flag_starttime bernilai True maka start kondisi counter
    if not flag_starttime:
        startCounter = True
        startTime = datetime.now()
        print("startTime started")
        # endTime = datetime.now()
    try:
        # roi y1:y2, x1:x2
        frame = x[38:308, 155:335] #ROI
        # frame = x
        #deteksi wajah
        (locs, faces) = detect_face(frame, faceNet)
        #jika wajah terdeksi eksekusi scipt
        if len(faces) > 0 :
            #jalankan semuanya
            label = classify_face(frame) #prediksi
            #logical conditions
            flag_condition = logical_conditions(label, dummy, faces)
            #draw boundary
            for (box,wajah) in zip(locs,faces): 
                (startX, startY, endX, endY) = box
                # (x, y, width, height)
                cv2.putText(frame, str(label), (startX, startY - 10), font, 0.8, (255,255,255), 2)
                cv2.putText(frame, str(dummy_2), (startX, startY - 50), font, 1, (0,0,0), 4)
                cv2.putText(frame, str(dummy_2), (startX, startY - 50), font, 1, (255,255,255), 1)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[flag_condition], 2)
                cv2.rectangle(frame, (startX, startY-40), (endX, endY), color_dict[flag_condition], 2)

            # save pict, jika kondisi terpenuhi dan gambar belum disimpan
            if not already_saved and flag_condition:
                #mulai menghitung detik
                if startCounter:
                    flag_starttime = True #reset flag agar starttime tidak update
                    if nSecond < totalSec: 
                        # draw the Nth second on each frame 
                        # till one second passes  
                        cv2.putText(frame, strSec[nSecond], (startX, startY - 80), font, 2, (0,0,0), 2)
                        timeElapsed = (datetime.now() - startTime).total_seconds()
                        print('startTime: {}'.format(startTime))
                        print('timeElapsed: {}'.format(timeElapsed))

                        if timeElapsed >= 1:
                            nSecond += 1
                            print('nthSec:{}'.format(nSecond))
                            timeElapsed = 0
                            startTime = datetime.now()

                    elif nSecond >= totalSec:
                        save_pict(frame,dummy)
                        already_saved = True 
                        # startCounter = True
                        # saveCount += 1
                        nSecond = 0 
                        print("Succes Write into DB")
        
        elif len(faces) < 1:
            already_saved = False 
            startCounter = False 
            flag_starttime = False 
        
        print(nSecond) #debugging

    except Exception as e:
        print(e)

    # show the output frame
    cv2.imshow("Frame", x)
    # cv2.imshow("ROI", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

if __name__ == '__main__':
    print("the label is", label)