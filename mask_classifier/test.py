import cv2
import os
import numpy as np
from pygame import mixer
import time
from label_detect import classify_face
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import random
import sys


color_dict={0:(0,0,255),1:(0,255,0)}
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
mixer.init()
sound = mixer.Sound('alarms.wav')
cam_sound = mixer.Sound('camera.wav')
counter = 0

def detect_and_predict_mask(frame, faceNet):
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
        preds = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # # the detection
            confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
            # if confidence > args["confidence"]:
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
                    
    return (locs, faces, face) #preds

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--face", type=str,
# 	default="face_detector",
# 	help="path to face detector model directory")
# ap.add_argument("-m", "--model", type=str,
# 	default="mask_detector.model",
# 	help="path to trained face mask detector model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
# prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
# weightsPath = os.path.sep.join([args["face"],
# 	"res10_300x300_ssd_iter_140000.caffemodel"])
path = r"C:/Users/aiforesee/Google Drive (bimapriambodowr@gmail.com)/Digital Rise Indonesia/Object Detection/Masker Detection - Resnet/mask_classifier/face_detector/"

prototxtPath = os.path.sep.join([path,"deploy.prototxt"])
weightsPath = os.path.sep.join([path,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

vs = VideoStream(src=0).start()
# cap = cv2.VideoCapture(1)
time.sleep(2.0)
dummy = 0

while True:
    x = vs.read()
    # ret, frame = cap.read()
    x = imutils.resize(x, width=480)
    cv2.rectangle(x, (155, 38), (335, 308), (0,0,0), 2)
    # cv.rec x1,y1 x2,y2

    try:
        # roi y1:y2, x1:x2
        frame = x[38:308, 155:335] #ROI
        (locs, faces, face) = detect_and_predict_mask(frame, faceNet) #preds
        label = classify_face(frame)
        dummy = sys.argv[1]
        dummy = int(dummy)
        print(type(dummyX))
        dummy_2 = str(dummy) + " C"
        if label == 'with_mask':
            label_2 = 0
        elif label == 'without_mask':
            label_2 = 1
        else:
            pass

        if (label_2== 0 & dummy < 37 & len(faces) >0 ): #MASKER
            a=1
            print("No Beep")
        
        elif (label_2== 0 & dummy > 37 & len(faces) >0 ): #MASKER SUHU TINGGI
            a=0
            sound.play()
            print("Beep")

        elif (label_2 == 1 & dummy > 37 & len(faces) >0): #GAK MASKER SUHU TINGGI
            a=0
            sound.play()
            print("Beep")

        elif (label_2 == 1 & dummy < 37 & len(faces) >0): #GAK MASKER SUHU RENDAH
            a=0
            sound.play()
            print("Beep")
        
        elif (label_2 == 1  & len(faces) >0): #GAK MASKER 
            a=0
            sound.play()
            print("Beep")

        else:
            a=0
        #draw boundary
        for box in locs: #pred, preds
            (startX, startY, endX, endY) = box
            # (x, y, width, height)

            cv2.putText(frame, str(label), (startX, startY - 10), font, 0.8, (255,255,255), 2)
            cv2.putText(frame, str(dummy_2), (startX, startY - 50), font, 1, (0,0,0), 4)
            cv2.putText(frame, str(dummy_2), (startX, startY - 50), font, 1, (255,255,255), 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[a], 2)
            cv2.rectangle(frame, (startX, startY-40), (endX, endY), color_dict[a], 2)

        #draw rectangle
        
        # cv2.putText(frame,str(label),(100,480-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        # cv2.imshow("Frame", frame)

        # save pict
        if a==1 :
            maxFrames = 4
            cpt = 0
            while cpt < maxFrames:
                cpt = cpt+1
                count = str(cpt)
                time.sleep(0.1)
                if cpt == 3:
                    path = r"C:\Users\aiforesee\Google Drive (bimapriambodowr@gmail.com)\Digital Rise Indonesia\Object Detection\Masker Detection - Resnet\mask_classifier\database"
                    time.sleep(0.5)
                    cv2.imwrite(os.path.join(path , 'pic{:}.jpg'.format(cpt)),x)
                    cam_sound.play()
                    time.sleep(0.5)
                    break
        elif len(faces)<1:
            maxFrames = 0
            cpt = 0
            a=0
        else:
            pass
    except:
        pass
    # show the output frame
    cv2.imshow("Frame", x)
    cv2.imshow("ROI", frame)
    # print(label)

    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

if __name__ == '__main__':
    print("the label is", label)