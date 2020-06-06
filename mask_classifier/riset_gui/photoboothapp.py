from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import numpy as np
from pygame import mixer
import time
from label_detect import detection_image
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import random
import sys

class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
		# initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!", command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        #tambahan
        self.color_dict={0:(0,0,255),1:(0,255,0)}
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        mixer.init()
        self.sound = mixer.Sound('alarms.wav')
        self.cam_sound = mixer.Sound('camera.wav')
        self.counter = 0
        self.path = r"C:/Users/aiforesee/Google Drive (bimapriambodowr@gmail.com)/Digital Rise Indonesia/Object Detection/Masker Detection - Resnet/mask_classifier/face_detector/"
        self.prototxtPath = os.path.sep.join([self.path,"deploy.prototxt"])
        self.weightsPath = os.path.sep.join([self.path,"res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
        self.dummy = 38
        self.dummy_2 = str(self.dummy) + " Degree C"
        self.a = None
        self.roi = None
        self.label = None

    def detect_and_predict_mask(self, frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
        try:
            (self.h, self.w) = self.frame.shape[:2]
            self.blob = cv2.dnn.blobFromImage(self.frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            self.detections = faceNet.forward()
            # initialize our list of faces, their corresponding locations,
            # # and the list of predictions from our face mask network
            self.faces = []
            self.locs = []
            self.preds = []
            # loop over the detections
            for i in range(0, self.detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # # the detection
                self.confidence = self.detections[0, 0, i, 2]
                    # filter out weak detections by ensuring the confidence is
                    # greater than the minimum confidence
                # if confidence > args["confidence"]:
                if self.confidence > 0.5:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the object
                    self.box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = self.box.astype("int")
                        # ensure the bounding boxes fall within the dimensions of
                        # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                        # extract the face ROI, convert it from BGR to RGB channel
                        # ordering, resize it to 224x224, and preprocess it
                    self.face = self.frame[startY:endY, startX:endX]

                    self.face_2 = cv2.cvtColor(self.face, cv2.COLOR_BGR2RGB)
                    self.face_2 = cv2.resize(self.face, (224, 224))
                    self.face_2 = img_to_array(self.face)
                    self.face_2 = preprocess_input(self.face)
                    self.face_2 = np.expand_dims(self.face, axis=0)
                        # add the face and bounding boxes to their respective
                        # lists
                    self.faces.append(self.face_2)
                    self.locs.append((startX, startY, endX, endY))
        except:
            pass
                    
        return (self.locs, self.faces, self.face) #preds
    
    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=480)
                cv2.rectangle(self.frame, (155, 38), (335, 308), (255,255,255), 2)
                # roi y1:y2, x1:x2
                self.roi = self.frame[38:308, 155:335] #ROI
                (self.locs, self.faces, self.face) = detect_and_predict_mask(self.roi, self.faceNet) #preds
                # self.label = detection_image.classify_face(self.roi)
                # print(label)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                
                # if the panel is not None, we need to initialize i
                if self.panel is None:s
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)
                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
        except: #RuntimeError, e:
            print("[INFO] caught a RuntimeError")
        

    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        # save the file
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))
    
    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        cv2.destroyAllWindows()
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()