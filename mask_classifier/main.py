from flask import Flask, render_template, Response
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
import cv2
import os
import random
import sys
import mysql.connector
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

