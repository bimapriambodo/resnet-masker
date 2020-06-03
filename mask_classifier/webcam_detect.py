import cv2
import os
import numpy as np
from pygame import mixer
import time
from label_detect import classify_face
from mtcnn.mtcnn import MTCNN
import time

color_dict={0:(0,0,255),1:(0,255,0)}
mixer.init()
sound = mixer.Sound('alarms.wav')

size = 10
face = cv2.CascadeClassifier(r"C:\Users\aiforesee\Documents\GitHub\observations\haarcascade_frontalface_default.xml")
detector = MTCNN()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2
counter = 0

while(True):
    ret, frame = cap.read()
    frame=cv2.flip(frame,1,1) #Flip to act as a mirror
    # Resize the image to speed up detection
    mini = cv2.resize(frame, (frame.shape[1] // size, frame.shape[0] // size))

    height_1,width_1 = frame.shape[:2]
    label= classify_face(frame)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = face.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    # )
    
    #mtcnn
    faces = detector.detect_faces(mini)

    if(label == 'with_mask'):
        a=1
        print("No Beep")
        counter = counter + 1
        if counter == 35:
            break

    elif(label == 'without_mask'):
        a=0
        counter = 0
        sound.play()
        print("Beep")

    else:
        pass

    # Draw rectangles around each face
    for result in faces:
        # get coordinates mtcnn
        # x, y, width, height = result['box']
        (x, y, width, height) = [v * size for v in result['box']] #Scale the shapesize backup
        cv2.rectangle(frame,(x,y),(x+width,y+height),color_dict[a],2)
        cv2.rectangle(frame,(x,y-40),(x+width,y),color_dict[a],-1)

        # cascade
        # x, y, w, h = result
        # cv2.rectangle(frame,(x,y-40),(x+w,y+h),color_dict[a],2)
        # cv2.rectangle(frame,(x,y-40),(x+h,y),color_dict[a],-1)

        cv2.putText(frame, str(label), (x, y-10),font,0.8,(255,255,255),2)

    cv2.putText(frame,str(label),(100,height_1-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    print("the label is", label)