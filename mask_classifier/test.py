import cv2
import imutils
from imutils.video import VideoStream
import time

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=480)
    cv2.rectangle(frame, (155, 38), (335, 308), (255,255,255), 2)
    x = cv2.rectangle(frame, (155, 38), (335, 308), (255,255,255), 2)
    print(x)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break