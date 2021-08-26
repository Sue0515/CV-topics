import numpy as np 
import cv2

# pretrained cascade model 
cascade_path = '..\sources\haarcascade_frontalface_default.xml'

# create haar cascade  
face_classifier = cv2.CascadeClassifier(cascade_path)

# settings
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontColor = (255,255,255)
lineType = 2

# open the camera 
cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    # capture frame by frame 
    ret, frame = cap.read()
   
    if ret is False:
        break

    # convert BGR to gray color 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces from the the video 
    faces = face_classifier.detectMultiScale(
        gray, 
        scaleFactor = 1.2, 
        minNeighbors = 5, 
        minSize = (30, 30)
    )

    # draw rectangle & text when detected face 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, 'Face Detected', (x, y - 10), font, 1, fontColor, lineType)
    # display the frame
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('x'):
        break 

# release the cap and destroy windows 
cap.release()
cv2.destroyAllWindows()

