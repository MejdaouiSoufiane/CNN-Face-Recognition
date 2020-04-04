import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detect():
    path = 'Video.mp4'
    cap = cv2.VideoCapture(path)
    while(True):

        ret,frame = cap.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.7,minNeighbors=5) # kant 1.5 w bdlta b 1.7
        k = 1
        for f in faces:
            x, y, w, h = [v for v in f]
            print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            face_file_name = "captured_pictures/face_" + str(k) + ".jpg"
            k = k + 1
            cv2.imwrite(face_file_name,roi_color)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

face_detect()