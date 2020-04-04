import cv2

def facechop(image):
    list_image_capt = []
    facedata = "../data/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image)
    faces = cascade.detectMultiScale(img)
    for x, y, w, h in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
        face_file_name = "../captured_pictures/face_" + str(x+y) + ".jpg"
        list_image_capt.append(face_file_name)
        cv2.imwrite(face_file_name, sub_face)
    return list_image_capt

def open_flux(flux):
    if (flux=='0'):
        cap = cv2.VideoCapture(0)
    else: 
        cap= cv2.VideoCapture(flux)

    facedata = "../data/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    while (True):
        ret,frame = cap.read()
            #la fonction cap.read() retourne un boolean true or false
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #detecter tous les visages au sein de la video
        faces=cascade.detectMultiScale(gray,scaleFactor=1.7,minNeighbors=5) # kant 1.5 w bdlta b 1.7

        i = 0
        for f in faces:
            x, y, w, h = [v for v in f]
            print(x,y,w,h)#afficher les abscisses de visage dans le video
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #roi_gray = gray[y:y+h,x:x+w]#region of interest--->le visage
            roi_color = frame[y:y+h,x:x+w]
            face_file_name="../captured_pictures/screen.jpg"
            cv2.imwrite(face_file_name,frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord(' '):
            image = '../captured_pictures/opencv' + str(i) + '.jpg'
            i += 1
            liste_images_captures = facechop(face_file_name)
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return liste_images_captures