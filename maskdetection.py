import cv2
import dlib
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('facemaskdetections.mp4',fourcc,30.0,(640,480))
while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    faces1 = face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)
    for (x, y, w, h) in faces1:
        if w>150 and h>150 and w<200 and h<200:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for face in faces:
        for (x, y, w, h) in faces1:
            if w>150 and h>150 and w<200 and h<200:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()   
cv2.destroyAllWindows()