# Emotion-Detection-By-applying-Cnn
This work is about detecting the Emotion of a person. The emotion of the person will be shown on the Front-end module which is based on web-development.
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
img_counter = 0
face_classifier = cv2.CascadeClassifier(r'C:\Users\SRI SANTHAN\PycharmProjects\cip project\Flask\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\SRI SANTHAN\PycharmProjects\cip project\Flask\model.h5')
emotion_labels =  ['Psychopath', 'Psychopath', 'Psychopath', 'sociopath', 'sociopath', 'Psychopath', 'Psychopath']
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    elif cv2.waitKey(33) == ord('i'):
        img_name = "C:/Users/SRI SANTHAN/PycharmProjects/cip project/Flask/pictures/pictures_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
cap.release()
cv2.destroyAllWindows()
