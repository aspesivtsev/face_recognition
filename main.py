import face_recognition
import cv2
import os
import numpy as np
from random import randrange


CWD = os.getcwd()
KNOWN_FACES_DIR = os.path.join(CWD, 'known_faces')
UNKNOWN_FACES_DIR = os.path.join(CWD, 'unknown_faces')
TOLERANCE = 0.5
FRAME_THICKNESS = 2
MODEL = "hog" #cnn
print(KNOWN_FACES_DIR)
print(UNKNOWN_FACES_DIR)
print("loading known faces!")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, name, filename))
        print(os.path.join(KNOWN_FACES_DIR, name, filename))
        encoding = face_recognition.face_encodings(image)
        known_faces.append(encoding)
        known_names.append(name)
        #print(known_faces)
        print(known_names)

#locating the cv2 engine and loading haar model
#cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
#haar_model = os.path.join(cv2_base_dir, r'data/haarcascades/haarcascade_frontalface_default.xml')
#print(haar_model)

#if os.path.isfile(haar_model):
#    print("I see da HAAR file!")
#else:
#    print("No HAAR file is located!")


#trained_face_data = cv2.CascadeClassifier(haar_model)

#processing webcam 0
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    #face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    locations = face_recognition.face_locations(grayscaled_img, model=MODEL)
    encodings = face_recognition.face_encodings(grayscaled_img, locations)


    for face_encoding, face_location in zip (encodings, locations):
        results = np.array(face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE))
        match = None
        if results.any():
            match = known_names[np.argmin(results)]
            print(f'{match}')
        x=face_location[0]
        y=face_location[1]
        w=face_location[2]
        h=face_location[3]
        cv2.rectangle(frame,(y, x), (h, w), (randrange(256),randrange(256),randrange(256)), FRAME_THICKNESS)
        #cv2.putText(frame,match(x, y), (x+w, y+h), (255,100,0), FRAME_THICKNESS)


            #(x, y, w, h) in face_coordinates:
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


'''
#This is cropping face
while True:

    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        #cv2.rectangle(frame,(x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)))
        #cv2.rectangle(frame,(x, y), (x+w, y+h), (255,100,0), 2)
        y=y-50 if y>50 else y
        x=x-50 if x>50 else x
        cropped = frame[y:y + h + 100, x:x + w + 100]

    cv2.imshow('Face Detection', cropped)
    cv2.waitKey(1)
'''

