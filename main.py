#!/usr/bin/env python3

import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

names = []
know_face = []
dname = {}

for f in os.listdir('image'):
    n = f.split('.')[0]
    names.append(n)
    img = face_recognition.load_image_file('image/' + f)
    try:
        know_face.append(face_recognition.face_encodings(img)[0])
        dname[n] = know_face[-1]
    except IndexError:
        print('I wasn\'t able to locate any faces in at least one of the images. Check the image files. Aborting...')
        quit()

print(dname)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    count = 0
    text = ""
    for face_location in face_locations:
        top, right, bottom, left = face_location
        res = face_recognition.compare_faces(know_face, face_encodings[count])
        print(res)
        jcount = 0
        for r in res:
            print('{0}r={1}'.format(jcount,r))
            if r :
                text += names[jcount]
                frame = cv2.putText(frame, text, (140, 150), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            jcount += 1
        count += 1

    print('Found {0} face(s)'.format(count))
    cv2.imshow('frame', frame)
    print(names)
    if cv2.waitKey(1) == ord('q'):
        break
