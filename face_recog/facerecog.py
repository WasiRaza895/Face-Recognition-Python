import face_recognition_models

face_recognition_models.__path__ = ['models']

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)



# load known faces

my_image = face_recognition.load_image_file("my_image.png")
myimg_encoding = face_recognition.face_encodings(my_image)[0]
my_image2 = face_recognition.load_image_file("other.jpeg")
myimg2_encoding = face_recognition.face_encodings(my_image2)[0]

known_face_encodings = [myimg_encoding, myimg2_encoding]

known_face_names = ["wasi", "Fakhir"]

# list of expected students

students = known_face_names.copy() 

face_locations = []
face_encodings = []  
face_names = []
process_this_frame = True


# get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Corrected the color conversion

    # Recognize Faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:  # Corrected variable name
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance) 

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # add the text if the person is present
            if name in students:
                font = cv2.FONT_HERSHEY_COMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                linetype = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                            linetype)

                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])  # Removed duplicate line
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
