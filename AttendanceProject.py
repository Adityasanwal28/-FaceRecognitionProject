import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Configuration
IMAGE_PATH = 'ImagesAttendance'
ATTENDANCE_FILE = 'Attendance.csv'


def load_images_and_names(path):
    images = []
    names = []

    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(path, filename))

            if image is not None:
                images.append(image)
                names.append(os.path.splitext(filename)[0])

    return images, names


def encode_faces(images):
    encode_list = []

    for img in images:
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list


def mark_attendance(student_name):
    with open(ATTENDANCE_FILE, 'a') as f:
        now = datetime.now()
        dt_string1 = now.strftime('%d/%m/%Y')
        dt_string2 = now.strftime('%H:%M:%S')
        f.write(f'\n{student_name},{dt_string1},{dt_string2}')


def main():
    # Load student images and their names
    images_list, class_names = load_images_and_names(IMAGE_PATH)
    encode_list_known = encode_faces(images_list)
    print('Encoding Complete')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize a set to keep track of students marked as present
    present_students = set()

    while True:
        success, frame = cap.read()
        img_s = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings in the current frame
        faces_cur_frame = face_recognition.face_locations(img_s)
        encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                current_student_name = class_names[match_index].upper()

                # Check if the student has not already been marked as present in this session
                if current_student_name not in present_students:
                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, current_student_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                2)
                    mark_attendance(current_student_name)

                    # Add the student to the set of present students
                    present_students.add(current_student_name)

        # Display the webcam feed
        cv2.imshow('Webcam', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
