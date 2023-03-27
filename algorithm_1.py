import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


def get_images(path):
    images = []
    class_names = []
    image_list = os.listdir(path)
    for image in image_list:
        read_current_img = cv2.imread(f'{path}/{image}')
        images.append(read_current_img)
        name = image.split('.')[0]
        class_names.append(name)

    return images, class_names


def find_encodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def mark_attendance(user):
    with open('attendance.csv', 'r+') as attendance_file:
        attendance_file.readlines()
        now = datetime.now()
        date_string = now.strftime('%H:%M:%S')
        attendance_file.writelines(f'\n{user},{date_string}')


def process_image(haarcascade_file, encoded_list, class_names):
    face_cascade = cv2.CascadeClassifier(haarcascade_file)
    capture = cv2.VideoCapture(0)   # 0 = default camera

    if not capture.isOpened():
        print('Error in video capturing')
        return

    while True:
        ret, img = capture.read()
        img_size = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(img_size, 1.1, 4)
        if not ret:
            print("failed to grab frame")
            break

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)   # For Image Frame

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            break

        faces_current_frame = face_recognition.face_locations(img_size)
        encoded_current_frame = face_recognition.face_encodings(img_size, faces_current_frame)

        for encoded_face, face_location in zip(encoded_current_frame, faces_current_frame):
            matches = face_recognition.compare_faces(encoded_list, encoded_face)
            print('matches', matches)
            face_distance = face_recognition.face_distance(encoded_list, encoded_face)
            print('face_distance', face_distance)
            match_index = np.argmin(face_distance)
            print('match_index', match_index)

            name = class_names[match_index].upper() if matches[match_index] else 'Unknown'
            mark_attendance(name)

            top, right, bottom, left = face_location
            cv2.putText(img, name, (left, bottom), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow('WebCam', img)

    cv2.waitKey()


def main():
    path = 'data_set_1'
    images, class_names = get_images(path)
    encoded_list = find_encodings(images)
    print('Encoding Complete')
    haarcascade_file_path = 'haarcascade_frontalface_default.xml'
    process_image(haarcascade_file_path, encoded_list, class_names)


if __name__ == '__main__':
    main()
