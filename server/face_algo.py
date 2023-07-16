import time

import cv2
import json
import numpy as np
import face_recognition
import os
from datetime import datetime


def get_images(root_path):
    data = {}
    folders = os.listdir(root_path)
    for folder in folders:
        data[folder] = []
        path = f'{root_path}/{folder}'
        images_path = os.listdir(path)
        for image in images_path:
            read_current_img = cv2.imread(f'{path}/{image}')
            data[folder].append(read_current_img)

    return data


def create_encodings(data):
    encoded_data = {}

    for key, images in data.items():
        encoded_data[key] = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encoded_data[key].append(encode.tolist())

    with open('image_encodings.json', 'w') as file:
        json.dump(encoded_data, file)

def encodings2(path):
    data = { }
    # path = "users/najam sawera/images"
    path_images = os.listdir(path)
    print(path_images)
    for index, image in enumerate(path_images):
        data[index] = []
        read_current_img = cv2.imread(f'{path}/{image}')
        data[index].append(read_current_img)


    encoded_data = {}
    for key, images in data.items():
        encoded_data[key] = []
        # print(key)
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # encode = face_recognition.face_encodings(img)[0]
            encode = face_recognition.face_encodings(img)
            if len(encode) > 0:
                encode = encode[0]
                encoded_data[key].append(encode.tolist())
                print(key)

    # final_data = [value[0] for key, value in encoded_data.items()]
    final_data = [value[0] for key, value in encoded_data.items() if len(value)> 0]

    with open(f'{path}/image_encodings.json', 'w') as file:
        json.dump({'encodings': final_data}, file)


def get_encodings_from_json(file_path):
    encoded_data = {}

    with open(file_path, 'r') as file:
        data = json.load(file)

    for key, values in data.items():
        encoded_data[key] = []
        for value in values:
            np_array = np.array(value)
            encoded_data[key].append(np_array)

    return encoded_data


def mark_attendance(user):
    print('mark_attendance')
    with open('Attendance.csv', 'r+') as attendance_file:
        attendance_file.readlines()
        now = datetime.now()
        date_string = now.strftime('%H:%M:%S')
        attendance_file.writelines(f'\n{user},{date_string}')


def process_image(haarcascade_file, encoded_data, name):
    face_cascade = cv2.CascadeClassifier(haarcascade_file)
    capture = cv2.VideoCapture(0)   # 0 = default camera

    if not capture.isOpened():
        print('Error in video capturing')
        return

    print_name = 'Unknown'
    t0 = int(time.time())
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
            filter_results = {}
            for key, encoded_list in encoded_data.items():
                matches = face_recognition.compare_faces(encoded_list, encoded_face)
                print('matches', matches)
                face_distance = face_recognition.face_distance(encoded_list, encoded_face)
                print('face_distance', face_distance)
                match_index = np.argmin(face_distance)
                print('match_index', match_index)

                if matches[match_index]:
                    filter_results[key] = face_distance[match_index]

            print('filter_results', filter_results)
            values = [value for key, value in filter_results.items() if filter_results != {}]
            print('final matches', values)

            if len(values) > 0:
                min_ = min(values)
                print(min_)

                for key, value in filter_results.items():
                    if value == min_:
                        print_name = name
                        break

            # mark_attendance(name)
            top, right, bottom, left = face_location
            cv2.putText(img, print_name, (left, bottom), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow('WebCam', img)

            t1 = int(time.time())
            if (t1 - t0) >= 10:
                break

    cv2.waitKey()
    if print_name == name:
        return True
    else:
        return False



def main():
    # Run When New DataSet Added
    ################################
    # path = 'data_set_2'
    # data = get_images(path)
    # create_encodings(data)
    # print('Encoding Complete')
    ################################

    # encodings_file = 'image_encodings.json'
    # encoded_data = get_encodings_from_json(encodings_file)
    #
    # haarcascade_file_path = 'haarcascade_frontalface_default.xml'
    # process_image(haarcascade_file_path, encoded_data)
    encodings2(" ")


if __name__ == '__main__':
    main()
