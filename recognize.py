import os
import csv
import cv2
import time
import json
import wave
import pickle
import pyaudio
import numpy as np
import face_recognition
from datetime import datetime
from scipy.io.wavfile import read
from sklearn import preprocessing
import python_speech_features as mfcc


class FaceAndVoiceRecognition:
    def __init__(self,
                 haarcascade_file_path,
                 face_encodings_file,
                 attendance_file_path,
                 voice_source,
                 voice_model_path,
                 voice_test_file
                 ):
        self.haarcascade_algo = haarcascade_file_path
        self.face_encodings_file = face_encodings_file
        self.face_encodings = {}
        self.attendance_file = attendance_file_path
        self.voice_source = voice_source
        self.voice_model_path = voice_model_path
        self.voice_test_file = voice_test_file
        self.Format = pyaudio.paInt16
        self.Channels = 1
        self.Rate = 44100
        self.Chunk = 512
        self.RecordSeconds = 10
        self._get_encodings_from_json()
        face_name = self.recognize_face()
        if face_name != 'Unknown':
            voice_name = self.recognize_voice()
            voice = True if face_name == voice_name else False
            self._mark_attendance(face_name, face=True, voice=voice)
        else:
            print('Could Not Recognize This Face')

    def recognize_face(self):
        face_cascade = cv2.CascadeClassifier(self.haarcascade_algo)
        capture = cv2.VideoCapture(0)  # 0 = default camera

        if not capture.isOpened():
            print('Error in video capturing')
            return

        t0 = time.time()
        name = 'Unknown'
        while True:
            ret, img = capture.read()
            img_size = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = face_cascade.detectMultiScale(img_size, 1.1, 4)
            if not ret:
                print("failed to grab frame")
                break

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # For Image Frame

            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC pressed
                break

            faces_current_frame = face_recognition.face_locations(img_size)
            encoded_current_frame = face_recognition.face_encodings(img_size, faces_current_frame)

            for encoded_face, face_location in zip(encoded_current_frame, faces_current_frame):
                filter_results = {}
                for key, encoded_list in self.face_encodings.items():
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
                            name = key.upper()
                            break

                top, right, bottom, left = face_location
                cv2.putText(img, name, (left, bottom), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow('WebCam', img)

            t1 = time.time()
            if (t1 - t0) >= self.RecordSeconds:
                break

        # cv2.waitKey()
        capture.release()
        cv2.destroyAllWindows()
        return name

    def recognize_voice(self):
        self.record_audio()

        file_paths = open(self.voice_test_file, 'r')

        gmm_files = [os.path.join(self.voice_model_path, file_name) for file_name in os.listdir(self.voice_model_path) if
                     file_name.endswith('.gmm')]

        # Load the Gaussian gender Models
        models = [pickle.load(open(file_name, 'rb')) for file_name in gmm_files]
        speakers = [file_name.split("/")[-1].split(".gmm")[0] for file_name in gmm_files]

        # Read the test directory and get the list of test audio files
        person_name = 'Unknown'
        for path in file_paths:
            path = path.strip()

            sr, audio = read(self.voice_source + path)
            vector = self._extract_features(audio, sr)

            log_likelihood = np.zeros(len(models))

            for i in range(len(models)):
                gmm = models[i]  # checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

            winner = np.argmax(log_likelihood)

            person_name = speakers[winner]
            print("\tdetected as - ", person_name)

        return person_name

    def record_audio(self):
        audio = pyaudio.PyAudio()

        print("---------------------- Recording Device List ---------------------")

        info = audio.get_host_api_info_by_index(0)
        num_of_devices = info.get('deviceCount')
        for i in range(0, num_of_devices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

        print("-------------------------------------------------------------")

        index = int(input("Type Device Index To Use For Recording: "))

        print("recording via index " + str(index))

        stream = audio.open(format=self.Format, channels=self.Channels,
                            rate=self.Rate, input=True, input_device_index=index,
                            frames_per_buffer=self.Chunk)

        print("recording started")

        record_frames = []
        for i in range(0, int(self.Rate / self.Chunk * self.RecordSeconds)):
            data = stream.read(self.Chunk)
            record_frames.append(data)

        print("recording stopped")

        stream.stop_stream()
        stream.close()

        audio.terminate()

        output_file_name = "sample.wav"
        wave_output_file_name = os.path.join(self.voice_source, output_file_name)

        testing_file_list = open(self.voice_test_file, 'a')
        testing_file_list.write(output_file_name + "\n")

        waveFile = wave.open(wave_output_file_name, 'wb')
        waveFile.setnchannels(self.Channels)
        waveFile.setsampwidth(audio.get_sample_size(self.Format))
        waveFile.setframerate(self.Format)
        waveFile.writeframes(b''.join(record_frames))
        waveFile.close()

    def _extract_features(self, audio, rate):
        mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        # print(mfcc_feature)
        delta = self.calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

    def calculate_delta(self, array):
        rows, cols = array.shape
        # print(rows)
        # print(cols)
        deltas = np.zeros((rows, 20))
        N = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= N:
                if i - j < 0:
                    first = 0
                else:
                    first = i - j
                if i + j > rows - 1:
                    second = rows - 1
                else:
                    second = i + j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
        return deltas

    def _get_encodings_from_json(self):
        with open(self.face_encodings_file, 'r') as file:
            data = json.load(file)

        for key, values in data.items():
            self.face_encodings[key] = []
            for value in values:
                np_array = np.array(value)
                self.face_encodings[key].append(np_array)

        return self.face_encodings

    def _mark_attendance(self, user, face, voice):
        file_exists = os.path.isfile(self.attendance_file)
        with open(self.attendance_file, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(['Date', 'Time', 'Name', 'Face Recognition', 'Voice Recognition'])

            now = datetime.now()
            date_string = now.strftime('%d-%b-%Y')
            time_string = now.strftime('%I:%M:%S %p')

            face_result = 'True' if face else 'False'
            voice_result = 'True' if voice else 'False'
            writer.writerow([date_string, time_string, user, face_result, voice_result])
