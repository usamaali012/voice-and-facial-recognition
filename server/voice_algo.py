import os
import wave
import time
import pickle
import pyaudio
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 4
# FILENAME = 'users/sample.wav'


def calculate_delta(array):
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


def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def record_audio_train(name, base_path):
    # name = input("Please Enter Your Name: ")
    # name = 'Usman'
    for count in range(5):
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

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)

        print("recording started")

        record_frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            record_frames.append(data)
        print("recording stopped")

        stream.stop_stream()
        stream.close()

        audio.terminate()

        output_file_name = name + "-sample" + str(count) + ".wav"
        wave_output_file_name = os.path.join(base_path, "train", "training_set", output_file_name)

        trained_file_list = open(f"{base_path}/train/training_set_addition.txt", 'a')
        trained_file_list.write(output_file_name + "\n")

        wave_file = wave.open(wave_output_file_name, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(record_frames))
        wave_file.close()


def record_audio_test(base_path):
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

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index=index,
                    frames_per_buffer=CHUNK)

    print ("recording started")

    record_frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        record_frames.append(data)

    print ("recording stopped")

    stream.stop_stream()
    stream.close()

    audio.terminate()

    output_file_name = "sample.wav"
    wave_output_file_name = os.path.join(base_path, "test/testing_set", output_file_name)

    testing_file_list = open(f"{base_path}/test/testing_set_addition.txt", 'a')
    testing_file_list.write(output_file_name+"\n")

    waveFile = wave.open(wave_output_file_name, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(record_frames))
    waveFile.close()


def train_model(base_path):
    source = f'{base_path}/train/training_set/'
    destination = f'{base_path}/train/trained_models/'
    train_file = f'{base_path}/train/training_set_addition.txt'

    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        print(path)

        sr, audio = read(source + path)
        print(sr)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            pickle_file = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(destination + pickle_file, 'wb'))
            print('+ modeling completed for speaker:', pickle_file, " with data point = ", features.shape)
            features = np.asarray(())
            count = 0

        count = count + 1


def test_model(base_path):
    source = f'{base_path}/test/testing_set/'
    model_path = f'{base_path}/train/trained_models/'
    test_file = f'{base_path}/test/testing_set_addition.txt'

    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(model_path, file_name) for file_name in os.listdir(model_path) if file_name.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(file_name, 'rb')) for file_name in gmm_files]
    speakers = [file_name.split("/")[-1].split(".gmm")[0] for file_name in gmm_files]
    # if len(models)==0:
    #     print("No user in the Database")
    #     return
    # sr,audio = read(FILENAME)
    # vector = extract_features(sr,audio)
    # log_likelihood = np.zeros(len(models))

    # Read the test directory and get the list of test audio files
    for path in file_paths:
        path = path.strip()

        sr, audio = read(source + path)
        # sr, audio = read(FILE_NAME)
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(models))

        print('*' * 100)
        print(log_likelihood)
        print('*' * 100)

    for i in range(len(models)):

        gmm = models[i]  # checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    pred = np.argmax(log_likelihood)
    identity = speakers[pred]

    print('*' * 100)
    print(log_likelihood)
    print('*' * 100)
    #
    # winner = np.argmax(log_likelihood)
    #
    # print("\tdetected as - ", speakers[winner])
    if identity == "Unknown":
        print("Not Recognizrd! Try again...")
        return
    print("Recognized as - ", identity)

    time.sleep(1.0)

# record_audio_train()
# train_model()
# record_audio_test()
# test_model()