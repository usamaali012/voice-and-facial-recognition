from recognize import FaceAndVoiceRecognition


def main():
    haarcascade_file_path = 'haarcascade_frontalface_default.xml'
    encodings_file = 'image_encodings.json'
    attendance_file = 'attendance_2.csv'

    FaceAndVoiceRecognition(haarcascade_file_path, encodings_file, attendance_file)


if __name__ == '__main__':
    main()
