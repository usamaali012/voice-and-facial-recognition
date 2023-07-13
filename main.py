from recognize import FaceAndVoiceRecognition


def main():
    haarcascade_file_path = 'server/haarcascade_frontalface_default.xml'
    encodings_file = 'image_encodings.json'
    attendance_file = 'attendance_2.csv'
    voice_source = 'test/testing_set/'
    voice_model_path = 'train/trained_models/'
    voice_test_file = 'test/testing_set_addition.txt'

    FaceAndVoiceRecognition(
        haarcascade_file_path,
        encodings_file,
        attendance_file,
        voice_source,
        voice_model_path,
        voice_test_file
    )


if __name__ == '__main__':
    main()
