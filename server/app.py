import os
import csv
import cv2

import tkinter as tk

import tornado.ioloop
import tornado.web
import tornado.escape

from urllib.parse import urlencode

from server.face_algo import process_image, get_encodings_from_json , encodings2
from server.voice_algo import record_audio_train,train_model,record_audio_test,test_model


class MainHandler(tornado.web.RequestHandler):
    def get(self, *args):

        if args[0] == 'index':
            self.render("index.html")

        elif args[0] == 'login':
            self.render("login.html")

        elif args[0] == 'signup':
            self.render("signup.html")

        elif args[0] == 'meeting-details':
            self.render("meeting-details.html")

        elif args[0] == 'recognize':
            self.render("recognize.html")

        elif args[0] == 'recognise-face':
            self.perform_face_recognition(self.get_argument('username'))
            print('Recognise Face')
            self.render("recognize.html")

        elif args[0] == 'recognise-voice':
            self.perform_voice_recognition(self.get_argument('username'))
            print('Recognise Voice')
            self.render("recognize.html")

        elif args[0] == 'capture-image':
            print('Capture Image')
            self.capture_image(self.get_argument('username'))
            self.render("recognize.html")

        elif args[0] == 'record-voice':
            print('Record Voice')
            self.record_voice(self.get_argument('username'))
            self.render("recognize.html")

        elif args[0] == 'record':
            self.render("record.html")

        else:
            self.render("index.html")

    def post(self, *args):
        if args[0] == 'sign-up':
            body = self.request.body.decode('utf8')
            pairs = body.split('&')

            data_dict = {key_value.split('=')[0]: key_value.split('=')[1] for key_value in pairs}
            username = data_dict['username']
            password = data_dict['password']

            # TODO: Work on making username unique
            # with open('user_data.csv', 'r', newline='') as read:
            #     reader = csv.reader(read)
            #     for row in reader:
            #         print(row)

            with open('user_data.csv', 'a', newline='') as write:
                writer = csv.writer(write)
                if write.tell() == 0:
                    writer.writerow(["username", "password"])
                writer.writerow([username, password])

            image_folder = f"users/{username}/images"
            voice_folder = f"users/{username}/voice"

            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(voice_folder, exist_ok=True)

            self.redirect('record?' + urlencode({'username': username}))

        elif args[0] == 'log-in':
            body = self.request.body.decode('utf8')
            pairs = body.split('&')

            data_dict = {key_value.split('=')[0]: key_value.split('=')[1] for key_value in pairs}

            with open('user_data.csv', 'r', newline='') as read:
                reader = csv.reader(read)
                is_verified = False
                for row in reader:
                    username = row[0]
                    password = row[1]

                    if username == data_dict['username'] and password == data_dict['password']:
                        is_verified = True
                        break

                if is_verified:
                    print('Login Successful')
                    self.redirect('recognize?' + urlencode({'username': username}))
                    return
                else:
                    print('Login Failed')
                    self.redirect('index')
                    return

        else:
            self.redirect("index")

    def capture_image(self, name):
        image_folder = f"users/{name}/images"
        os.makedirs(image_folder, exist_ok=True)

        capture_window = tk.Tk()
        capture_window.title("Capture Images and Voice")
        capture_window.geometry("400x200")

        video_capture = cv2.VideoCapture(0)
        count = 0

        while count < 300:
            ret, frame = video_capture.read()
            cv2.imshow('Capture Images', frame)

            # Save each frame as an image
            image_name = f"{image_folder}/{name}_{count}.jpg"
            cv2.imwrite(image_name, frame)

            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        path = f"{image_folder}"
        encodings2(path)

        # Release the webcam and close the OpenCV window
        video_capture.release()
        cv2.destroyAllWindows()

        # Close the capture window
        capture_window.destroy()

    def record_voice(self, name):
        voice_folder = f"users/{name}/voice"
        os.makedirs(voice_folder, exist_ok=True)

        os.makedirs(f'{voice_folder}/test/testing_set', exist_ok=True)
        os.makedirs(f'{voice_folder}/train/trained_models', exist_ok=True)
        os.makedirs(f'{voice_folder}/train/training_set', exist_ok=True)

        capture_window = tk.Tk()
        capture_window.title("Capture Images and Voice")
        capture_window.geometry("400x200")

        record_audio_train(name, voice_folder)
        train_model(voice_folder)

        capture_window.destroy()

    def perform_face_recognition(self, name):
        encodings_folder = f"users/{name}/images"

        encodings_file = f'{encodings_folder}/image_encodings.json'
        encoded_data = get_encodings_from_json(encodings_file)

        haarcascade_file_path = 'haarcascade_frontalface_default.xml'
        recognized = process_image(haarcascade_file_path, encoded_data, name)
        if not recognized:
            print('Could not recognize the face, Returning to home page')
            self.redirect('index')

    def perform_voice_recognition(self, name):
        voice_folder = f"users/{name}/voice"

        record_audio_test(voice_folder)
        test_model(voice_folder)

def make_app():
    settings = {
        'template_path': 'templates/',
        'static_url_prefix': '/static/',
        'static_path': 'static',
    }

    return tornado.web.Application([
        (r"/(.*)", MainHandler),
    ], **settings)


if __name__ == "__main__":
    app = make_app()
    app.listen(4002)
    print('Server is up and running @4002')
    tornado.ioloop.IOLoop.current().start()
