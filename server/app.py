import tornado.ioloop
import tornado.web
import tornado.escape

import csv

class MainHandler(tornado.web.RequestHandler):
    def get(self, *args):
        print(args[0])
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
            print('Recognise Face')
            self.render("recognize.html")
        elif args[0] == 'recognise-voice':
            print('Recognise Voice')
            self.render("recognize.html")
        else:
            self.render("index.html")

    def post(self, *args):
        if args[0] == 'sign-up':
            body = self.request.body.decode('utf8')
            pairs = body.split('&')

            data_dict = {key_value.split('=')[0]: key_value.split('=')[1] for key_value in pairs}

            # with open('user_data.csv', 'r', newline='') as read:
            #     reader = csv.reader(read)
            #     for row in reader:
            #         print(row)

            with open('user_data.csv', 'a', newline='') as write:
                writer = csv.writer(write)
                if write.tell() == 0:
                    writer.writerow(["username", "password"])
                writer.writerow([data_dict['username'], data_dict['password']])
            self.redirect("index")

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
                    self.redirect('recognize')
                    return
                else:
                    print('Login Failed')
                    self.redirect('index')
                    return
        else:
            self.redirect("index")

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
    tornado.ioloop.IOLoop.current().start()
