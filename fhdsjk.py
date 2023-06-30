from flask import Flask
from flask_sockets import Sockets

app = Flask(__name__)
sockets = Sockets(app)

@sockets.route('/')
def websocket(ws):
    while not ws.closed:
        message = ws.receive()
        print(message)
        for i in range(10):
            i += 1
        # 비동기 작업 처리
        ws.send('비동기 작업 완료')

if __name__ == '__main__':
    app.run()