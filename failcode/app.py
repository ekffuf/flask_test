from flask import Flask
from flask_restx import Api, Resource


app = Flask(__name__)
api = Api(app)


def to_wav(mp4):
    return mp4.replace(".mp4", ".wav")


def stt(wav):
    return "고객님 안녕하십니까? 이번에 SK텔레콤ㅇㄹ버"


def model(text):
    return 1


@api.route('/fraud/filename/<string:name>')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self, name):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        m4a = name
        wav = to_wav(m4a)
        text = stt(wav)
        result = model(text)
        return {"result": result}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
