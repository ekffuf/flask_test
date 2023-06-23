from flask import Flask
from flask_restx import Api, Resource
from keras.utils import pad_sequences
import speech_recognition as sr
import pickle as pk


#m4a를 wav파일로 변환
def m4a_wavconvent(path):

    return audio.write_audiofile(i.replace(".m4a", ".wav"))

def stt(wav):
    # 파일로부터 음성 불러오기, STT변환
    r = sr.Recognizer()
    with sr.AudioFile(m4a_wavconvent) as source:
        audio = r.record(source)
    r_text = r.recognize_google(audio, language='ko')
    return r_text

#모델 호출
#--pre방식
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pk.load(f)
with open("model_Bi-LSTM.pickle", "rb") as f:
    model = pk.load(f)

#--post방식
with open("tokenizer2.pickle", "rb") as f:
    tokenizer2 = pk.load(f)
with open("model2.pickle", "rb") as f:
    model2 = pk.load(f)

#새로운 음성 판별
i = stt()
real_sequences = tokenizer.texts_to_sequences(i)
real_seq = pad_sequences(real_sequences, maxlen=1000, truncating="pre")
result = model.predict(real_seq)

real_sequences2 = tokenizer2.texts_to_sequences(i)
real_seq2 = pad_sequences(real_sequences2, maxlen=1000, truncating="post")
result2 = model2.predict(real_seq2)

if result or result2 >= 0.35:
    detect = 1
else:
    detect = 0

app = Flask(__name__)
api = Api(app)


@api.route('/fraud/filename/<string:name>')  # 데코레이터 이용, '/hello' 경로에 클래스 등록
class HelloWorld(Resource):
    def hello_world(self,name):
        return {"result": detect}


     # def get(self, name):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
     #     m4a = name
     #     wav = m4a_wavconvent(m4a)
     #     text = stt(wav)
     #     detect = model(text)
     #     return {"result": detect}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)

