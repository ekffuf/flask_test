# -*- coding: utf-8 -*-

from flask import Flask
from flask_restx import Api, Resource
from keras.utils import pad_sequences
import speech_recognition as sr
from pydub import AudioSegment
import urllib.parse
import pickle as pk
import os
from keras.models import load_model

# m4a를 wav파일로 변환
def m4a_wav_convert(path):
    encoded_path = urllib.parse.quote("쇼핑_462.m4a")
    wav_path = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    wav_path.export("쇼핑_462.wav", format="wav")
    return wav_path


def stt(wav):
    # 파일로부터 음성 불러오기, STT변환
    r = sr.Recognizer()
    with sr.AudioFile(m4a_wav_convert) as source:
        audio = r.record(source)
    r_text = r.recognize_google(audio, language='ko')
    return r_text


# 모델 호출
# --pre방식
with open("tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
# with open("model1.pickle", "rb") as f:
#     model1 = pk.load(f)
model1 = load_model("model_pre.h5")

# --post방식
with open("tokenizer_post.pickle", "rb") as f:
    tokenizer2 = pk.load(f)
# with open("model2.pickle", "rb") as f:
#     model2 = pk.load(f)
model2 = load_model("model_post.h5")


# 새로운 음성 판별
def predict(wav_path):
    string = stt(wav_path)
    real_sequences1 = tokenizer1.texts_to_sequences(string)
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)

    real_sequences2 = tokenizer2.texts_to_sequences(string)
    real_seq2 = pad_sequences(real_sequences2, maxlen=1000, truncating="post")
    result2 = model2.predict(real_seq2)

    if result1 or result2 >= 0.35:
        detect = 1
    else:
        detect = 0
    return detect


app = Flask(__name__)
api = Api(app)


@api.route("/fraud/filename/<string:m4a_path>")
class HelloWorld(Resource):
    def get(self, m4a_path):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        print(os.getcwd())
        wav_path = m4a_wav_convert(m4a_path)
        detect = predict(wav_path)
        return {"result": detect}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
