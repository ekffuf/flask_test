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
    encoded_path = urllib.parse.unquote(path)
    m4a_file = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    wav_path = m4a_file.export(path.replace(".m4a", ".wav파일"), format="wav파일")
    return wav_path


def stt(wav):
    # 파일로부터 음성 불러오기, STT변환
    r = sr.Recognizer()
    with sr.AudioFile(m4a_wav_convert(wav)) as source:
            audio = r.record(source)
    r_text = r.recognize_google(audio, language='ko')
    return r_text


# 모델 호출
# --pre방식
with open("../model/tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
model1 = load_model("../model/model_pre.h5")

# --post방식
with open("../model/tokenizer_post.pickle", "rb") as f:
    tokenizer2 = pk.load(f)
model2 = load_model("../model/model_post.h5")


# 새로운 음성 판별
def predict(wav_path):
    string = stt(wav_path)
    string = " ".join(string)
    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)
    print(result1)

    real_sequences2 = tokenizer2.texts_to_sequences([string])
    real_seq2 = pad_sequences(real_sequences2, maxlen=1000, truncating="post")
    result2 = model2.predict(real_seq2)
    print(result2)

    if (result1 >= 0.35) or (result2 >= 0.35):
        detect = 1
    else:
        detect = 0
    return detect


app = Flask(__name__)
api = Api(app)


@api.route("/fraud/filename/<string:m4apath>")
class HelloWorld(Resource):
    def get(self, m4apath):  # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        m4apath = "./203.m4a"
        wav_path = m4a_wav_convert(m4apath)
        detect = predict(wav_path)
        return {"result": detect}


if __name__ == "__main__":
    # app.run(debug=True, host='182.229.34.184', port=9966)
    app.run(debug=True, port=9966)
