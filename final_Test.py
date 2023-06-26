from flask import Flask, request
from flask_restx import Api, Resource
from pydub import AudioSegment
import speech_recognition as sr
from keras.models import load_model
from keras.utils import pad_sequences
import urllib.parse
import os
import pickle as pk
import math


def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    m4a_file = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    wav_path = encoded_path.replace(".m4a", ".wav파일")
    m4a_file.export(wav_path, format="wav파일")
    return wav_path

def cut_audio(wav_path, start_time, end_time):
    audio = AudioSegment.from_file(wav_path)
    cut_audio = audio[start_time:end_time]
    return cut_audio

def transcribe_audio(wav_path):
    real = ""
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    text = r.recognize_google(audio, language='ko-KR')
    real += text
    return real

def concatenate_texts(text_list):
    concatenated_text = ' '.join(text_list)
    return concatenated_text


# 모델 호출
# --pre방식
with open("tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
model1 = load_model("model_pre.h5")


# --post방식
with open("tokenizer_post.pickle", "rb") as f:
    tokenizer2 = pk.load(f)
model2 = load_model("model_post.h5")


def predict(wav_path):
    string = transcribe_audio(wav_path)
    string = concatenate_texts(string)

    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)

    real_sequences2 = tokenizer2.texts_to_sequences([string])
    real_seq2 = pad_sequences(real_sequences2, maxlen=1000, truncating="post")
    result2 = model2.predict(real_seq2)

    if (result1 >= 0.35) or (result2 >= 0.35):
        detect = 1
    else:
        detect = 0
    return detect


app = Flask(__name__)
api = Api(app)


@api.route("/fraud/filename/<string:m4apath>")
class HelloWorld(Resource):
    def get(self, m4apath):
        wav_path = m4a_wav_convert(m4apath)
        detect = predict(wav_path)
        return {"result": detect}


if __name__ == "__main__":
    app.run(debug=True, port=9966)
