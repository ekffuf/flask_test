# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from keras.utils import pad_sequences
from keras.models import load_model
from pydub import AudioSegment
import speech_recognition as sr
import soundfile as sf
import urllib.parse
import pickle as pk
import asyncio
import mariadb
import librosa
import math
import os
import re


M4A_PATH = "./downloaded_m4a"
SPLITWAV_PATH = "./cut_wav"


def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    encoded_path = re.sub('가-힣ㄱ-ㅎ', "", encoded_path)
    wav_dst = encoded_path.replace(".m4a", ".wav")
    m4a_src = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    m4a_src.export(wav_dst, format="wav")
    return wav_dst


def trim_audio_data(wav_filename, start_time=0.0, sec=120):
    sample_rate = 44100
    audio_array, sample_rate = librosa.load(wav_filename, sr=sample_rate)
    audio_splitted = audio_array[start_time * sample_rate:sample_rate * (sec + start_time)]
    sf.write(os.path.join(SPLITWAV_PATH, os.path.basename(wav_filename.replace(".wav", f"_{str(start_time).zfill(5)}.wav"))),
                          audio_splitted, sample_rate)
    return


def cut_wav(wav_filename):
    f = sf.SoundFile(wav_filename)
    total_sec = f.frames // f.samplerate
    print(f"{wav_filename} 파일의 총 길이는 {total_sec} 초입니다.")

    split_sec = 120
    interval_count = math.ceil(total_sec / split_sec)
    for i in range(interval_count):
        if i * split_sec > total_sec:
            break
        trim_audio_data(wav_filename, i * split_sec, split_sec)
    return


async def transcribe_audio(data_list):
    text_list = []
    for i in data_list:
        r = sr.Recognizer()
        with sr.AudioFile(os.path.join(SPLITWAV_PATH, i)) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language='ko-KR')
        text_list.append(text)
    return text_list


def concatenate_texts(text_list):
    concatenated_text = ' '.join(text_list)
    return concatenated_text


# 모델 호출
# --pre방식
with open("model/tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
model1 = load_model("model/model_pre.h5")


def predict(string):
    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)

    if (result1 >= 0.35):
        detect = 1
    else:
        detect = 0
    return detect


def get_datalist(wav_filename):
    basename = os.path.basename(wav_filename)
    data_list = [i for i in os.listdir(SPLITWAV_PATH) if i.startswith(os.path.splitext(basename)[0])]
    return data_list


app = Flask(__name__)
api = Api(app)


@api.route("/api/client/file/<string:user_id>/<string:declaration>", methods=["POST"])
class HelloWorld(Resource):
    async def post(self, user_id, declaration):
        global M4A_PATH, SPLITWAV_PATH
        if request.method == 'POST':
            file = request.files["file"]
            m4a_filename = os.path.join(M4A_PATH, file.filename)
            file.save(m4a_filename)
            wav_filename = m4a_wav_convert(m4a_filename)
            cut_wav(wav_filename)
            data_list = get_datalist(wav_filename)

            async def process_data(data_list):
                stt_result_list = await asyncio.gather(*[transcribe_audio(data) for data in data_list])
                text_final = concatenate_texts(stt_result_list)
                prediction = predict(text_final)
                data = {
                    'result': prediction
                }
                return data

            data = await process_data(data_list)
            declaration = re.sub("[^0-9]", "", declaration)
            return jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=9966)
