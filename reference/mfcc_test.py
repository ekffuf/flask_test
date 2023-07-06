# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from keras.utils import pad_sequences
from keras.models import load_model
from pydub import AudioSegment
import speech_recognition as sr
import soundfile as sf
import urllib.parse
import requests
import pickle as pk
import mariadb
import librosa
import math
import os
import re
import numpy as np


M4A_PATH = "../downloaded_m4a"
SPLITWAV_PATH = "../cut_wav"
m4a_filename = ""
WAV_PATH = "../convert_wav"


def m4a_wav_convert(path, WAV_PATH):
    encoded_path = urllib.parse.unquote(path)
    encoded_path = re.sub('가-힣ㄱ-ㅎ', "", encoded_path)
    wav_filename = os.path.basename(encoded_path).replace('.m4a', '.wav')
    wav_dst = os.path.join(WAV_PATH, wav_filename)
    m4a_src = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    m4a_src.export(wav_dst, format="wav")
    return wav_dst


# def wav_mfcc(wav_dst):
#     X_data = []
#     y_data = []
#     y, sr = librosa.load(wav_dst)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T
#     X_data.append(mfcc)
#     #y_data.append([0, 1] if mfcc else [1, 0])
#     if mfcc.shape[0] > 0:
#         y_data.append([0, 1])
#     else:
#         y_data.append([1, 0])
#     y_data = np.array(y_data)
#     return X_data, y_data

def wav_mfcc(wav_dst):
    X_data = []
    y_data = []
    y, sr = librosa.load(wav_dst)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T
    X_data.append(mfcc)

    # 각각의 MFCC에 해당하는 레이블을 생성하여 y_data에 추가
    for _ in range(len(X_data[0])):
        y_data.append([0, 1] if mfcc.shape[0] > 0 else [1, 0])

    y_data = np.array(y_data)
    return X_data, y_data


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


def transcribe_audio(data_list):
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


with open("../model/tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
model1 = load_model("../model/model_pre.h5")
model2 = load_model("../model/mfcc.h5")


def predict1(string):
    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)

    if (result1 >= 0.35):
        detect = 1
    else:
        detect = 0
    return detect


def predict2(X_data, y_data):
    result2 = model2.evaluate(x=X_data, y=y_data)[1]

    if (result2 >= 0.5):
        result_mfcc = 1
    else:
        result_mfcc = 0
    return result_mfcc


def get_datalist(wav_filename):
    basename = os.path.basename(wav_filename)
    data_list = [i for i in os.listdir(SPLITWAV_PATH) if i.startswith(os.path.splitext(basename)[0])]
    return data_list


app = Flask(__name__)
api = Api(app)


@api.route("/api/file", methods=["POST", "GET"])
class HelloWorld(Resource):
    def post(self):
        global M4A_PATH, SPLITWAV_PATH, m4a_filename, WAV_PATH
        if request.method == 'POST':
            file = request.files["file"]
            m4a_filename = os.path.join(M4A_PATH, file.filename)
            file.save(m4a_filename)
            wav_filename = m4a_wav_convert(m4a_filename, WAV_PATH)
            cut_wav(wav_filename)
            data_list = get_datalist(wav_filename)
            stt_result_list = transcribe_audio(data_list)
            text_final = concatenate_texts(stt_result_list)
            prediction = predict1(text_final)
            X_data, y_data = wav_mfcc(wav_filename)
            result_mfcc = predict2(X_data, y_data)
            data = {
                'result': prediction,
                'result2': result_mfcc
            }

            for filename1 in os.listdir(M4A_PATH):
                file_path1 = os.path.join(M4A_PATH, filename1)
                if os.path.isfile(file_path1):
                    os.remove(file_path1)
                    print(f"{filename1} 파일이 삭제되었습니다.")

            for filename2 in os.listdir(SPLITWAV_PATH):
                file_path2 = os.path.join(SPLITWAV_PATH, filename2)
                if os.path.isfile(file_path2):
                    os.remove(file_path2)
                    print(f"{filename2} 파일이 삭제되었습니다.")

            return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False, port=9966)
