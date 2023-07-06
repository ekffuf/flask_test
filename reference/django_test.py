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
from requests.exceptions import RequestException


M4A_PATH = "../downloaded_m4a"
SPLITWAV_PATH = "../cut_wav"

# 파일을 받았고 저장했다(스프링부트쪽에 알려주기)
def notify_file_received(filename):
    url = "http://127.0.0.1:9966/upload/"
    payload = {
        "event": "file_received",
        "filename": filename
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 요청이 성공적으로 전송되지 않으면 예외 발생
        print("File received successfully")
    except requests.exceptions.RequestException as e:
        print("Failed to send file received notification:", str(e))
        # 예외 처리를 위한 작업 수행 (예: 로깅, 오류 응답 반환 등)


def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    encoded_path = re.sub('가-힣ㄱ-ㅎ', "", encoded_path)
    wav_dst = encoded_path.replace(".m4a", ".wav")
    m4a_src = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    m4a_src.export(wav_dst, format="wav")
    return wav_dst

# 파일을 wav로 변환을 시켰다(스프링부트쪽에 알려주기)
def notify_wav_conversion(filename):
    url = "http://127.0.0.1:9966/upload/"
    payload = {
        "event": "wav_conversion",
        "filename": filename
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 요청이 성공적으로 전송되지 않으면 예외 발생
        print("WAV conversion successfully")
    except requests.exceptions.RequestException as e:
        print("Failed to send WAV conversion notification:", str(e))
        # 예외 처리를 위한 작업 수행 (예: 로깅, 오류 응답 반환 등)


def trim_audio_data(wav_filename, start_time=0.0, sec=120):
    sample_rate = 44100
    audio_array, sample_rate = librosa.load(wav_filename, sr=sample_rate)
    audio_splitted = audio_array[start_time * sample_rate:sample_rate * (sec + start_time)]
    sf.write(
        os.path.join(SPLITWAV_PATH, os.path.basename(wav_filename.replace(".wav", f"_{str(start_time).zfill(5)}.wav"))),
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

# 파일들을 STT로 변환을 시켰다(스프링부트쪽에 알려주기)
def notify_stt_conversion(text_list):
    url = "http://127.0.0.1:9966/upload/"
    payload = {
        "event": "stt_conversion",
        "text": text_list
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 요청이 성공적으로 전송되지 않으면 예외 발생
        print("STT conversion successfully")
    except requests.exceptions.RequestException as e:
        print("Failed to send STT conversion notification:", str(e))
        # 예외 처리를 위한 작업 수행 (예: 로깅, 오류 응답 반환 등)


def concatenate_texts(text_list):
    concatenated_text = ' '.join(text_list)
    return concatenated_text


# 모델 호출
# --pre방식
with open("../model/tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
model1 = load_model("../model/model_pre.h5")


def predict(string):
    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)

    if (result1 >= 0.35):
        detect = 1
    else:
        detect = 0
    return detect

# STT로 변환된 text들을 판별을 해봤다(스프링부트쪽에 알려주기)
def notify_prediction(prediction):
    url = "http://127.0.0.1:9966/upload/"
    payload = {
        "event": "prediction",
        "prediction": prediction
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 요청이 성공적으로 전송되지 않으면 예외 발생
        print("Prediction successfully")
    except requests.exceptions.RequestException as e:
        print("Failed to send prediction notification:", str(e))
        # 예외 처리를 위한 작업 수행 (예: 로깅, 오류 응답 반환 등)


def get_datalist(wav_filename):
    basename = os.path.basename(wav_filename)
    data_list = [i for i in os.listdir(SPLITWAV_PATH) if i.startswith(os.path.splitext(basename)[0])]
    return data_list


app = Flask(__name__)
api = Api(app)
url = "http://127.0.0.1:9966/upload/"  # Flask 서버에서 Django 서버로 요청을 보낼 URL

@api.route("/upload", methods=["POST"])
class FileUpload(Resource):
    def post(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})
        file = request.files['file']
        m4a_filename = os.path.join(M4A_PATH, file.filename)
        file.save(m4a_filename)
        notify_file_received(m4a_filename)
        wav_filename = m4a_wav_convert(m4a_filename)
        notify_wav_conversion(wav_filename)
        cut_wav(wav_filename)
        data_list = get_datalist(wav_filename)
        stt_result_list = transcribe_audio(data_list)
        notify_stt_conversion(stt_result_list)
        text_final = concatenate_texts(stt_result_list)
        prediction = predict(text_final)  # detect를 prediction으로 변경함
        notify_prediction(prediction)

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
        return jsonify({'result': prediction})


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=9966)
