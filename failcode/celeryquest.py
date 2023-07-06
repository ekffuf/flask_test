from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from keras.utils import pad_sequences
from keras.models import load_model
from pydub import AudioSegment
from celery import Celery
import speech_recognition as sr
import soundfile as sf
import urllib.parse
import pickle as pk
import mariadb
import librosa
import math
import os
import re

# M4A 파일을 저장할 경로
M4A_PATH = "../downloaded_m4a"

# WAV 파일을 분할하여 저장할 경로
SPLITWAV_PATH = "../cut_wav"


# M4A 파일을 WAV 파일로 변환하는 함수
def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    encoded_path = re.sub('가-힣ㄱ-ㅎ', "", encoded_path)
    wav_dst = encoded_path.replace(".m4a", ".wav")
    m4a_src = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    m4a_src.export(wav_dst, format="wav")
    return wav_dst


# WAV 파일을 분할하는 함수
def trim_audio_data(wav_filename, start_time=0.0, sec=120):
    sample_rate = 44100
    audio_array, sample_rate = librosa.load(wav_filename, sr=sample_rate)
    audio_splitted = audio_array[start_time * sample_rate:sample_rate * (sec + start_time)]
    sf.write(
        os.path.join(SPLITWAV_PATH, os.path.basename(wav_filename.replace(".wav", f"_{str(start_time).zfill(5)}.wav"))),
        audio_splitted, sample_rate)
    return


# WAV 파일을 여러 조각으로 분할하는 함수
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


# 분할된 WAV 파일의 음성을 텍스트로 변환하는 함수
def transcribe_audio(data_list):
    text_list = []
    for i in data_list:
        r = sr.Recognizer()
        with sr.AudioFile(os.path.join(SPLITWAV_PATH, i)) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language='ko-KR')
        text_list.append(text)
    return text_list


# 텍스트 리스트를 하나의 텍스트로 합치는 함수
def concatenate_texts(text_list):
    concatenated_text = ' '.join(text_list)
    return concatenated_text


# 전처리된 텍스트를 예측하여 결과를 반환하는 함수
def predict(string):
    with open("../model/tokenizer_pre.pickle", "rb") as f:
        tokenizer1 = pk.load(f)
    model1 = load_model("../model/model_pre.h5")

    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)
    if result1 >= 0.35:
        detect = 1
    else:
        detect = 0
    return detect


# Flask 애플리케이션 및 API 설정
app = Flask(__name__)
api = Api(app)

# Celery 객체 생성
celery = Celery(__name__, broker='amqp://ekffuf:12345678@localhost:25672//')
celery.conf.update(app.config)

# 오디오 처리를 수행하는 함수
@celery.task
def process_audio(audio_filename, user_id, declaration):
    # M4A 파일을 WAV 파일로 변환
    wav_filename = m4a_wav_convert(audio_filename)

    # WAV 파일을 분할
    cut_wav(wav_filename)

    # 분할된 WAV 파일의 음성을 텍스트로 변환
    wav_files = os.listdir(SPLITWAV_PATH)
    text_list = transcribe_audio(wav_files)

    # 텍스트 리스트를 하나의 텍스트로 합치기
    concatenated_text = concatenate_texts(text_list)

    # 전처리된 텍스트를 예측하여 결과 반환
    result1 = predict(concatenated_text)

    # 결과를 데이터베이스에 저장 또는 다른 처리 수행
    # ...

    # 작업 완료 여부를 확인하고 결과를 반환
    return {
        'result1': result1
    }

# 파일 업로드 및 작업 상태 확인을 위한 API 엔드포인트
@api.route('/audio-upload/audio')
class AudioUpload(Resource):
    def post(self):
        # 파일 업로드 처리
        audio_file = request.files['audio']
        audio_filename = os.path.join(M4A_PATH, audio_file.filename)
        audio_file.save(audio_filename)

        # 작업을 Celery에 전달
        result = process_audio.delay(audio_filename, request.form['user_id'], request.form['declaration'])

        # 작업 완료 여부를 확인하고 결과를 반환
        return {
            'task_id': result.id
        }


# 메인 애플리케이션 실행
if __name__ == "__main__":
    # Celery 워커 설정
    app.config['CELERY_BROKER_URL'] = 'amqp://ekffuf:12345678@localhost:25672//'
    app.config['CELERY_RESULT_BACKEND'] = 'amqp://ekffuf:12345678@localhost:25672//'

    # 애플리케이션 실행
    app.run(debug=True, port=9966)
