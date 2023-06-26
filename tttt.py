from flask import Flask, request
from flask_restx import Api, Resource
from pydub import AudioSegment
import speech_recognition as sr
from keras.models import load_model
from keras.utils import pad_sequences
import urllib.parse
import os
import pickle as pk
import librosa
import soundfile as sf

def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    m4a_file = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    wav_path = encoded_path.replace(".m4a", ".wav")
    m4a_file.export(wav_path, format="wav")
    print(wav_path)
    return wav_path


def trim_audio_data(wav_path, save_file, start_time=0.0):
    sr = 44100
    y, sr = librosa.load(wav_path, sr=sr)
    sec = int(librosa.get_duration(y=y, sr=sr))
    ny = y[start_time * sr:sr * (sec + start_time)]
    my = sf.write(save_file + f"_{start_time}.wav", ny, sr)
    print(my)
    return my

audio_list = os.listdir(r"C:\Users\HKIT\PycharmProjects\yhdatabase")


def cut_wav(my):
    for audio_name in audio_list:
        if audio_name.find('wav') != -1:
            audio_file = my
            save_file = my + "\\" + "wav" + "\\" + audio_name[:-4]
            f = sf.SoundFile(audio_file)
            f_sec = f.frames // f.samplerate
            print(audio_file, " seconds, ", f_sec)

            sec = 30
            for i in range(f_sec - sec):
                if i * 30 > f_sec:
                    break
                a = trim_audio_data(audio_file, save_file, i * 30)
    return a


def transcribe_audio(a):
    r = sr.Recognizer()
    with sr.AudioFile(a) as source:
        audio = r.record(source)
    text = r.recognize_google(audio, language='ko-KR')
    print(text)
    return text

def concatenate_texts(text):
    concatenated_text = ' '.join(text)
    print(concatenated_text)
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


def predict(concatenated_text):
    string = concatenate_texts(concatenated_text)

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
        cut = cut_wav(wav_path)
        wav_final = transcribe_audio(cut)
        text_final = concatenate_texts(wav_final)
        detect = predict(text_final)
        return {"result": detect}


if __name__ == "__main__":
    app.run(debug=True, port=9966)
