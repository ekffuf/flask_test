from flask import Flask, request, jsonify
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
import math
import mariadb


base_path = r"/"
audio_path = base_path + r"\wav_filename"
save_path = base_path + r"\wav"
audio_list = os.listdir(audio_path)


def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    m4a_file = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    wav_path = encoded_path.replace(".m4a", ".wav")
    m4a_file.export(os.path.join(base_path, audio_path, wav_path), format="wav")
    return wav_path


def trim_audio_data(wav_path, save_file, start_time=0.0, sec=120):
    sr = 44100
    y, sr = librosa.load(wav_path, sr=sr)
    sec_total = int(librosa.get_duration(y=y, sr=sr))
    ny = y[start_time * sr:sr * (sec + start_time)]
    my = sf.write(save_file + f"_{str(start_time).zfill(5)}.wav", ny, sr)  # 고쳐야 함.
    return my


def cut_wav(my):
    audio_name = my
    if audio_name.find('wav'):
        audio_file = audio_path + "\\" + audio_name
        save_file = save_path + "\\" + audio_name[:-4]
        f = sf.SoundFile(audio_file)
        f_sec = f.frames // f.samplerate
        print(audio_file, " seconds, ", f_sec)

        sec = 120
        for i in range(math.ceil(f_sec / sec)):
            if i * sec > f_sec:
                break
            trim_audio_data(audio_file, save_file, i * sec, sec)
        data_list = [i for i in os.listdir(os.path.join(base_path, save_path)) if i.startswith(my[:-4])]
    return data_list


def transcribe_audio(data_list):
    text_list = []
    for i in data_list:
        r = sr.Recognizer()
        with sr.AudioFile(os.path.join(base_path, "../wav", i)) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language='ko-KR')
        text_list.append(text)

    return text_list


def concatenate_texts(text_list):
    concatenated_text = ' '.join(text_list)
    return concatenated_text

# 모델 호출
# --pre방식
with open("../model/tokenizer_pre.pickle", "rb") as f:
    tokenizer1 = pk.load(f)
model1 = load_model("../model/model_pre.h5")


# --post방식
# with open("tokenizer_post.pickle", "rb") as f:
#     tokenizer2 = pk.load(f)
# model2 = load_model("model_post.h5")

def predict(string):
    real_sequences1 = tokenizer1.texts_to_sequences([string])
    real_seq1 = pad_sequences(real_sequences1, maxlen=1000, truncating="pre")
    result1 = model1.predict(real_seq1)

    # real_sequences2 = tokenizer2.texts_to_sequences([string])
    # real_seq2 = pad_sequences(real_sequences2, maxlen=1000, truncating="post")
    # result2 = model2.predict(real_seq2)

    if (result1 >= 0.35):
        detect = 1
    else:
        detect = 0
    return detect


app = Flask(__name__)
api = Api(app)


@api.route("/api/client/file", methods=["POST"])
class HelloWorld(Resource):
    def post(self):
        if request.method == 'POST':
            file = request.files["file"]
            file.save(file.filename)
            print("[1/6] file saved")
            m4a_path = file.filename
            wav_path = m4a_wav_convert(m4a_path)
            print("[2/6] wav converted")
            cut = cut_wav(wav_path)
            print("[3/6] wav cut")
            stt = transcribe_audio(cut)
            print("[4/6] wav stt completed")
            text_final = concatenate_texts(stt)
            print("[5/6] text concatenated")
            detect = predict(text_final)
            print("[6/6] prediction completed")
            data = {'result': detect}

            conn = mariadb.connect(
                user="root",
                password="1234",
                host="127.0.0.1",
                port=3309,
                database="mysql",
            )

            cursor = conn.cursor()
            query = f"INSERT INTO flask(textt, result) values ('{text_final}','{detect}')"
            cursor.execute(query)
            conn.commit()
            cursor.close()
            conn.close()
            print("db-saved completed")

            return jsonify(data)

@api.route("/fraud/filename/<string:m4apath>", methods=["GET"])
class HelloWorld(Resource):
    def get(self, m4apath):
        wav_path = m4a_wav_convert(m4apath)
        cut = cut_wav(wav_path)
        stt = transcribe_audio(cut)
        text_final = concatenate_texts(stt)
        detect = predict(text_final)

        with open("voice.txt", "w", encoding="UTF-8") as f:
            f.write(text_final)

        # for filename in os.listdir(save_path):
        #     file_path = os.path.join(save_path, filename)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        #         print(f"{filename} 파일이 삭제되었습니다.")
        return {"result": detect}


if __name__ == "__main__":
    app.run(debug=True, port=9966)
