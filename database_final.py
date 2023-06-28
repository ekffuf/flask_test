from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from keras.utils import pad_sequences
from keras.models import load_model
from pydub import AudioSegment
import speech_recognition as sr
import soundfile as sf
import urllib.parse
import pickle as pk
import mariadb
import librosa
import math
import os


audio_path = "./wav_files"
save_path = "./cut_wav"

def m4a_wav_convert(path):
    encoded_path = urllib.parse.unquote(path)
    m4a_file = AudioSegment.from_file(encoded_path, format="m4a", encoding="utf-8")
    wav_path = encoded_path.replace(".m4a", ".wav")
    m4a_file.export(wav_path, format="wav")
    return wav_path


def trim_audio_data(wav_path, save_file, start_time=0.0, sec=120):
    sr = 44100
    y, sr = librosa.load(wav_path, sr=sr)
    sec_total = int(librosa.get_duration(y=y, sr=sr))
    ny = y[start_time * sr:sr * (sec + start_time)]
    my = sf.write(save_file + f"_{str(start_time).zfill(5)}.wav", ny, sr)  # 고쳐야 함.
    return my


def cut_wav(my):
    if my.find('wav'):
        save_file = my[:-4]
        f = sf.SoundFile(my)
        f_sec = f.frames // f.samplerate
        print(my, " seconds, ", f_sec)

        sec = 120
        for i in range(math.ceil(f_sec / sec)):
            if i * sec > f_sec:
                break
            trim_audio_data(my, os.path.join(save_path, save_file), i * sec, sec)
        data_list = [i for i in os.listdir(save_path) if i.startswith(my[:-4])]
    return data_list


def transcribe_audio(data_list):
    text_list = []
    for i in data_list:
        r = sr.Recognizer()
        with sr.AudioFile(os.path.join(save_path, i)) as source:
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

# conn = mariadb.connect(
#                 user="root",
#                 password="1234",
#                 host="127.0.0.1",
#                 port=3309,
#                 database="mysql",
#             )


@api.route("/api/client/file", methods=["POST"])
class HelloWorld(Resource):
    def post(self):
        if request.method == 'POST':
            file = request.files["file"]
            file.save(os.path.join(audio_path, file.filename))
            m4a_path = os.path.join(audio_path, file.filename)
            wav_path = m4a_wav_convert(m4a_path)
            cut = cut_wav(wav_path)
            stt = transcribe_audio(cut)
            text_final = concatenate_texts(stt)
            detect = predict(text_final)

            # cursor = conn.cursor()
            # query = f"INSERT INTO flask(textt, result) values ('{text_final}','{detect}')"
            # cursor.execute(query)
            # conn.commit()
            # cursor.close()
            # conn.close()

            # for filename1 in os.listdir(save_path):
            #     file_path1 = os.path.join(save_path, filename1)
            #     if os.path.isfile(file_path1):
            #         os.remove(file_path1)
            #         print(f"{filename1} 파일이 삭제되었습니다.")
            #
            # for filename2 in os.listdir(audio_path):
            #     file_path2 = os.path.join(audio_path, filename2)
            #     if os.path.isfile(file_path2):
            #         os.remove(file_path2)
            #         print(f"{filename2} 파일이 삭제되었습니다.")

            data = {'result': detect}
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

        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"{filename} 파일이 삭제되었습니다.")
        return {"result": detect}


if __name__ == "__main__":
    app.run(debug=True, port=9966)
