import os
from glob import glob

from keras.utils import pad_sequences
from moviepy.editor import VideoFileClip
from tkinter.filedialog import askopenfilenames
from pydub import AudioSegment
import speech_recognition as sr
import pickle as pk

#동영상 파일을 음성(wav파일 등)변환
# -- coding: UTF-8 -*-
def mp4_wavconvent(path):
    video = VideoFileClip(i)
    video.audio.write_audiofile(i.replace(".mp4", ".wav"))

if __name__ == '__main__':
    filelist = askopenfilenames(initialdir="c:\\공유폴더",
                                defaultextension="mp4",
                                title="변환할 동영상 선택")
    for i in filelist:
        mp4_wavconvent(i)


#파일로부터 음성 불러오기, STT변환
r = sr.Recognizer()
with sr.AudioFile('0002.wav') as source:
    audio = r.record(source)

r_text = r.recognize_google(audio, language='ko')

# #STT변환한걸 합치기(ver2)
# with open(r_text, "w", encoding = "UTF-8") as f:
#     for i in os.listdir():
#         with open(i, encoding = "UTF-8") as dst:
#             text = dst.read().replace("\n", " ") + "\n"
#             f.write(text)



#모델 호출

with open("tokenizer.pickle", "wb") as f:
    tokenizer = pk.load(f)
with open("model_Bi-LSTM.pickle", "wb") as f:
    model = pk.load(f)


#새로운 음성 판별
i = r_text
real_sequences = tokenizer.texts_to_sequences(i)
real_seq = pad_sequences(real_sequences, maxlen=1000, truncating="pre")
result = model.predict(real_seq)


if result >= 0.35:
    detect = 1
else:
    detect = 0

