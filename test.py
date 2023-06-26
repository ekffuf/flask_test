import os
import librosa
import numpy as np
import soundfile as sf


def trim_audio_data(audio_file, save_file, start_time=0.0):
    sr = 44100
    y, sr = librosa.load(audio_file, sr=sr)
    sec = int(librosa.get_duration(y=y, sr=sr))
    ny = y[start_time * sr:sr * (sec + start_time)]

    sf.write(save_file + f"_{start_time}.wav", ny, sr)


base_path = r"C:\Users\HKIT\PycharmProjects\yhdatabase"
audio_path = base_path + r"\wav파일"
save_path = base_path + r"\wav"

audio_list = os.listdir(audio_path)

for audio_name in audio_list:
    if audio_name.find('wav') != -1:
        audio_file = audio_path + "\\" + audio_name
        save_file = save_path + "\\" + audio_name[:-4]

        f = sf.SoundFile(audio_file)
        f_sec = f.frames // f.samplerate
        print(audio_file, " seconds, ", f_sec)

        sec = 30
        for i in range(f_sec - sec):
            if i * 30 > f_sec:
                break
            trim_audio_data(audio_file, save_file, i * 30)

