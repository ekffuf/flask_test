import math
import os
import librosa
import soundfile as sf
import speech_recognition as sr



def trim_audio_data(wav_path, save_file, start_time=0.0, sec=30):
    sr = 44100
    y, sr = librosa.load(wav_path, sr=sr)
    sec_total = int(librosa.get_duration(y=y, sr=sr))
    ny = y[start_time * sr:sr * (sec + start_time)]
    my = sf.write(save_file + f"_{str(start_time).zfill(5)}.wav", ny, sr)  # 고쳐야 함.
    return my


def cut_wav(my):
    audio_name = wav_path
    if audio_name.find('wav'):
        audio_file = audio_path + "\\" + audio_name
        save_file = save_path + "\\" + audio_name[:-4]
        f = sf.SoundFile(audio_file)
        f_sec = f.frames // f.samplerate
        print(audio_file, " seconds, ", f_sec)

        sec = 30
        for i in range(math.ceil(f_sec / sec)):
            if i * 30 > f_sec:
                break
            trim_audio_data(audio_file, save_file, i * 30, sec)
        data_list = [i for i in os.listdir(os.path.join(base_path, save_path)) if i.startswith(wav_path[:-4])]
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


if __name__ == '__main__':
    base_path = r"/"
    audio_path = base_path
    save_path = base_path + r"\wav"
    audio_list = os.listdir(audio_path)  # 고쳐야 함.
    audio_list = ["882.wav"]  # 임시 라인임!!!!!!!!!!!!
    os.chdir(base_path)
    wav_path = "882.wav"
    cut = cut_wav(wav_path)
    stt = transcribe_audio(cut)
    text_final = concatenate_texts(stt)
    text_name=""
    with open("882.txt","w",encoding="UTF-8") as asd:
        asd.write(text_name+"".join(text_final))
    print(text_final)
