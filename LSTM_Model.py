# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mariadb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#------MariaDB연결
conn = mariadb.connect(
    user="root",
    password="hkit301301",
    host="182.229.34.184",
    port=3306,
    database="301project"
)

#DB메인코드
# 커서로 sql문 실행
cursor = conn.cursor()
#실행할 SQL문
query = "SELECT CONVERT(content USING UTF8),disdata from voicedata"
cursor.execute(query)
result = cursor.fetchall()
 # 커서닫기
cursor.close()
conn.close()

#--------------경로지정
X = [i[0] for i in result]
y = np.array([i[1] for i in result]).astype('float64')
df = pd.DataFrame({"document": X, "label": y})


#-------------토크나이저 설정
MAX_LEN = 1000
TRUNC = "pre"

train_input, val_input, train_target, val_target = train_test_split(df["document"], df["label"], test_size=0.4, stratify=df["label"])
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_input)
train_sequences = tokenizer.texts_to_sequences(train_input)

train_seq = pad_sequences(train_sequences, maxlen=MAX_LEN, truncating=TRUNC)
val_sequences = tokenizer.texts_to_sequences(val_input)
val_seq = pad_sequences(val_sequences, maxlen=MAX_LEN, truncating=TRUNC)


#--------------모델 층 나누기
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=100000, output_dim=64, input_length=MAX_LEN))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2, return_sequences=False)))
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_seq, train_target, epochs=50, batch_size=32, validation_data=(val_seq, val_target))

# 저장할 폴더 경로
folder_path = './model'

# 저장할 파일 이름
file_name = 'modelVer.h5'

# 이미 있는 파일들의 목록 가져오기
existing_files = os.listdir(folder_path)

# 같은 이름의 파일이 이미 존재하는지 확인
if file_name in existing_files:
    # 다른 이름으로 저장할 파일의 번호 결정
    next_number = 1
    while True:
        next_file_name = f'{file_name.split(".")[0]}_{next_number}.h5'
        if next_file_name not in existing_files:
            break
        next_number += 1
else:
    # 같은 이름의 파일이 없을 경우 그대로 저장
    next_file_name = file_name

# 저장할 파일의 경로 생성
next_file_path = os.path.join(folder_path, next_file_name)

# 모델 저장
model.save(next_file_path)


