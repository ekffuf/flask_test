# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import mariadb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


conn = mariadb.connect(
    user="root",
    password="hkit301301",
    host="182.229.34.184",
    port=3306,
    database="301project"
)

cursor = conn.cursor()
query = "SELECT CONVERT(content USING UTF8),disdata from voicedata"
cursor.execute(query)
result = cursor.fetchall()
cursor.close()
conn.close()

X = [i[0] for i in result]
y = np.array([i[1] for i in result]).astype('float64')
df = pd.DataFrame({"document": X, "label": y})


MAX_LEN = 1000
TRUNC = "pre"
train_input, val_input, train_target, val_target = train_test_split(df["document"], df["label"], test_size=0.4, stratify=df["label"])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_input)
train_sequences = tokenizer.texts_to_sequences(train_input)
train_seq = pad_sequences(train_sequences, maxlen=MAX_LEN, truncating=TRUNC)
val_sequences = tokenizer.texts_to_sequences(val_input)
val_seq = pad_sequences(val_sequences, maxlen=MAX_LEN, truncating=TRUNC)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=100000, output_dim=64, input_length=MAX_LEN))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2, return_sequences=False)))
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_seq, train_target, epochs=50, batch_size=32, validation_data=(val_seq, val_target))


folder_path = './model'
file_name = 'modelVer.h5'
existing_files = os.listdir(folder_path)

if file_name in existing_files:
    next_number = 1
    while True:
        next_file_name = f'{file_name.split(".")[0]}_{next_number}.h5'
        if next_file_name not in existing_files:
            break
        next_number += 1
else:
    next_file_name = file_name

next_file_path = os.path.join(folder_path, next_file_name)
model.save(next_file_path)


