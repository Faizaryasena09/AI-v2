import tensorflow as tf
import numpy as np
import json
import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

MODEL_PATH = "models/chat_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"
LABEL_ENCODER_PATH = "models/label_encoder.json"
TRAINING_DATA = "train_data.json"

os.makedirs("models", exist_ok=True)

train_x, train_y = [], []
if os.path.exists(TRAINING_DATA):
    with open(TRAINING_DATA, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                train_x.append(data["input"])
                train_y.append(data["output"])
            except json.JSONDecodeError:
                print("❌ Format JSON salah, lewati baris.")

if not train_x or not train_y:
    print("❌ Data training kosong! Harap tambahkan data ke train_data.json")
    exit()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?!. ]", "", text)
    return text

train_x = [clean_text(text) for text in train_x]
train_y = [clean_text(text) for text in train_y]

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_x + train_y)

train_x_seq = tokenizer.texts_to_sequences(train_x)
train_x_pad = pad_sequences(train_x_seq, maxlen=50, padding="post")

label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
train_y_pad = to_categorical(train_y_encoded, num_classes=len(label_encoder.classes_))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=50),
    tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_x_pad, train_y_pad, epochs=30, batch_size=16, validation_split=0.1)

model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "w") as f:
    json.dump(tokenizer.to_json(), f)

with open(LABEL_ENCODER_PATH, "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

print("✅ Model AI berhasil dilatih & disimpan!")
