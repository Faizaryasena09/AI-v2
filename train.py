import tensorflow as tf
import numpy as np
import json
import os
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

nltk.download("punkt")

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
                train_x.append(data["input"].lower())  # Lowercase tanpa stemming
                train_y.append(data["output"].lower())  
            except json.JSONDecodeError:
                pass

if not train_x or not train_y:
    exit()

# Tokenizer hanya pada input, tidak termasuk output
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_x)

train_x_seq = tokenizer.texts_to_sequences(train_x)
max_length = max(len(seq) for seq in train_x_seq)  # Ambil max panjang input
train_x_pad = pad_sequences(train_x_seq, maxlen=max_length, padding="post")

# Label Encoding
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
train_y_pad = to_categorical(train_y_encoded, num_classes=len(label_encoder.classes_))

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=15000, output_dim=128, input_length=max_length),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
]

# Training
model.fit(train_x_pad, train_y_pad, epochs=10, batch_size=16, validation_split=0.1, verbose=2, callbacks=callbacks)

# Save Model
model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "w") as f:
    json.dump(tokenizer.to_json(), f)

with open(LABEL_ENCODER_PATH, "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

print("✅ Model AI berhasil dilatih & disimpan!")
