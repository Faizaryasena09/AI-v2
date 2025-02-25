from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import json
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")

app = Flask(__name__)

MODEL_PATH = "models/chat_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"
LABEL_ENCODER_PATH = "models/label_encoder.json"
TRAIN_DATA_PATH = "train_data.json"  # Data pelatihan yang sudah tersimpan

# Load model jika tersedia
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer = tokenizer_from_json(json.load(f))
    with open(LABEL_ENCODER_PATH, "r") as f:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(json.load(f))
else:
    model, tokenizer, label_encoder = None, None, None

# Load data pelatihan jika ada
if os.path.exists(TRAIN_DATA_PATH):
    with open(TRAIN_DATA_PATH, "r") as f:
        train_data = [json.loads(line) for line in f]
else:
    train_data = []

def preprocess_text(text):
    """ Hanya melakukan tokenisasi tanpa pembersihan karakter """
    words = word_tokenize(text)
    return " ".join(words)

def find_best_match(user_input):
    """ Mencari pertanyaan terdekat dari data pelatihan """
    input_words = set(user_input.split())

    best_match = None
    best_score = 0

    for data in train_data:
        question = preprocess_text(data["input"])
        question_words = set(question.split())

        # Gunakan Jaccard Similarity untuk mencari kesamaan
        intersection = len(input_words & question_words)
        union = len(input_words | question_words)
        similarity = intersection / union if union > 0 else 0

        if similarity > best_score:
            best_score = similarity
            best_match = data["output"]

    return best_match if best_score > 0.5 else None  # Hanya ambil jika cukup mirip

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    """ Menyimpan data untuk pelatihan model di masa depan """
    data = request.json
    with open(TRAIN_DATA_PATH, "a") as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"message": "Data percakapan tersimpan!"})

@app.route("/use", methods=["POST"])
def use():
    """ API untuk chatbot """
    if not model or not tokenizer or not label_encoder:
        return jsonify({"error": "Model belum tersedia, silakan latih dulu!"})

    user_input = request.json.get("input", "").strip()
    if not user_input:
        return jsonify({"error": "Input tidak boleh kosong!"})

    clean_input = preprocess_text(user_input)

    # Coba temukan jawaban dari dataset pelatihan
    best_match = find_best_match(clean_input)
    if best_match:
        return jsonify({"output": best_match})

    # Jika tidak ditemukan di dataset, gunakan model AI
    input_seq = tokenizer.texts_to_sequences([clean_input])
    input_pad = pad_sequences(input_seq, maxlen=model.input_shape[1], padding="post")

    result = model.predict(input_pad)[0]
    predicted_label = np.argmax(result)
    response_text = label_encoder.inverse_transform([predicted_label])[0]

    return jsonify({"output": response_text})

if __name__ == "__main__":
    app.run(debug=True)
