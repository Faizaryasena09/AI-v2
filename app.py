from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import json
import re
import nltk
import ast
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

MODEL_PATH = "models/chat_model.keras"
TOKENIZER_PATH = "models/tokenizer.json"
LABEL_ENCODER_PATH = "models/label_encoder.json"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer = tokenizer_from_json(json.load(f))
    with open(LABEL_ENCODER_PATH, "r") as f:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(json.load(f))
else:
    model, tokenizer, label_encoder = None, None, None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?+*/().-]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("indonesian")]
    return " ".join(words)

def is_math_expression(text):
    return bool(re.match(r"^[0-9+\-*/(). ]+$", text))

def calculate_math_expression(expression):
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except:
        return "Perhitungan tidak valid."

def softmax(x, temperature=0.7):
    x = np.array(x) / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.json
    with open("train_data.json", "a") as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"message": "Data percakapan tersimpan!"})

@app.route("/use", methods=["POST"])
def use():
    if not model or not tokenizer or not label_encoder:
        return jsonify({"error": "Model belum dilatih!"})

    user_input = request.json.get("input").strip()
    if is_math_expression(user_input):
        return jsonify({"output": calculate_math_expression(user_input)})

    clean_input = preprocess_text(user_input)
    input_seq = tokenizer.texts_to_sequences([clean_input])
    input_pad = pad_sequences(input_seq, maxlen=model.input_shape[1], padding="post")

    result = model.predict(input_pad)[0]
    probabilities = softmax(result, temperature=0.5)
    predicted_label = np.random.choice(len(result), p=probabilities)
    response_text = label_encoder.inverse_transform([predicted_label])[0]

    return jsonify({"output": response_text})

if __name__ == "__main__":
    app.run(debug=True)
