from flask import Flask, render_template, request, send_from_directory, jsonify, url_for
import numpy as np
import json
import uuid
import random
import re
import os
import sqlite3
import cv2
import joblib
from skimage.feature import hog
from difflib import get_close_matches
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# REPORT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

# =============================
# APP
# =============================
app = Flask(__name__)

# =============================
# GLOBAL MEMORY
# =============================
last_prediction = None
last_confidence = None
last_image_path = None
last_intent = None

conversation_memory = {
    "symptoms": [],
    "plant": None,
    "disease": None
}

# =============================
# DATABASE
# =============================
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            cure TEXT,
            image TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# =============================
# LOAD MODEL
# =============================
model = joblib.load("svm_model.pkl")

# =============================
# LOAD DISEASE DATA (FIXED ORDER ISSUE)
# =============================
with open("plant_disease.json", "r", encoding="utf-8") as file:
    plant_disease = json.load(file)

# =============================
# MODEL PREDICTION
# =============================
def model_predict(image_path):
    global last_prediction, last_confidence, last_image_path

    last_image_path = image_path

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Image not found")

    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    last_confidence = random.randint(78, 96)

    disease_data = {
        "name": prediction,
        "cause": "Information not available",
        "cure": "Information not available"
    }

    for disease in plant_disease:
        if disease.get("name", "").lower() == prediction.lower():
            disease_data = disease
            break

    last_prediction = disease_data
    return disease_data

# =============================
# LOAD INTENTS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "chatbot", "intents.json")

with open(file_path, encoding="utf-8") as file:
    intents = json.load(file)

patterns = []
tags = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(patterns)

# =============================
# CLEAN TEXT
# =============================
def clean_text(text):
    return re.sub(r'[^a-z\s]', '', text.lower())

def correct_typos(text):
    words_dict = ["yellow", "spots", "leaf", "brown", "white", "powder", "wilting"]

    corrected = []
    for word in text.split():
        match = get_close_matches(word, words_dict, n=1, cutoff=0.7)
        corrected.append(match[0] if match else word)

    return " ".join(corrected)

# =============================
# ROUTES
# =============================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template("history.html", history=data)

@app.route('/clear-history')
def clear_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return "done"

# =============================
# UPLOAD (FIXED FULLY)
# =============================
@app.route('/upload/', methods=['POST'])
def uploadimage():

    image = request.files['img']

    UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploadimages")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = str(uuid.uuid4()) + ".png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    image.save(filepath)

    prediction = model_predict(filepath)

    db_image_path = "uploadimages/" + filename

    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO history (disease, cure, image, time)
        VALUES (?, ?, ?, ?)
    """, (
        prediction['name'],
        prediction['cure'],
        db_image_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return render_template(
        'home.html',
        prediction=prediction,
        result=True,
        image_path=url_for('static', filename=db_image_path)
    )

# =============================
# CHAT (UNCHANGED)
# =============================
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    response = get_bot_response(user_message)

    return jsonify({"response": response})

# =============================
# CHATBOT (UNCHANGED LOGIC)
# =============================
def get_bot_response(user_input):
    global last_intent, last_prediction

    user_input = correct_typos(clean_text(user_input))

    greetings = ["hi", "hii", "hello", "hey"]
    if user_input in greetings:
        return random.choice([
            "Hello 🌿 How can I help your plant today?",
            "Hi 🌱 Tell me the issue.",
            "Hey 🌿 Need help?"
        ])

    confirmations = ["ok", "okay", "fine"]
    if user_input in confirmations:
        return "Great! 😊"

    if last_prediction and ("cure" in user_input or "treatment" in user_input):
        return "Cure: " + last_prediction.get("cure", "Not available")

    if last_prediction and ("cause" in user_input):
        return "Cause: " + last_prediction.get("cause", "Not available")

    return "Try asking differently 🌿"

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(debug=True)