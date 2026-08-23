import numpy as np
import json
import uuid
import random
import re
import os

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
    url_for,
    send_file
)

import sqlite3
import cv2
import joblib

from skimage.feature import hog
from difflib import get_close_matches
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# REPORT IMPORTS
# =============================
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    HRFlowable,
    Table,
    TableStyle
)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch


# ============================================================
# APP
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL MEMORY
# ============================================================

last_prediction = None
last_confidence = None
last_image_path = None
last_intent = None

conversation_memory = {
    "symptoms": [],
    "plant": None,
    "disease": None
}


# ============================================================
# DATABASE
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

# IMPORTANT:
# Prediction logic is kept exactly as your original code.

model = joblib.load("svm_model.pkl")


# ============================================================
# LOAD DISEASE DATA
# ============================================================

with open(
    "plant_disease.json",
    "r",
    encoding="utf-8"
) as file:

    plant_disease = json.load(file)


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_predict(image_path):

    global last_prediction
    global last_confidence
    global last_image_path

    last_image_path = image_path

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Image not found")

    # ========================================================
    # DO NOT CHANGE THIS PREDICTION LOGIC
    # ========================================================

    image = cv2.resize(image, (128, 128))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    # Keep your existing confidence logic
    last_confidence = random.randint(78, 96)

    # ========================================================
    # FIND DISEASE INFORMATION
    # ========================================================

    disease_data = {
        "name": prediction,
        "cause": "Information not available",
        "cure": "Information not available"
    }

    for disease in plant_disease:

        if disease.get(
            "name",
            ""
        ).lower() == prediction.lower():

            disease_data = disease
            break

    last_prediction = disease_data

    # Save disease to conversation memory
    conversation_memory["disease"] = prediction

    return disease_data


# ============================================================
# LOAD CHATBOT INTENTS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

file_path = os.path.join(
    BASE_DIR,
    "chatbot",
    "intents.json"
)


with open(
    file_path,
    encoding="utf-8"
) as file:

    intents = json.load(file)


# ============================================================
# PREPARE CHATBOT DATA
# ============================================================

patterns = []
tags = []

for intent in intents.get("intents", []):

    tag = intent.get("tag", "")

    for pattern in intent.get(
        "patterns",
        []
    ):

        patterns.append(pattern)
        tags.append(tag)


# ============================================================
# TF-IDF CHATBOT MODEL
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(patterns)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TYPO CORRECTION
# ============================================================

def correct_typos(text):

    words_dict = [
        "yellow",
        "spots",
        "leaf",
        "leaves",
        "brown",
        "white",
        "powder",
        "wilting",
        "black",
        "disease",
        "cure",
        "treatment",
        "cause",
        "plant",
        "fungus",
        "fungal",
        "rust",
        "mildew",
        "blight",
        "rot",
        "virus"
    ]

    corrected = []

    for word in text.split():

        match = get_close_matches(
            word,
            words_dict,
            n=1,
            cutoff=0.75
        )

        if match:
            corrected.append(match[0])
        else:
            corrected.append(word)

    return " ".join(corrected)


# ============================================================
# FIND INTENT
# ============================================================

def find_best_intent(user_input):

    cleaned_input = clean_text(user_input)

    corrected_input = correct_typos(
        cleaned_input
    )

    user_vector = vectorizer.transform(
        [corrected_input]
    )

    similarities = cosine_similarity(
        user_vector,
        X
    )[0]

    best_index = np.argmax(similarities)

    best_score = similarities[best_index]

    best_tag = tags[best_index]

    return best_tag, best_score


# ============================================================
# FIND INTENT DATA
# ============================================================

def get_intent_data(tag):

    for intent in intents.get(
        "intents",
        []
    ):

        if intent.get("tag") == tag:

            return intent

    return None


# ============================================================
# FIND DISEASE IN INTENTS
# ============================================================

def find_disease_intent(user_input):

    cleaned_input = clean_text(
        user_input
    )

    for intent in intents.get(
        "intents",
        []
    ):

        tag = intent.get(
            "tag",
            ""
        )

        patterns_list = intent.get(
            "patterns",
            []
        )

        for pattern in patterns_list:

            pattern_clean = clean_text(
                pattern
            )

            if pattern_clean and pattern_clean in cleaned_input:

                return intent

        if tag.replace("_", " ") in cleaned_input:

            return intent

    return None


# ============================================================
# CHATBOT
# ============================================================

def get_bot_response(user_input):

    global last_intent
    global last_prediction

    if not user_input:

        return "Please type something 🌱"


    # --------------------------------------------------------
    # CLEAN INPUT
    # --------------------------------------------------------

    original_input = user_input

    user_input = clean_text(
        user_input
    )

    user_input = correct_typos(
        user_input
    )


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not user_input:

        return "Please ask me something about your plant 🌱"


    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    greetings = [
        "hi",
        "hii",
        "hello",
        "hey",
        "good morning",
        "good evening",
        "good afternoon"
    ]

    if user_input in greetings:

        return random.choice([
            "Hello 🌿 How can I help your plant today?",
            "Hi 🌱 Tell me about your plant problem.",
            "Hey 🌿 You can describe your symptoms or upload a plant image."
        ])


    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    thanks_words = [
        "thanks",
        "thank you",
        "thx",
        "thanks a lot"
    ]

    if user_input in thanks_words:

        return random.choice([
            "You're welcome! 🌱",
            "Happy to help! 😊",
            "You're welcome. Take care of your plants! 🌿"
        ])


    # --------------------------------------------------------
    # CONFIRMATIONS
    # --------------------------------------------------------

    confirmations = [
        "ok",
        "okay",
        "fine",
        "alright",
        "got it"
    ]

    if user_input in confirmations:

        return "Great! 😊 Let me know if you need anything else."


    # ========================================================
    # QUESTIONS ABOUT CURRENT PREDICTION
    # ========================================================

    if last_prediction:

        # ----------------------------------------------------
        # CURE / TREATMENT
        # ----------------------------------------------------

        cure_words = [
            "cure",
            "treatment",
            "treat",
            "fix",
            "solution",
            "what should i do",
            "how to cure",
            "how do i treat"
        ]

        if any(
            word in user_input
            for word in cure_words
        ):

            cure = last_prediction.get(
                "cure",
                "Information not available"
            )

            return (
                f"🌿 Treatment for "
                f"{last_prediction.get('name', 'this disease')}:\n\n"
                f"{cure}"
            )


        # ----------------------------------------------------
        # CAUSE
        # ----------------------------------------------------

        cause_words = [
            "cause",
            "caused",
            "why",
            "reason",
            "what causes"
        ]

        if any(
            word in user_input
            for word in cause_words
        ):

            cause = last_prediction.get(
                "cause",
                "Information not available"
            )

            return (
                f"🔍 Cause of "
                f"{last_prediction.get('name', 'this disease')}:\n\n"
                f"{cause}"
            )


        # ----------------------------------------------------
        # CURRENT DISEASE INFORMATION
        # ----------------------------------------------------

        info_words = [
            "what is it",
            "what is this",
            "tell me about it",
            "what disease is this",
            "which disease"
        ]

        if any(
            phrase in user_input
            for phrase in info_words
        ):

            return (
                f"🌱 The detected disease is:\n\n"
                f"{last_prediction.get('name', 'Unknown')}"
            )


    # ========================================================
    # DIRECT DISEASE QUESTIONS
    # ========================================================

    disease_intent = find_disease_intent(
        user_input
    )

    if disease_intent:

        last_intent = disease_intent.get(
            "tag"
        )

        response_list = disease_intent.get(
            "responses",
            []
        )

        if response_list:

            response = random.choice(
                response_list
            )

        else:

            response = (
                "I found information about "
                f"{disease_intent.get('tag', 'this disease')}."
            )


        # ----------------------------------------------------
        # ADD CAUSE
        # ----------------------------------------------------

        cause = disease_intent.get(
            "cause"
        )

        cure = disease_intent.get(
            "cure"
        )


        if cause and (
            "cause" in user_input
            or "caused" in user_input
            or "why" in user_input
        ):

            return (
                f"🔍 Cause:\n\n{cause}"
            )


        if cure and (
            "cure" in user_input
            or "treatment" in user_input
            or "treat" in user_input
            or "fix" in user_input
        ):

            return (
                f"🌿 Treatment:\n\n{cure}"
            )


        if cause and cure:

            return (
                f"{response}\n\n"
                f"🔍 Cause: {cause}\n\n"
                f"🌿 Treatment: {cure}"
            )


        return response


    # ========================================================
    # TF-IDF INTENT MATCHING
    # ========================================================

    best_tag, score = find_best_intent(
        original_input
    )

    # Confidence threshold
    if score >= 0.20:

        intent_data = get_intent_data(
            best_tag
        )

        if intent_data:

            last_intent = best_tag

            responses = intent_data.get(
                "responses",
                []
            )

            if responses:

                return random.choice(
                    responses
                )


    # ========================================================
    # SYMPTOM KEYWORD FALLBACK
    # ========================================================

    symptom_keywords = {

        "yellow": (
            "Yellow leaves may indicate "
            "overwatering or nutrient deficiency 🌱"
        ),

        "brown spots": (
            "Brown spots may be caused by "
            "fungal infection, bacteria, or excess moisture 🍂"
        ),

        "black spots": (
            "Black spots may indicate a fungal "
            "or bacterial leaf disease 🍂"
        ),

        "white powder": (
            "White powder on leaves is commonly "
            "associated with powdery mildew 🌿"
        ),

        "orange spots": (
            "Orange or rusty spots may indicate "
            "rust disease 🍁"
        ),

        "wilting": (
            "Wilting can be caused by underwatering, "
            "root damage, disease, or heat stress 🌿"
        ),

        "curling": (
            "Leaf curling can be caused by viruses, "
            "pests, heat, or environmental stress 🌱"
        )
    }


    for keyword, response in symptom_keywords.items():

        if keyword in user_input:

            return response


    # ========================================================
    # FINAL FALLBACK
    # ========================================================

    return (
        "I'm not completely sure about that 🌿\n\n"
        "Try asking about a disease such as "
        "black spot, powdery mildew, rust disease, "
        "early blight, late blight, or bacterial spot."
    )


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():

    return render_template(
        'home.html'
    )


@app.route('/crop')
def crop():

    return render_template(
        'crop.html'
    )


@app.route('/fertilizer')
def fertilizer():

    return render_template(
        'fertilizer.html'
    )


@app.route('/dashboard')
def dashboard():

    return render_template(
        'dashboard.html'
    )


@app.route('/history')
def history():

    conn = sqlite3.connect(
        "history.db"
    )

    c = conn.cursor()

    c.execute(
        "SELECT * FROM history ORDER BY id DESC"
    )

    data = c.fetchall()

    conn.close()

    return render_template(
        'history.html',
        history=data
    )


@app.route('/clear-history')
def clear_history():

    conn = sqlite3.connect(
        "history.db"
    )

    c = conn.cursor()

    c.execute(
        "DELETE FROM history"
    )

    conn.commit()
    conn.close()

    return "done"


# ============================================================
# REPORT GENERATION
# ============================================================

@app.route('/report.pdf')
def download_report():

    report_path = os.path.join(
        app.root_path,
        "report.pdf"
    )


    # --------------------------------------------------------
    # CHECK PREDICTION
    # --------------------------------------------------------

    if last_prediction is None:

        return (
            "Please upload a plant image first "
            "to generate the report.",
            404
        )


    # --------------------------------------------------------
    # CREATE PDF
    # --------------------------------------------------------

    doc = SimpleDocTemplate(

        report_path,

        pagesize=A4,

        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()


    # --------------------------------------------------------
    # STYLES
    # --------------------------------------------------------

    title_style = ParagraphStyle(

        "TitleStyle",

        parent=styles["Title"],

        fontSize=22,

        leading=28,

        alignment=1,

        spaceAfter=10
    )


    subtitle_style = ParagraphStyle(

        "SubtitleStyle",

        parent=styles["BodyText"],

        fontSize=11,

        alignment=1,

        textColor=colors.grey,

        spaceAfter=20
    )


    heading_style = ParagraphStyle(

        "HeadingStyle",

        parent=styles["Heading2"],

        fontSize=14,

        leading=18,

        spaceBefore=15,

        spaceAfter=8
    )


    normal_style = ParagraphStyle(

        "NormalStyle",

        parent=styles["BodyText"],

        fontSize=11,

        leading=17
    )


    # --------------------------------------------------------
    # STORY
    # --------------------------------------------------------

    story = []


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Plant Disease Prediction Report",
            title_style
        )
    )


    story.append(
        Paragraph(
            "AI-Based Plant Disease Detection System",
            subtitle_style
        )
    )


    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.grey
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # REPORT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Prediction Result",
            heading_style
        )
    )


    prediction_name = last_prediction.get(
        "name",
        "Unknown"
    )


    confidence = (
        f"{last_confidence:.2f}%"
        if last_confidence is not None
        else "Not available"
    )


    report_data = [

        [
            Paragraph(
                "<b>Detected Disease</b>",
                normal_style
            ),

            Paragraph(
                str(prediction_name),
                normal_style
            )
        ],

        [
            Paragraph(
                "<b>Confidence</b>",
                normal_style
            ),

            Paragraph(
                confidence,
                normal_style
            )
        ],

        [
            Paragraph(
                "<b>Generated</b>",
                normal_style
            ),

            Paragraph(
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                normal_style
            )
        ]
    ]


    table = Table(
        report_data,
        colWidths=[
            2 * inch,
            4 * inch
        ]
    )


    table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )


    story.append(table)


    # --------------------------------------------------------
    # CAUSE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Cause",
            heading_style
        )
    )


    cause = last_prediction.get(
        "cause",
        "Information not available"
    )


    story.append(
        Paragraph(
            str(cause),
            normal_style
        )
    )


    # --------------------------------------------------------
    # TREATMENT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Recommended Treatment",
            heading_style
        )
    )


    cure = last_prediction.get(
        "cure",
        "Information not available"
    )


    story.append(
        Paragraph(
            str(cure),
            normal_style
        )
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if (
        last_image_path
        and os.path.exists(last_image_path)
    ):

        story.append(
            Paragraph(
                "Analyzed Image",
                heading_style
            )
        )


        try:

            img = Image(
                last_image_path,
                width=3.5 * inch,
                height=3.5 * inch
            )

            story.append(img)

            story.append(
                Spacer(1, 15)
            )

        except Exception:

            pass


    # --------------------------------------------------------
    # FOOTER INFORMATION
    # --------------------------------------------------------

    story.append(
        Spacer(1, 20)
    )


    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.grey
        )
    )


    story.append(
        Spacer(1, 10)
    )


    story.append(
        Paragraph(
            "This report was generated by the "
            "Plant Disease Prediction System.",
            normal_style
        )
    )


    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(story)


    # --------------------------------------------------------
    # DOWNLOAD PDF
    # --------------------------------------------------------

    return send_file(

        report_path,

        as_attachment=True,

        download_name=(
            "Plant_Disease_Prediction_Report.pdf"
        ),

        mimetype="application/pdf"
    )


# ============================================================
# UPLOAD
# ============================================================

@app.route(
    '/upload/',
    methods=['POST']
)
def uploadimage():

    image = request.files['img']


    UPLOAD_FOLDER = os.path.join(
        app.root_path,
        "static",
        "uploadimages"
    )


    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )


    filename = (
        str(uuid.uuid4())
        + ".png"
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    image.save(filepath)


    # IMPORTANT:
    # Existing prediction logic remains unchanged.

    prediction = model_predict(
        filepath
    )


    db_image_path = (
        "uploadimages/"
        + filename
    )


    conn = sqlite3.connect(
        "history.db"
    )

    c = conn.cursor()


    c.execute("""

        INSERT INTO history
        (disease, cure, image, time)

        VALUES (?, ?, ?, ?)

    """, (

        prediction['name'],

        prediction['cure'],

        db_image_path,

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))


    conn.commit()

    conn.close()


    return render_template(

        'home.html',

        prediction=prediction,

        result=True,

        image_path=url_for(
            'static',
            filename=db_image_path
        )
    )


# ============================================================
# CHAT API
# ============================================================

@app.route(
    '/chat',
    methods=['POST']
)
def chat():

    data = request.get_json()

    if not data:

        return jsonify({
            "response":
            "Please enter a message 🌱"
        })


    user_message = data.get(
        "message",
        ""
    )


    response = get_bot_response(
        user_message
    )


    return jsonify({
        "response": response
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )