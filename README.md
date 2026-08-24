# 🌱 Plant Disease Prediction and Chatbot Assistant using SVM

An AI-powered web application that detects plant diseases from leaf images using a **Support Vector Machine (SVM)** model and provides disease information, causes, recommended treatments, and an interactive plant-health chatbot.

---

## 📌 Project Overview

The **Plant Disease Prediction and Chatbot Assistant** is designed to help users identify possible plant diseases by uploading an image of a plant leaf.

The system:

* 📷 Accepts plant/leaf images from users
* 🔍 Extracts visual features using **HOG (Histogram of Oriented Gradients)**
* 🤖 Uses a trained **SVM machine-learning model** for disease prediction
* 🌿 Displays the predicted disease
* 📋 Provides disease cause and recommended cure
* 💬 Provides an interactive plant-health chatbot
* 📄 Generates a downloadable disease prediction report
* 📜 Stores previous predictions in a local history database
* 🌱 Provides crop recommendation and fertilizer-related features

---

# ✨ Features

### 🔬 Plant Disease Detection

Users can upload a plant image through the web interface.

The image is processed using:

```text
Image
  ↓
Resize
  ↓
Grayscale Conversion
  ↓
HOG Feature Extraction
  ↓
SVM Model
  ↓
Disease Prediction
```

### 🤖 AI-Based Prediction

The project uses:

* **HOG** for image feature extraction
* **SVM** for classification
* A trained model stored as `svm_model.pkl`

### 🌿 Disease Information

After prediction, the application displays:

* Disease name
* Cause
* Recommended cure/treatment
* Prediction confidence

Disease information is maintained in:

```text
plant_disease.json
```

### 💬 Plant Assistant Chatbot

The application contains a chatbot that can respond to plant-related questions such as:

```text
What is black spot?
What causes powdery mildew?
What is the cure?
What caused this disease?
My leaves are turning yellow
```

The chatbot uses:

* `intents.json`
* TF-IDF vectorization
* Pattern matching
* Disease information
* Current prediction context

### 📄 PDF Report Generation

Users can download a report containing:

* Detected disease
* Confidence
* Cause
* Recommended treatment
* Uploaded image
* Report generation date

### 📜 Prediction History

Previous predictions are stored using **SQLite**.

The history contains:

* Disease
* Cure
* Uploaded image
* Date/time

---

# 🛠️ Tech Stack

| Category           | Technology                       |
| ------------------ | -------------------------------- |
| Frontend           | HTML, CSS, Bootstrap, JavaScript |
| Backend            | Python, Flask                    |
| Machine Learning   | Scikit-learn                     |
| Image Processing   | OpenCV                           |
| Feature Extraction | HOG / scikit-image               |
| ML Algorithm       | Support Vector Machine (SVM)     |
| Chatbot            | TF-IDF + Intent Matching         |
| Database           | SQLite                           |
| Reports            | ReportLab                        |
| Model Storage      | Joblib                           |
| Data Format        | JSON                             |

---

# 🧠 System Architecture

```mermaid
flowchart TD

    A[👤 User] --> B[🌐 Flask Web Interface]

    B --> C{Choose Function}

    C -->|Upload Plant Image| D[📷 Image Upload]

    D --> E[🖼️ OpenCV Image Processing]

    E --> F[📐 Resize Image]

    F --> G[⚫ Convert to Grayscale]

    G --> H[🔍 HOG Feature Extraction]

    H --> I[🤖 SVM Model]

    I --> J[🌿 Disease Prediction]

    J --> K[📋 Disease Information]

    K --> L[plant_disease.json]

    K --> M[🖥️ Display Result]

    M --> N[📄 Generate PDF Report]

    M --> O[📜 Save Prediction History]

    O --> P[(SQLite Database)]

    C -->|Chat| Q[💬 Plant Assistant]

    Q --> R[🧹 Text Cleaning]

    R --> S[✏️ Typo Correction]

    S --> T[🔤 TF-IDF Processing]

    T --> U[🎯 Intent Matching]

    U --> V[💡 Chatbot Response]

    V --> B

    C -->|Dashboard| W[📊 Dashboard]

    C -->|Crop Recommendation| X[🌱 Crop Recommendation]

    C -->|Fertilizer| Y[🧪 Fertilizer Calculator]

    C -->|History| Z[📜 Prediction History]

    Z --> P
```

---

# 🔄 Disease Prediction Flow

```mermaid
flowchart LR

    A[Plant Leaf Image] --> B[OpenCV]

    B --> C[Resize to 128x128]

    C --> D[Grayscale]

    D --> E[HOG Feature Extraction]

    E --> F[SVM Model]

    F --> G[Predicted Disease]

    G --> H[Search plant_disease.json]

    H --> I[Cause]

    H --> J[Cure]

    I --> K[Display Result]

    J --> K

    G --> K
```

---

# 💬 Chatbot Flow

```mermaid
flowchart TD

    A[User Message] --> B[Flask /chat API]

    B --> C[Convert to Lowercase]

    C --> D[Remove Unnecessary Characters]

    D --> E[Correct Common Typos]

    E --> F{Greeting?}

    F -->|Yes| G[Greeting Response]

    F -->|No| H{Prediction Available?}

    H -->|Yes| I{Cause or Cure Question?}

    I -->|Cause| J[Return Disease Cause]

    I -->|Cure| K[Return Disease Cure]

    I -->|No| L[Intent / Pattern Matching]

    H -->|No| L

    L --> M[Generate Chatbot Response]

    G --> N[Display Response]

    J --> N

    K --> N

    M --> N
```

---

# 📂 Project Structure

```text
plant disease prediction advanced/
│
├── app.py
├── svm_model.pkl
├── plant_disease.json
├── history.db
├── report.pdf
│
├── chatbot/
│   └── intents.json
│
├── templates/
│   ├── home.html
│   ├── dashboard.html
│   ├── crop.html
│   ├── fertilizer.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   ├── bootstrap.min.css
│   │   └── style.css
│   │
│   ├── images/
│   │   └── ...
│   │
│   ├── sounds/
│   │   └── beep.mp3
│   │
│   └── uploadimages/
│       └── ...
│
└── README.md
```

---

# ⚙️ How the System Works

## 1. Image Upload

The user selects a plant image through the web interface.

## 2. Image Preprocessing

OpenCV loads the image and resizes it to:

```text
128 × 128 pixels
```

The image is then converted to grayscale.

## 3. Feature Extraction

The system extracts HOG features from the processed image.

HOG captures important visual structures such as:

* Edges
* Shapes
* Textures
* Leaf patterns

## 4. Disease Classification

The extracted features are passed to the trained SVM model.

```text
HOG Features
      ↓
SVM
      ↓
Predicted Disease
```

## 5. Disease Information

The predicted disease is matched against `plant_disease.json`.

The application retrieves:

```text
Disease
Cause
Cure
```

## 6. Result

The result is displayed on the web page along with the uploaded image.

---

# 💬 Example Chatbot Questions

The chatbot can handle questions such as:

```text
What is black spot?
What causes black spot?
What is the cure?
What is powdery mildew?
What causes rust disease?
My leaves are turning yellow
My plant is wilting
What is leaf blight?
What is bacterial spot?
What should I do?
```

---

# 📄 Report Generation

The application provides a **Download Report** option.

The generated PDF contains:

```text
Plant Disease Prediction Report

Detected Disease
Confidence

Cause

Recommended Treatment

Analyzed Image

Generated Date
```

---

# 🗄️ Database

SQLite is used to maintain prediction history.

### History Table

```text
id
disease
cure
image
time
```

Every successful image prediction can be stored for later viewing.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Chintana-Projects/Plant-Disease-Prediction-and-Chatbot-Assistant-using-SVM.git
```

## 2. Navigate to the Project

```bash
cd "plant disease prediction advanced"
```

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install flask numpy opencv-python scikit-image scikit-learn joblib reportlab
```

## 6. Run the Application

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000
```

---

# 🖥️ Application Workflow

```mermaid
sequenceDiagram

    actor User
    participant UI as Flask Web UI
    participant App as Flask Backend
    participant CV as OpenCV/HOG
    participant SVM as SVM Model
    participant JSON as Disease JSON
    participant DB as SQLite
    participant Report as PDF Generator

    User->>UI: Upload plant image
    UI->>App: POST /upload/
    App->>CV: Process image
    CV->>CV: Resize + grayscale
    CV->>CV: Extract HOG features
    CV->>SVM: Send features
    SVM-->>App: Predicted disease
    App->>JSON: Search disease information
    JSON-->>App: Cause + Cure
    App->>DB: Save prediction
    App-->>UI: Display prediction
    User->>UI: Download report
    UI->>Report: Generate PDF
    Report-->>User: Plant disease report
```

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Develop an accessible plant disease detection system.
2. Apply machine learning to plant image classification.
3. Extract meaningful visual features from plant images.
4. Provide useful disease information to users.
5. Provide an interactive plant-health assistant.
6. Maintain prediction history.
7. Generate downloadable disease reports.
8. Provide a simple web-based interface for users.

---

# 🔮 Future Improvements

Possible future improvements include:

* 🧠 CNN/Deep Learning-based image classification
* 🌿 Training with real-world plant images
* 📷 Better handling of complex backgrounds
* 🔍 Leaf segmentation/background removal
* 📊 Improved confidence estimation
* 🌱 Support for more plant species
* 📱 Mobile-friendly interface
* ☁️ Cloud deployment
* 🌍 Multilingual chatbot
* 📈 Model performance dashboard
* 🖼️ Real-world Google-image testing dataset

---

# 🏆 Academic Showcase

This project was selected for participation in the **NASSCOM Future Forge 2026 – Academic Showcase** at Taj Yeshwantpur, Bengaluru.

The project demonstrates the application of:

```text
Machine Learning
        +
Computer Vision
        +
Natural Language Processing
        +
Web Development
        +
Database Management
```

to develop an accessible plant-health assistance system.

---

# 👩‍💻 Developer

**Chintana B**

AIML Student
Sapthagiri NPS University
Expected Graduation: 2028

### Skills

```text
Python
Machine Learning
SQL
HTML
CSS
JavaScript
Computer Vision
```

### GitHub

**Chintana-Projects**

---

# 📜 License

This project is developed for academic and educational purposes.

---

## 🌱 Project Summary

> **Plant Disease Prediction and Chatbot Assistant** combines machine learning, computer vision, and a conversational assistant to help users identify plant diseases and understand possible causes and treatments through a simple web interface.
