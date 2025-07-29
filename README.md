# 🎓 Career Guidance Platform – Final Report

An AI-powered career guidance platform that helps users discover suitable career paths based on their **interests, skills, and preferences**. The platform uses machine learning to analyze user responses and provides **personalized career recommendations**, along with a downloadable **PDF report**.

---

## 🚀 Project Overview

This system is built to guide students and professionals by analyzing their psychometric responses and generating career suggestions across **15 predefined career paths**. The platform includes:

- 🤖 ML-powered career prediction
- 📑 Insightful PDF report generation
- 💬 Chat-style frontend interface
- 🌐 Flask-based API backend

---

## 🧠 Components Implemented

### 1️⃣ Machine Learning Model
- **Model Type:** Multi-class Classification
- **Dataset:** Synthetic career test data (1000 entries)
- **Features:** 10 questions about preferences, work style, personality
- **Target Classes:** 15 career paths
- **Models Tested:** Logistic Regression, Random Forest, SVC
- **Best Model:** ✅ SVC with `C=1`, `gamma='auto'`, `kernel='rbf'`
- **Performance:** Moderate accuracy (limited feature set, extensible)

---

### 2️⃣ Backend API – Flask
- **Framework:** Flask 3.1.0
- **Key Endpoints:**
  - `/api/questions`: Get all questions
  - `/api/question`: Get next adaptive question
  - `/api/predict`: Predict career matches
  - `/api/generate-pdf`: Generate guidance report
- **Features:**
  - Adaptive questioning logic
  - Personality type detection
  - Confidence-based predictions
  - PDF report generation with insights

---

### 3️⃣ Frontend UI
- **Tech Stack:** HTML, CSS, JavaScript
- **Features:**
  - Chat-style interface with animations
  - Mobile-responsive and lightweight
  - Real-time response handling
  - Dynamic results display
  - PDF download functionality

---

## 💡 How to Use the Platform

1. **Start Test**: Click **"Start Career Test"** on homepage  
2. **Answer Questions**: Multiple-choice psychometric questions  
3. **View Results**:
   - Personality Type
   - Do’s and Don’ts
   - Top Career Matches (with confidence %)
   - Career Facts  
4. **Download Report**: Click **"Download PDF Report"**

---

## ⚙️ Technical Implementation

### 🧪 ML Pipeline
- **Preprocessing:**
  - Categorical encoding of answers
  - Feature scaling and splitting
- **Model Selection:**
  - Evaluated 3 models with F1-score, precision, recall
- **Best Result:**
  - Support Vector Classifier (SVC)
- **Tuning:**
  - GridSearchCV for hyperparameters

### 🧩 Backend Architecture
- **API Design:**
  - RESTful endpoints with stateless logic
- **PDF Generation:**
  - `fpdf` library for custom layout
  - Includes scores, insights, recommendations
- **Adaptive Logic:**
  - Rule-based question flow for personalization

### 🎨 Frontend Design
- **UX/UI:**
  - Chat-like flow
  - Mobile-first responsive layout
- **JavaScript:**
  - API calls with async handling
  - State tracking for user progress
  - PDF download integration

---

## 🛠️ Installation & Setup

### 🧾 Prerequisites
- Python 3.11+
- Flask (see `requirements.txt`)
- Any modern browser

### 📦 Local Setup

```bash
git clone https://github.com/ImaduddeenKhan/Career-Compass.git
cd Career-Compass

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py

