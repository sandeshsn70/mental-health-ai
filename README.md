# 🧠 MindSense AI — Mental Health Prediction & Chat Support System

> **IEEE Third Year Engineering Project**
> An end-to-end AI system for mental health prediction using NLP and Machine Learning, with an empathetic chatbot interface.

---

## 📌 Project Abstract

MindSense AI is a web-based application that analyses free-form text input (journal entries, thoughts, feelings) to predict a user's mental health state — **Depressed**, **Stressed**, **Anxious**, **Happy**, or **Neutral** — using a Natural Language Processing pipeline. The system also provides a chatbot-based emotional support interface and a mood-tracking dashboard.

**Core Stack:** React.js · Flask · TF-IDF + Logistic Regression · SQLite · Explainable AI
---

## 📊 Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 96.67%  |
| Precision | 97.14%  |
| Recall    | 96.67%  |
| F1 Score  | 96.64%  |
| CV (5-fold)| 98.50% ± 2.00% |

---

## 📁 Project Structure

```
mental-health-ai/
│
├── backend/                         # Flask Python Backend
│   ├── app.py                       # Main Flask API (all endpoints)
│   ├── requirements.txt             # Python dependencies
│   ├── mental_health.db             # SQLite database (auto-created)
│   │
│   └── model/
│       ├── train_model.py           # ML training script
│       ├── mental_health_model.pkl  # Trained pipeline (TF-IDF + LR)
│       ├── preprocessor_fn.pkl      # Stop-words list
│       ├── metrics.json             # Model performance metrics
│       ├── feature_importance.json  # XAI feature weights per class
│       └── label_info.json          # Label metadata for frontend
│
├── frontend/                        # React.js Frontend
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js                 # App entry point
│       ├── App.jsx                  # Main app + all pages
│       └── utils/
│           └── api.js               # Axios API client
│
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ and npm
- Git

---

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/mental-health-ai.git
cd mental-health-ai
```

### Step 2 — Set up the Python backend
```bash
cd backend
pip install -r requirements.txt
```

### Step 3 — Train the ML model
```bash
python model/train_model.py
```
This will produce:
- `model/mental_health_model.pkl`
- `model/metrics.json`
- `model/feature_importance.json`
- `model/label_info.json`

### Step 4 — Start the Flask server
```bash
python app.py
```
Flask will run on **http://localhost:5000**

### Step 5 — Set up and start the React frontend
```bash
cd ../frontend
npm install
npm start
```
React will run on **http://localhost:3000**

---

## 🔌 API Endpoints

| Method | Endpoint      | Description                               |
|--------|---------------|-------------------------------------------|
| POST   | `/predict`    | Predict mental state from text input      |
| POST   | `/chat`       | Get chatbot response for a given state    |
| GET    | `/history`    | Retrieve user's mood prediction history   |
| GET    | `/metrics`    | Model performance metrics                 |
| GET    | `/features`   | Feature importance (XAI) per label        |
| GET    | `/helplines`  | Emergency mental health helplines         |
| GET    | `/health`     | API health check                          |

### POST /predict — Example

**Request:**
```json
{
  "text": "I feel so empty and hopeless. Nothing brings me joy anymore.",
  "user_id": "user_abc123"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "label": "depressed",
    "confidence": 0.8912,
    "confidence_pct": "89.1%",
    "all_scores": {
      "depressed": 0.8912,
      "neutral": 0.0412,
      "anxious": 0.0312,
      "stressed": 0.0212,
      "happy": 0.0152
    },
    "description": "Signs of depression detected in your text",
    "color": "#818cf8"
  },
  "xai": {
    "word_impact": [
      { "word": "hopeless", "impact": 0.824, "positive": true },
      { "word": "empty", "impact": 0.712, "positive": true }
    ],
    "top_features": [
      { "word": "depression", "weight": 2.14 },
      { "word": "hopeless",   "weight": 1.98 }
    ]
  },
  "chatbot": {
    "message": "I hear you, and your feelings are completely valid...",
    "crisis": false,
    "resources": null
  }
}
```

---

## 🧠 ML Architecture

```
Raw Text
   ↓
Preprocessing (lowercase → strip URLs → remove punctuation → stopwords)
   ↓
TF-IDF Vectorizer
  • ngram_range = (1, 2)    ← Unigrams + Bigrams
  • max_features = 8000
  • sublinear_tf = True     ← log(TF) + 1
   ↓
Logistic Regression
  • C = 5.0  (regularisation)
  • solver = 'lbfgs'
  • max_iter = 2000
   ↓
Predicted Class + Probability Scores
   ↓
XAI: word impact via LR coefficients
```

---

## 💡 Explainable AI (XAI)

Each prediction is accompanied by word-level importance scores derived from the Logistic Regression coefficient matrix:

- `coef_[label_index][feature_index]` gives the weight of each TF-IDF feature for a specific class.
- Positive weight → word supports that class prediction.
- Displayed in the Result page as highlighted impact words.

---

## 🆘 Mental Health Resources

| Organisation         | Number              | Hours            |
|----------------------|---------------------|------------------|
| iCall (India)        | 9152987821          | Mon–Sat 8AM–10PM |
| Vandrevala Foundation| 1860-2662-345       | 24/7             |
| NIMHANS Helpline     | 080-46110007        | 24/7             |
| Crisis Text Line (US)| Text HOME to 741741 | 24/7             |

---

## ⚠️ Disclaimer

This tool is designed **exclusively for educational and research purposes**. It is **not** a substitute for professional mental health diagnosis, therapy, or emergency intervention. If you or someone you know is experiencing a mental health crisis, please contact a qualified healthcare professional or emergency services immediately.

---

## 📄 License

MIT License — Free for academic and educational use.

---

## 👥 Contributors

Built as an IEEE Third Year Engineering Project demonstrating:
- Natural Language Processing (NLP)
- Machine Learning Classification
- Explainable AI (XAI)
- Full-stack Web Development (React + Flask)
- RESTful API Design
- SQLite persistence

---

*MindSense AI — Bringing empathy and intelligence to mental health awareness.*
