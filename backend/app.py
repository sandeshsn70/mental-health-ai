"""
=============================================================================
AI-Based Mental Health Prediction & Chat Support System
Flask Backend API
=============================================================================
IEEE Final Year Project
Endpoints:
  POST /predict     → NLP prediction + confidence + XAI word highlights
  POST /chat        → Empathetic chatbot response
  GET  /history     → User mood history
  POST /history     → Save prediction to history
  GET  /metrics     → ML model performance metrics
  GET  /features    → Feature importance (Explainable AI)
=============================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import re
import os
import sqlite3
import datetime
import random
import numpy as np

# ─── App Initialization ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all cross-origin requests (for React frontend)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
# SQLite path:
# - Locally: keeps the existing backend/mental_health.db location.
# - Render: can be overridden with the DB_PATH environment variable.
#   Example for a persistent disk: /var/data/mental_health.db
DB_PATH = os.getenv(
    'DB_PATH',
    os.path.join(BASE_DIR, 'mental_health.db')
)

# ─── Load ML Model Artifacts ─────────────────────────────────────────────────
def load_model():
    """Load trained pipeline and supplementary JSON assets."""
    pipeline = pickle.load(
        open(os.path.join(MODEL_DIR, 'mental_health_model.pkl'), 'rb')
    )
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_importance.json')) as f:
        feature_importance = json.load(f)
    with open(os.path.join(MODEL_DIR, 'label_info.json')) as f:
        label_info = json.load(f)
    return pipeline, metrics, feature_importance, label_info

pipeline, MODEL_METRICS, FEATURE_IMPORTANCE, LABEL_INFO = load_model()

# Stop-words used during training (same set – kept in sync)
STOP_WORDS = set([
    'i','me','my','we','our','you','your','he','him','his','she','her','it','its',
    'they','them','their','what','which','who','this','that','these','those','am',
    'is','are','was','were','be','been','being','have','has','had','having','do',
    'does','did','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','into','through','during','before',
    'after','to','from','up','down','in','out','on','off','over','under','again',
    'then','once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','own','same','so','than','too','very',
    'can','will','just','should','now','also','get','got','go','going','went',
    'today','day','time','feel'
]) - {'no', 'not', 'nor', 'never'}


# ─── Text Preprocessing ───────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Clean and normalize input text (mirrors training pipeline)."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ─── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    """Create SQLite tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT    NOT NULL DEFAULT 'anonymous',
            text      TEXT    NOT NULL,
            label     TEXT    NOT NULL,
            confidence REAL   NOT NULL,
            timestamp TEXT    NOT NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT NOT NULL DEFAULT 'anonymous',
            role      TEXT NOT NULL,
            message   TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

print(f"🧠 MindSense AI backend initialized")
print(f"📦 Model directory: {MODEL_DIR}")
print(f"🗄️ SQLite database: {DB_PATH}")


# ─── Chatbot Response Engine ──────────────────────────────────────────────────
CHATBOT_RESPONSES = {
    "depressed": [
        "I hear you, and I want you to know — your feelings are completely valid. Depression can make everything feel heavy and hopeless. You're not alone in this. 💙",
        "Thank you for sharing something so personal. What you're feeling is real, and reaching out takes courage. One small step at a time — that's all that's needed right now.",
        "It sounds like you're carrying a very heavy weight. Depression lies to us about our worth and our future. Please know: things CAN get better, even when it doesn't feel that way.",
        "I'm really glad you're talking about this. Your pain matters, and so do you. Would it help to talk about one small thing that might bring a moment of comfort today?",
        "Feeling this way is incredibly hard, and I want you to know I'm here. You deserve support — have you been able to connect with a counselor or trusted person recently?",
    ],
    "stressed": [
        "It sounds like you have so much on your plate right now. Stress like this is exhausting. Let's take a breath together — one deep breath in, and slowly out. 🌬️",
        "You're dealing with a lot! Feeling overwhelmed is a completely normal response to too many demands. What's the ONE most pressing thing right now? Let's focus on just that.",
        "Stress can feel all-consuming. Remember: you don't have to solve everything at once. Breaking tasks into smaller pieces can help make the mountain feel climbable.",
        "I can sense the pressure you're under. Try the 5-4-3-2-1 grounding technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "When stress piles up, our body goes into overdrive. Even 5 minutes of slow breathing or a brief walk can genuinely reset your nervous system. You've got this. 💪",
    ],
    "anxious": [
        "Anxiety can feel absolutely overwhelming — your heart races, your mind spirals. You're safe right now, in this moment. Take a slow breath with me. 🌿",
        "I understand anxiety can make the future feel terrifying. One technique that helps: ask yourself 'Is this thought a fact, or a fear?' Most worst-case scenarios never happen.",
        "What you're feeling is anxiety's alarm system going off — but the danger your mind perceives may not match reality. Ground yourself: feel your feet on the floor. You're here. You're safe.",
        "Anxiety lies to us about how bad things will be. Box breathing can help right now: breathe in 4 counts, hold 4, out 4, hold 4. Repeat until calmer.",
        "It's okay to feel anxious. These feelings, as uncomfortable as they are, will pass. You are stronger than your anxiety, even when it doesn't feel that way.",
    ],
    "happy": [
        "That's wonderful to hear! 😊 Your positive energy is beautiful. What's been the highlight of your day? I'd love to hear more!",
        "It's so great that you're feeling good! Savoring these positive moments is so important. What's been contributing to this good mood?",
        "Your happiness is contagious through the screen! 🌟 Keep nurturing the things that bring you joy — you deserve this.",
        "Wonderful! Happy moments are precious. Have you thought about journaling this feeling so you can revisit it on harder days?",
        "Love to hear this! Positive mental states are worth celebrating. Keep doing whatever you've been doing — it's clearly working! 🎉",
    ],
    "neutral": [
        "It's perfectly okay to have steady, calm days. Stability is actually a form of wellbeing. Is there anything on your mind you'd like to explore?",
        "A neutral mood is a baseline we can build from. Is there something small you could do today that brings you a spark of joy?",
        "Sometimes 'okay' is just right. How are you taking care of yourself these days? Even in neutral moments, self-care matters.",
        "Steady days have their own quiet value. Is there anything you'd like to reflect on or talk through while things are calm?",
        "Being in a neutral place is nothing to worry about. It can be a good time to check in with yourself — what do you need more of in your life right now?",
    ],
    "general": [
        "Thank you for sharing how you feel. I'm here to listen and support you. How are things going overall?",
        "I appreciate you opening up. Your mental wellbeing matters. What else would you like to talk about?",
        "Every feeling you have is valid. I'm here for you. What's been on your mind lately?",
        "It takes strength to reflect on your emotions. You're doing something important by paying attention to how you feel.",
    ],
    "crisis": [
        "🚨 I'm concerned about what you've shared. If you're having thoughts of harming yourself, please reach out immediately:\n\n📞 **iCall (India)**: 9152987821\n📞 **Vandrevala Foundation**: 1860-2662-345 (24/7)\n📞 **Crisis Text Line**: Text HOME to 741741\n\nYou matter, and help is available right now. ❤️",
    ]
}

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
    'harm myself', 'hurt myself', 'self harm', 'cut myself', 'no reason to live'
]

def get_chatbot_response(user_message: str, predicted_label: str) -> dict:
    """Generate an empathetic chatbot response with optional resources."""
    lower_msg = user_message.lower()

    # Check for crisis indicators first
    if any(kw in lower_msg for kw in CRISIS_KEYWORDS):
        response = CHATBOT_RESPONSES["crisis"][0]
        return {"message": response, "crisis": True, "resources": get_helplines()}

    # Select response pool based on prediction
    pool = CHATBOT_RESPONSES.get(predicted_label, CHATBOT_RESPONSES["general"])
    response = random.choice(pool)

    # Append resource suggestions for clinical states
    resources = None
    if predicted_label in ('depressed', 'anxious'):
        resources = get_helplines()

    return {
        "message": response,
        "crisis": False,
        "resources": resources
    }


def get_helplines() -> list:
    """Return emergency mental health helplines."""
    return [
        {"name": "iCall (India)", "number": "9152987821", "available": "Mon–Sat, 8AM–10PM"},
        {"name": "Vandrevala Foundation", "number": "1860-2662-345", "available": "24/7"},
        {"name": "NIMHANS Helpline", "number": "080-46110007", "available": "24/7"},
        {"name": "Crisis Text Line (US)", "number": "Text HOME to 741741", "available": "24/7"},
    ]


# ─── XAI: Word Impact Scoring ─────────────────────────────────────────────────
def get_word_impact(text: str, predicted_label: str) -> list:
    """
    Explainable AI: score each word's contribution to the prediction.
    Uses TF-IDF feature weights from the trained Logistic Regression.
    """
    tfidf = pipeline.named_steps['tfidf']
    clf   = pipeline.named_steps['clf']
    classes = list(clf.classes_)
    label_idx = classes.index(predicted_label) if predicted_label in classes else 0

    clean = preprocess_text(text)
    words = clean.split()
    vocab = tfidf.vocabulary_

    word_impacts = []
    for word in set(words):
        if word in vocab:
            feat_idx = vocab[word]
            weight = float(clf.coef_[label_idx][feat_idx])
            word_impacts.append({
                "word": word,
                "impact": round(weight, 4),
                "positive": weight > 0
            })

    # Sort by absolute impact
    word_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return word_impacts[:10]


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "loaded", "version": "1.0.0"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "text": "...", "user_id": "..." }
    Returns: prediction label, confidence scores, XAI word impacts, chatbot reply
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        raw_text = data.get('text', '').strip()
        user_id  = data.get('user_id', 'anonymous')

        if len(raw_text) < 5:
            return jsonify({"error": "Text too short. Please provide more detail."}), 400
        if len(raw_text) > 5000:
            return jsonify({"error": "Text too long. Please limit to 5000 characters."}), 400

        # Preprocess
        clean_text = preprocess_text(raw_text)
        if not clean_text.strip():
            clean_text = raw_text.lower()  # Fallback: use raw text

        # Predict
        proba = pipeline.predict_proba([clean_text])[0]
        classes = pipeline.classes_
        label = classes[int(np.argmax(proba))]
        confidence = float(np.max(proba))

        # All class probabilities
        all_scores = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, proba)
        }

        # XAI word impact
        word_impact = get_word_impact(raw_text, label)

        # Chatbot response
        chat_data = get_chatbot_response(raw_text, label)

        # Persist to DB
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            'INSERT INTO predictions (user_id, text, label, confidence, timestamp) VALUES (?,?,?,?,?)',
            (user_id, raw_text[:500], label, round(confidence, 4),
             datetime.datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "prediction": {
                "label": label,
                "confidence": round(confidence, 4),
                "confidence_pct": f"{confidence * 100:.1f}%",
                "all_scores": all_scores,
                "description": LABEL_INFO['label_descriptions'].get(label, ''),
                "color": LABEL_INFO['label_colors'].get(label, '#64748b'),
            },
            "xai": {
                "word_impact": word_impact,
                "top_features": FEATURE_IMPORTANCE.get(label, [])[:5]
            },
            "chatbot": chat_data,
            "metadata": {
                "text_length": len(raw_text),
                "clean_length": len(clean_text.split()),
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    POST /chat
    Body: { "message": "...", "label": "...", "user_id": "..." }
    Returns standalone chatbot reply (without running prediction).
    """
    try:
        data    = request.get_json()
        message = data.get('message', '').strip()
        label   = data.get('label', 'neutral')
        user_id = data.get('user_id', 'anonymous')

        if not message:
            return jsonify({"error": "Empty message"}), 400

        chat_data = get_chatbot_response(message, label)

        # Persist chat messages
        ts = datetime.datetime.utcnow().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            'INSERT INTO chat_messages (user_id, role, message, timestamp) VALUES (?,?,?,?)',
            (user_id, 'user', message[:1000], ts)
        )
        cur.execute(
            'INSERT INTO chat_messages (user_id, role, message, timestamp) VALUES (?,?,?,?)',
            (user_id, 'assistant', chat_data['message'][:1000], ts)
        )
        conn.commit()
        conn.close()

        return jsonify({"success": True, **chat_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """
    GET /history?user_id=anonymous&limit=50
    Returns the mood prediction history for a user.
    """
    try:
        user_id = request.args.get('user_id', 'anonymous')
        limit   = min(int(request.args.get('limit', 50)), 200)

        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            '''SELECT id, text, label, confidence, timestamp
               FROM predictions
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?''',
            (user_id, limit)
        )
        rows = cur.fetchall()
        conn.close()

        history = [
            {
                "id": r[0],
                "text": r[1][:100] + '...' if len(r[1]) > 100 else r[1],
                "label": r[2],
                "confidence": r[3],
                "timestamp": r[4],
                "color": LABEL_INFO['label_colors'].get(r[2], '#64748b')
            }
            for r in rows
        ]

        # Mood distribution summary
        label_counts = {}
        for h in history:
            label_counts[h['label']] = label_counts.get(h['label'], 0) + 1

        return jsonify({
            "success": True,
            "history": history,
            "total": len(history),
            "distribution": label_counts
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """GET /metrics — Return model performance statistics."""
    return jsonify({"success": True, "metrics": MODEL_METRICS})


@app.route('/features', methods=['GET'])
def get_features():
    """GET /features?label=depressed — Return top XAI feature words."""
    label = request.args.get('label')
    if label and label in FEATURE_IMPORTANCE:
        return jsonify({"success": True, "features": FEATURE_IMPORTANCE[label], "label": label})
    return jsonify({"success": True, "features": FEATURE_IMPORTANCE})


@app.route('/helplines', methods=['GET'])
def helplines():
    """GET /helplines — Return emergency mental health contacts."""
    return jsonify({"success": True, "helplines": get_helplines()})


# ─── Run ──────────────────────────────────────────────────────────────────────
# Render provides the PORT environment variable at runtime.
# Gunicorn imports this module as `app:app`, so this block is mainly for
# local development/testing.
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    print(f"🧠 Mental Health AI Backend starting on http://{host}:{port}")
    app.run(
        host=host,
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )
