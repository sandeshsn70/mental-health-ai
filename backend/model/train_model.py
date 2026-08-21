"""
=============================================================================
  AI-Based Mental Health Prediction System
  Model Training Script — IEEE Final Year Project
=============================================================================

  Architecture : TF-IDF Vectorizer + Logistic Regression
  Classes      : Depressed | Stressed | Anxious | Happy | Neutral
  Features     : Bigram TF-IDF (max 8000 features, sublinear_tf=True)
  Output       : mental_health_model.pkl, metrics.json,
                 feature_importance.json, label_info.json

  Run          : python model/train_model.py
=============================================================================
"""

import re
import os
import json
import pickle
import numpy  as np
import pandas as pd

from sklearn.model_selection        import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model           import LogisticRegression
from sklearn.pipeline               import Pipeline
from sklearn.metrics                import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. STOP-WORDS  (no external download needed)
# ─────────────────────────────────────────────────────────────────────────────
ENGLISH_STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves',
    'you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those',
    'am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as',
    'until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out',
    'on','off','over','under','again','further','then','once',
    'here','there','when','where','why','how',
    'all','both','each','few','more','most','other','some','such',
    'own','same','so','than','too','very',
    'can','will','just','should','now',
    'also','get','got','go','going','went','today','day','time','feel',
]) - {'no', 'not', 'nor', 'never', 'neither'}   # keep negation words


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────
DATASET = {
    "depressed": [
        "empty hopeless worthless nothing matters anymore depression crushing me",
        "suicidal thoughts dark place cannot see future no hope",
        "crying sobbing cannot stop tears sadness overwhelming despair",
        "cannot get out of bed depressed paralyzed by darkness",
        "isolating alone nobody understands my pain depression",
        "lost interest everything anhedonia no pleasure nothing enjoyable",
        "burden everyone dead inside numb empty shell depression",
        "exhausted fatigued depressed sleep all day escape reality",
        "worthless failure loser depression eating alive",
        "hopeless future bleak dark tunnel no light depression",
        "deep sadness grief loss mourning depressed heartbroken",
        "cannot function depression paralyzed helpless powerless",
        "self hatred despise myself ugly worthless depressed",
        "gave up dreams no ambition depressed defeated broken",
        "melancholy sorrow grief deep sadness depression weighing down",
        "disconnected dissociated empty robot depression numb feeling",
        "low mood depressed sad unhappy miserable every single day",
        "guilt shame regret depression haunting memories dark",
        "dark thoughts death depression no will to live anymore",
        "tearful weeping melancholy depression heavy heart sadness",
        "appetite gone weight lost depressed not eating anything",
        "concentration poor foggy brain depressed cannot think clearly",
        "morning dread depression waking up terrible every day",
        "abandoned rejected unloved depression alone nobody cares",
        "emotional suffering depression unbearable torture pain",
        "past trauma haunting depression flashbacks suffering daily",
        "future looks dark bleak hopeless depressed no point",
        "dragging myself depression heavy weight invisible burden",
        "cannot enjoy anything pleasureless depressed flat numb empty",
        "defeated broken crushed depression spirit shattered gone",
        "depression relapse falling back darkness again suffering",
        "zombie walking dead depression no energy no motivation",
        "silently suffering depression hiding pain behind fake smile",
        "depressed cannot find reason smile happiness completely gone",
        "stuck bottom pit depression cannot climb out helpless",
        "worthless nothing depression self loathing pain",
        "crying myself to sleep depression every night hopeless",
        "lost all hope depressed no light at end tunnel",
        "invisible sadness depression weighs me down",
        "life has no meaning depression emptiness hollowness inside",
    ],
    "stressed": [
        "deadline tomorrow project incomplete panicking stressed overwhelmed",
        "workload impossible overwhelming stressed burnout exhausted pressure",
        "boss demanding impossible targets stressed overworked frustrated",
        "exam pressure stress studying all night cramming panicking",
        "financial stress bills debt money problems worried stressed",
        "too many responsibilities juggling work family stressed breaking",
        "time running out stressed cannot complete everything pressure",
        "overwhelmed tasks pile mounting stress no end visible",
        "pressure performance stressed anxiety work demands crushing",
        "multitasking chaos stressed mind scattered cannot focus",
        "stressed irritable snapping losing temper constant pressure",
        "tense muscles headache stressed physical symptoms workplace",
        "cannot relax stressed even holidays mind racing about work",
        "work life balance destroyed stressed family suffering neglected",
        "perfectionism stress never good enough pressure on myself",
        "corporate pressure KPI targets stressed performance review",
        "assignment due tomorrow havent started stressed procrastinated",
        "rent bills salary insufficient stressed financial pressure",
        "relationship stress arguments fighting partner stressed exhausted",
        "parenting stress children demands responsibilities overwhelming",
        "stressed sleep deprivation exhaustion work demands piling",
        "presentations public speaking performance anxiety stress",
        "job insecurity stressed losing employment financial worry",
        "competitive environment stressed colleagues pressure succeed",
        "stressed upcoming interview nervous preparation pressure",
        "project deadline crunch stressed team problems delivery",
        "overwhelmed new responsibilities promotion stress impossible",
        "stressed caregiver sick parent responsibilities heavy burden",
        "academic pressure grades performance stressed university exams",
        "burnout complete exhaustion stressed work no motivation",
        "too much on plate stressed cannot cope anymore",
        "running on empty stressed no time for self care",
        "pressure from all sides stressed mind body breaking",
        "urgent deadlines everywhere stressed cannot prioritize",
        "constant demands stressed never enough hours in day",
        "grinding non stop stressed no rest no recovery",
        "overwhelmed by demands stressed cannot see the way out",
        "deadline pressure crushing stressed cannot think clearly",
        "work overload stressed sacrificing sleep health relationships",
        "so much pressure stressed feel like going to break",
    ],
    "anxious": [
        "panic attack heart racing dizzy cannot breathe anxiety",
        "worried catastrophe disaster something terrible happen anxiety",
        "social anxiety meeting people embarrass myself scared",
        "generalized anxiety constant worry everything going wrong",
        "health anxiety convinced dying illness symptoms",
        "fear uncertainty future unknown terrifying anxiety",
        "intrusive thoughts obsessive worry anxiety cannot stop",
        "phobia fear triggers avoiding situations anxiety",
        "nervous trembling shaking anxiety before important event",
        "overthinking replay conversations worry anxious endlessly",
        "anticipatory anxiety dreading upcoming event terrified scared",
        "separation anxiety leaving home scared being alone",
        "performance anxiety stage fright presenting terrified",
        "sleep anxiety lying awake worried cannot fall asleep",
        "stomach knots butterflies anxiety nausea constant worry",
        "constant dread impending doom anxiety irrational fear",
        "second guessing decisions anxiety indecision afraid mistakes",
        "rejection sensitivity anxiety afraid disappointment failure",
        "existential anxiety meaning purpose life terrifying uncertainty",
        "medical anxiety results tests waiting period anxious worried",
        "travel anxiety flying heights new places scared terrified",
        "abandonment anxiety afraid losing people relationships",
        "financial anxiety poverty losing everything terrified",
        "anxious mind racing thoughts cannot quiet the noise",
        "hypervigilant always alert scanning danger anxiety",
        "worst case scenarios anxiety constantly planning disaster",
        "anxiety attack symptoms physical racing heart sweating trembling",
        "worry spiral anxiety cannot stop catastrophizing everything",
        "fear failure anxiety paralyzed cannot take any action",
        "social situations anxiety sweating blushing nervous mess",
        "nervous wreck anxiety constant tension unease daily",
        "catastrophic thinking anxiety blown out of proportion",
        "anxiety exhausting constant vigilance draining energy",
        "what if thoughts anxiety spiral cannot escape",
        "scared of everything anxious leaving house avoidance",
        "always waiting for bad news anxious mind restless",
        "anxiety making me physically sick stomach pain nausea",
        "frozen by fear anxiety cannot make decisions",
        "anxious about everything small things feel catastrophic",
        "constant background anxiety never fully relaxed",
    ],
    "happy": [
        "joyful wonderful amazing blessed grateful happiness today",
        "celebrating success achievement proud happy excited milestone",
        "love connection belonging happy fulfilling relationships",
        "peaceful content serene calm happy satisfied with life",
        "excited positive enthusiastic optimistic happy about future",
        "grateful thankful appreciative blessed happy with my life",
        "accomplished goal dream come true happy proud achievement",
        "laughing fun enjoying happy carefree cheerful today",
        "energized motivated passionate happy inspired to create",
        "love partner spouse happiness relationship thriving beautiful",
        "vacation travel adventure excitement happy exploring world",
        "creative flow inspired artistic happy fulfillment satisfaction",
        "helping others purpose meaning happy giving back community",
        "nature walk sunshine beautiful happy refreshed restored",
        "music dancing celebration joy happiness pure bliss",
        "promotion achievement career success happy proud milestone",
        "family gathering warmth love happiness together bonding",
        "healthy body exercise endorphins happy fit feeling great",
        "learning growing improving happy making progress daily",
        "friendship bonding connection laughter happy memories",
        "cooking delicious meal satisfaction happiness joy",
        "reading book relaxation enjoyment happy peaceful afternoon",
        "morning sunshine coffee calm happy fresh start today",
        "winning competition victory happy proud elated thrilled",
        "reconciliation forgiveness peace happy resolved conflict",
        "meditation mindfulness peace happiness inner calm found",
        "grateful for small things happiness found in moments",
        "surrounded by love family friends happy truly blessed",
        "dreams coming true happiness working hard paying off",
        "things going well optimistic happy positive mindset",
        "laughter joy today reminded me life is beautiful",
        "deeply satisfied contented happy everything working out",
        "life is good happy thriving flourishing",
        "positive energy happiness radiating grateful for today",
        "joyful heart happy grateful wonderful day",
        "excitement happiness anticipation looking forward to life",
        "smiling genuinely happy heart is full",
        "everything wonderful happy feeling on top of world",
        "grateful for life happy loving every moment",
        "pure joy happiness bliss contentment today",
    ],
    "neutral": [
        "average day nothing special happened routine tasks completed",
        "went work came home dinner watched television normal day",
        "ordinary morning coffee commute meetings lunch afternoon",
        "nothing remarkable today stable mood regular activities",
        "completed checklist tasks unremarkable steady pace",
        "moderate day neither particularly good nor bad",
        "flat mood okay stable not excited not sad",
        "routine grocery shopping cooking cleaning normal chores",
        "usual commute standard lunch break ordinary afternoon",
        "mild weather nothing eventful today steady comfortable",
        "normal conversation friend brief chat nothing special",
        "watched movie after dinner relaxed nothing unusual",
        "standard gym workout regular exercise routine done",
        "cooked usual meal nothing extraordinary average evening",
        "neutral indifferent neither positive negative today",
        "went through motions day automatic pilot routine",
        "regular appointment nothing concerning standard visit",
        "average sleep neither rested nor tired normal",
        "steady consistent mood baseline normal functioning",
        "maintenance mode surviving neutral okay",
        "paying bills managing finances ordinary adult tasks",
        "replying messages normal social interaction nothing special",
        "plant watered laundry done dishes clean routine chores",
        "phone scrolling reading news nothing exciting neutral",
        "walked dog short walk nothing unusual ordinary",
        "standard video call meeting routine work interaction",
        "bought groceries cooked meal ate slept routine",
        "nothing bothered neutral comfortable baseline state",
        "mild satisfaction completing tasks nothing extraordinary",
        "stable predictable consistent normal neutral day",
        "same as yesterday nothing changed neutral stable",
        "work done tasks complete nothing exciting neutral mood",
        "just another regular Tuesday nothing to report",
        "meh feeling okay fine nothing special neutral",
        "baseline mood no highs no lows just steady",
        "regular day regular tasks regular mood nothing notable",
        "uneventful comfortable neutral day as expected",
        "life proceeding normally neutral undisturbed",
        "ordinary routine nothing to celebrate nothing to worry",
        "just existing today neither happy nor sad neutral",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.stop_words = ENGLISH_STOPWORDS

    def clean(self, text: str) -> str:
        text   = text.lower()
        text   = re.sub(r'http\S+|www\S+', '', text)
        text   = re.sub(r'[^a-z\s]',       ' ', text)
        tokens = [t for t in text.split()
                  if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range  = (1, 2),
            max_features = 8000,
            sublinear_tf = True,
            min_df       = 1,
        )),
        ('clf', LogisticRegression(
            C            = 5.0,
            max_iter     = 2000,
            solver       = 'lbfgs',
            random_state = 42,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score   (y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score       (y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"  ACCURACY  : {acc*100:.2f}%")
    print(f"  PRECISION : {prec*100:.2f}%")
    print(f"  RECALL    : {rec*100:.2f}%")
    print(f"  F1 SCORE  : {f1*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy"  : round(float(acc),  4),
        "precision" : round(float(prec), 4),
        "recall"    : round(float(rec),  4),
        "f1_score"  : round(float(f1),   4),
        "classes"   : list(pipeline.classes_),
        "test_size" : int(len(X_test)),
        "train_size": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE  (Explainable AI)
# ─────────────────────────────────────────────────────────────────────────────
def compute_feature_importance(pipeline: Pipeline) -> dict:
    tfidf         = pipeline.named_steps['tfidf']
    clf           = pipeline.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    importance    = {}

    for idx, cls in enumerate(clf.classes_):
        coefs   = clf.coef_[idx]
        top_idx = np.argsort(coefs)[-15:][::-1]
        importance[cls] = [
            {"word": feature_names[i], "weight": round(float(coefs[i]), 4)}
            for i in top_idx
        ]
        top5 = [importance[cls][k]['word'] for k in range(5)]
        print(f"  [{cls.upper():<10}] top-5: {top5}")

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(pipeline, preprocessor, metrics, feature_imp, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'mental_health_model.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

    with open(os.path.join(save_dir, 'preprocessor_fn.pkl'), 'wb') as f:
        pickle.dump({'stop_words': list(preprocessor.stop_words)}, f)

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(save_dir, 'feature_importance.json'), 'w') as f:
        json.dump(feature_imp, f, indent=2)

    label_info = {
        "classes": list(pipeline.classes_),
        "label_descriptions": {
            "depressed": "Signs of depression detected in your text",
            "stressed" : "High stress levels detected",
            "anxious"  : "Anxiety indicators found in your text",
            "happy"    : "Positive emotional state detected",
            "neutral"  : "Neutral emotional state",
        },
        "label_colors": {
            "depressed": "#818cf8",
            "stressed" : "#fb923c",
            "anxious"  : "#fbbf24",
            "happy"    : "#4ade80",
            "neutral"  : "#94a3b8",
        },
        "label_emoji": {
            "depressed": "😔",
            "stressed" : "😤",
            "anxious"  : "😰",
            "happy"    : "😊",
            "neutral"  : "😐",
        },
    }
    with open(os.path.join(save_dir, 'label_info.json'), 'w') as f:
        json.dump(label_info, f, indent=2)

    print("\n  Artifacts saved:")
    for fn in ['mental_health_model.pkl','preprocessor_fn.pkl',
               'metrics.json','feature_importance.json','label_info.json']:
        print(f"    ✅  {fn}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  🧠  Mental Health AI — Model Training Pipeline")
    print("="*60)

    save_dir     = os.path.dirname(os.path.abspath(__file__))
    preprocessor = TextPreprocessor()

    # Build DataFrame
    rows = []
    for label, samples in DATASET.items():
        for text in samples:
            rows.append({'raw': text, 'clean': preprocessor.clean(text), 'label': label})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nDataset: {len(df)} samples\n{df['label'].value_counts().to_string()}")

    X = df['clean']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # Train
    pipe = build_pipeline()
    print("\n🔄 Training …")
    pipe.fit(X_train, y_train)

    # Cross-validation
    cv = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(f"✅ 5-Fold CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

    # Evaluate
    metrics              = evaluate(pipe, X_test, y_test)
    metrics['train_size'] = int(len(X_train))
    metrics['cv_mean']    = round(float(cv.mean()), 4)
    metrics['cv_std']     = round(float(cv.std()),  4)

    # Feature importance
    print("\n🔍 Feature importance …")
    feature_imp = compute_feature_importance(pipe)

    # Smoke test
    tests = [
        ("I feel so hopeless and empty, nothing matters anymore", "depressed"),
        ("Too many deadlines, I'm completely overwhelmed by work", "stressed"),
        ("I keep having panic attacks and worrying about everything", "anxious"),
        ("Feeling wonderful today, truly grateful for everything",  "happy"),
        ("Nothing special today, just a regular Tuesday routine",   "neutral"),
    ]
    print("\n🧪 Smoke test:")
    for txt, exp in tests:
        clean  = preprocessor.clean(txt)
        pred   = pipe.predict([clean])[0]
        conf   = pipe.predict_proba([clean]).max()
        mark   = "✅" if pred == exp else "⚠️ "
        print(f"  {mark} [{exp:<10} → {pred:<10}] {conf*100:.1f}%")

    # Save
    print("\n💾 Saving …")
    save_artifacts(pipe, preprocessor, metrics, feature_imp, save_dir)
    print(f"\n🎉 Done!  Model saved to: {save_dir}/\n")


if __name__ == "__main__":
    main()
