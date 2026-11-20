"""Machine-learning part of IncidentIQ.

Two small models, both from scikit-learn:

1. Isolation Forest   -> looks at the 4 numbers (latency, error rate, CPU, memory) and says
                         "does this minute look unusual compared with normal behaviour?"
2. TF-IDF + Logistic Regression -> reads the log message and guesses the problem type
                         (database, network, application, deployment, resource).

Run `python -m app.ml` to generate data, train both models and save them in models/.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.data_gen import HOLDOUT_SEED, TRAIN_SEED, generate_dataset, save_csvs

FEATURES = ["latency_ms", "error_rate", "cpu_pct", "memory_pct"]

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
ANOMALY_FILE = MODEL_DIR / "anomaly_model.joblib"
CLASSIFIER_FILE = MODEL_DIR / "classifier.joblib"
INFO_FILE = MODEL_DIR / "model_info.json"


def clean_text(text):
    """Prepare a log line for TF-IDF: lowercase, drop trace ids, turn every number into 'num'.

    Numbers differ on every log line (8400ms, 6200ms ...) but carry no meaning for the
    problem type, so we replace them with one shared word.
    """
    text = str(text).lower()
    text = re.sub(r"trace_id=\w+", " ", text)
    text = re.sub(r"\d+(\.\d+)?", " num ", text)
    return text


def train_models(seed=TRAIN_SEED):
    """Generate data, train both models, test them on fresh data, and save everything."""
    MODEL_DIR.mkdir(exist_ok=True)
    save_csvs()  # also writes data/training_data.csv so students can open it
    train = generate_dataset(seed=seed)
    holdout = generate_dataset(seed=HOLDOUT_SEED)  # different seed = data the models have never seen

    # ---- Model 1: Isolation Forest, trained ONLY on normal rows ------------------------
    # The model learns what "normal" looks like. Anything far from that gets a high score.
    normal_rows = train[train["true_category"] == "normal"]
    anomaly_model = Pipeline([
        ("scale", StandardScaler()),  # put all four numbers on the same scale
        ("forest", IsolationForest(n_estimators=100, contamination=0.01, random_state=seed)),
    ])
    anomaly_model.fit(normal_rows[FEATURES])
    # score_samples returns "normality" (higher = more normal). We flip the sign so that
    # a HIGHER anomaly score means MORE unusual, which is easier to understand.
    score_threshold = float(-anomaly_model.named_steps["forest"].offset_)

    # Check on hold-out data: how many injected incident rows does it catch?
    hold_scores = -anomaly_model.score_samples(holdout[FEATURES])
    hold_pred = hold_scores > score_threshold
    hold_true = (holdout["true_category"] != "normal").to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        hold_true, hold_pred, average="binary", zero_division=0)

    # ---- Model 2: TF-IDF + Logistic Regression on log messages -------------------------
    # TF-IDF turns text into numbers (which words/word-pairs matter), and Logistic
    # Regression learns which words point to which category.
    incident_train = train[train["true_category"] != "normal"]
    classifier = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)),
        ("logreg", LogisticRegression(max_iter=1000, C=5.0)),
    ])
    classifier.fit(incident_train["message"], incident_train["true_category"])

    incident_hold = holdout[holdout["true_category"] != "normal"]
    predicted = classifier.predict(incident_hold["message"])
    report = classification_report(incident_hold["true_category"], predicted,
                                   output_dict=True, zero_division=0)

    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "features": FEATURES,
        "classes": [str(c) for c in classifier.classes_],
        "anomaly_model": {
            "algorithm": "Isolation Forest (scikit-learn)",
            "n_estimators": 100,
            "contamination": 0.01,
            "training_rows": int(len(normal_rows)),
            "score_threshold": round(score_threshold, 4),
            "baseline": {f: {"mean": round(float(normal_rows[f].mean()), 2),
                             "std": round(float(normal_rows[f].std()), 2)} for f in FEATURES},
            "holdout": {"rows": int(len(holdout)), "precision": round(float(precision), 3),
                        "recall": round(float(recall), 3), "f1": round(float(f1), 3)},
        },
        "classifier": {
            "algorithm": "TF-IDF + Logistic Regression (scikit-learn)",
            "training_messages": int(len(incident_train)),
            "holdout_messages": int(len(incident_hold)),
            "holdout_accuracy": round(float(accuracy_score(incident_hold["true_category"], predicted)), 3),
            "per_class": {c: {"precision": round(report[c]["precision"], 3),
                              "recall": round(report[c]["recall"], 3),
                              "support": int(report[c]["support"])}
                          for c in classifier.classes_},
        },
    }

    joblib.dump(anomaly_model, ANOMALY_FILE)
    joblib.dump(classifier, CLASSIFIER_FILE)
    INFO_FILE.write_text(json.dumps(info, indent=2))
    return info


def ensure_models():
    """Train the models if the saved files are missing (first start on a fresh machine)."""
    if not (ANOMALY_FILE.exists() and CLASSIFIER_FILE.exists() and INFO_FILE.exists()):
        train_models()


def load_models():
    """Load saved models from disk into one dictionary."""
    ensure_models()
    return {
        "anomaly": joblib.load(ANOMALY_FILE),
        "classifier": joblib.load(CLASSIFIER_FILE),
        "info": json.loads(INFO_FILE.read_text()),
    }


def detect_anomalies(bundle, df):
    """Return (anomaly_score, is_anomaly) for every row. Higher score = more unusual."""
    model = bundle["anomaly"]
    features = df[FEATURES]
    scores = -model.score_samples(features)
    flags = model.predict(features) == -1  # -1 means "outlier" in scikit-learn
    return scores, flags


def classify_messages(bundle, messages):
    """Return (class_names, probability_matrix) for a list of log messages."""
    model = bundle["classifier"]
    probabilities = model.predict_proba(list(messages))
    return np.array(model.classes_), probabilities


if __name__ == "__main__":
    # Import ourselves under the real module name. Otherwise joblib would save `clean_text`
    # as "__main__.clean_text" and the models could not be loaded by the web app later.
    from app.ml import train_models as _train

    print(json.dumps(_train(), indent=2))
