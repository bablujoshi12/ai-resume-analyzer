import os
import re
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "role_classifier.joblib")

ROLE_KEYWORD_WEIGHTS = {
    "Data Science": {
        "python": 2.0,
        "machine learning": 3.0,
        "ml": 2.5,
        "ai": 2.5,
        "deep learning": 2.5,
        "tensorflow": 2.0,
        "pytorch": 2.0,
        "pandas": 1.5,
        "numpy": 1.5,
        "statistics": 1.5,
    },
    "Web Development": {
        "html": 3.0,
        "css": 3.0,
        "javascript": 3.5,
        "react": 2.5,
        "node": 2.0,
        "nodejs": 2.0,
        "frontend": 2.0,
        "backend": 1.5,
        "django": 1.5,
        "flask": 1.2,
    },
    "Android Development": {
        "android": 3.5,
        "kotlin": 3.0,
        "java": 2.0,
        "xml": 1.5,
        "flutter": 2.5,
        "firebase": 1.5,
    },
    "IOS Development": {
        "ios": 3.5,
        "swift": 3.0,
        "xcode": 2.5,
        "cocoa": 2.0,
        "objective c": 2.5,
    },
    "UI-UX Development": {
        "figma": 3.0,
        "adobe xd": 2.5,
        "wireframe": 2.5,
        "prototype": 2.5,
        "ux": 2.0,
        "ui": 2.0,
        "user research": 2.0,
    },
}


def _synthetic_training_data():
    samples = [
        ("python pandas numpy machine learning deep learning tensorflow pytorch data analysis statistics", "Data Science"),
        ("data visualization sql jupyter sklearn regression classification feature engineering", "Data Science"),
        ("react javascript typescript html css frontend api node express web app", "Web Development"),
        ("django flask backend restful api mysql postgresql web development", "Web Development"),
        ("android kotlin java xml gradle android studio mobile app firebase", "Android Development"),
        ("flutter dart android app mobile development play store ui activity", "Android Development"),
        ("swift ios xcode cocoa touch ui kit storyboard ios app", "IOS Development"),
        ("objective c swift ui ios development app store xcode", "IOS Development"),
        ("figma adobe xd wireframe prototype user research usability design system", "UI-UX Development"),
        ("ux ui design thinking user journey visual design interaction design", "UI-UX Development"),
    ]
    return [x[0] for x in samples], [x[1] for x in samples]


def _clean_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\+#]", " ", text)
    tokens = [t for t in text.split() if t and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def _contains_all(text, terms):
    return all(term in text for term in terms)


def _manual_override_role(clean_text):
    # Rule 1: HTML + CSS + JavaScript => Web Development
    if _contains_all(clean_text, ["html", "css", "javascript"]):
        return "Web Development", 0.95

    # Rule 2: Python + ML/AI terms => Data Science
    if "python" in clean_text and (
        "machine learning" in clean_text
        or "ml" in clean_text.split()
        or "ai" in clean_text.split()
        or "deep learning" in clean_text
    ):
        return "Data Science", 0.95

    return None, None


def _apply_keyword_weighting(clean_text, class_scores):
    boosted = dict(class_scores)
    for role, kw_map in ROLE_KEYWORD_WEIGHTS.items():
        role_boost = 0.0
        for kw, weight in kw_map.items():
            if kw in clean_text:
                role_boost += weight
        if role in boosted:
            boosted[role] += role_boost * 0.03
    total = sum(boosted.values()) or 1.0
    normalized = {k: float(v / total) for k, v in boosted.items()}
    return normalized


def train_and_save_model(model_path=MODEL_PATH):
    X_train, y_train = _synthetic_training_data()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


def load_or_train_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return train_and_save_model(model_path=model_path)


def predict_job_role(text):
    model = load_or_train_model()
    clean_text = _clean_text(text)

    # Manual override for high-signal combinations
    override_role, override_conf = _manual_override_role(clean_text)
    if override_role:
        return override_role, override_conf, {override_role: override_conf}

    probs = model.predict_proba([clean_text])[0]
    classes = model.classes_
    class_scores = dict(zip(classes, [float(p) for p in probs]))

    # Hybrid: ML probability + weighted keyword boosting
    boosted_scores = _apply_keyword_weighting(clean_text, class_scores)
    prediction = max(boosted_scores, key=boosted_scores.get)
    confidence = float(boosted_scores[prediction])
    return prediction, confidence, boosted_scores
