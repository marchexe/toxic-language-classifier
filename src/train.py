import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import config as cfg
from utils import clean_text

cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(cfg.DATA_DIR / "train.csv")
df["comment_text"] = df["comment_text"].apply(clean_text)

for label in cfg.LABELS:
    print(f"\nTraining model for: {label}")

    X_train, X_val, y_train, y_val = train_test_split(
        df["comment_text"], df[label], test_size=0.1, random_state=42, stratify=df[label])

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    val_preds = model.predict(X_val_vec)
    print(f"F1 for {label}: {f1_score(y_val, val_preds):.4f}")

    joblib.dump((vectorizer, model), cfg.MODEL_DIR / f"{label}_model.joblib")
