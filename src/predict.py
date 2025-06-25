import pandas as pd
import joblib

import config as cfg
from utils import clean_text

df_test = pd.read_csv(cfg.DATA_DIR / "test.csv")
df_test["comment_text"] = df_test["comment_text"].apply(clean_text)

preds = {"id": df_test["id"]}

for label in cfg.LABELS:
    print(f"Predicting: {label}")
    vectorizer, model = joblib.load(cfg.MODEL_DIR / f"{label}_model.joblib")
    X_test_vec = vectorizer.transform(df_test["comment_text"])
    proba = model.predict_proba(X_test_vec)[:, 1]
    preds[label] = (proba >= 0.4).astype(int)

submission = pd.DataFrame(preds)
submission.to_csv(cfg.SUBMISSION_FILE, index=False)
print(f"Saved to {cfg.SUBMISSION_FILE}")
