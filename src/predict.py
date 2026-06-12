import json
from pathlib import Path

import pandas as pd
import joblib

import config as cfg
from utils import clean_text

# load test
df_test = pd.read_csv(cfg.DATA_DIR / "test.csv")
df_test["comment_text"] = df_test["comment_text"].apply(clean_text)

# load thresholds if available, fallback to default 0.4
THR_PATH = Path("reports") / "thresholds.json"
if THR_PATH.exists():
    with open(THR_PATH, "r") as f:
        thr_data = json.load(f)
else:
    thr_data = {}

preds = {"id": df_test["id"]}

for label in cfg.LABELS:
    print(f"Predicting: {label}")
    vec_model_path = cfg.MODEL_DIR / f"{label}_model.joblib"
    if not vec_model_path.exists():
        print(f"Model not found: {vec_model_path}, filling zeros")
        preds[label] = [0] * len(df_test)
        continue

    vectorizer, model = joblib.load(vec_model_path)
    X_test_vec = vectorizer.transform(df_test["comment_text"])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_vec)[:, 1]
    else:
        # fallback to decision_function (may not be calibrated)
        proba = model.decision_function(X_test_vec)

    # choose threshold: from thresholds.json if present, else 0.4
    thr = 0.4
    if label in thr_data and isinstance(thr_data[label], dict) and "thr" in thr_data[label]:
        thr = float(thr_data[label]["thr"])

    preds[label] = (proba >= thr).astype(int)

submission = pd.DataFrame(preds)
submission.to_csv(cfg.SUBMISSION_FILE, index=False)
print(f"Saved to {cfg.SUBMISSION_FILE}")
