import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg
from utils import clean_text

MODEL_DIR = cfg.MODEL_DIR
LABELS = cfg.LABELS

# load thresholds if present
THR_PATH = Path("reports") / "thresholds.json"
if THR_PATH.exists():
    import json
    with open(THR_PATH, "r") as f:
        THRESHOLDS = {k: float(v.get("thr", 0.4)) for k, v in json.load(f).items()}
else:
    THRESHOLDS = {l: 0.4 for l in LABELS}

@st.cache_data
def load_models():
    models = {}
    for label in LABELS:
        path = MODEL_DIR / f"{label}_model.joblib"
        if path.exists():
            vec, model = joblib.load(path)
            models[label] = (vec, model)
    return models

models = load_models()

st.title("Toxic Comment Demo")
text = st.text_area("Enter comment to classify", value="I love this project!")

# show main summary plots if present
REPORTS_DIR = Path("reports")
MAIN_METRICS = REPORTS_DIR / "metrics_summary_main.png"
MAIN_THRESH = REPORTS_DIR / "thresholds_summary_main.png"
if MAIN_METRICS.exists():
    st.image(str(MAIN_METRICS), caption="Metrics summary (F1 / PR AUC / ROC AUC)")
if MAIN_THRESH.exists():
    st.image(str(MAIN_THRESH), caption="Optimal thresholds per label")

# allow downloading CSVs
metrics_csv = REPORTS_DIR / "metrics_summary.csv"
thresholds_csv = REPORTS_DIR / "thresholds_summary.csv"
if metrics_csv.exists():
    st.download_button("Download metrics CSV", data=open(metrics_csv, "rb"), file_name="metrics_summary.csv")
if thresholds_csv.exists():
    st.download_button("Download thresholds CSV", data=open(thresholds_csv, "rb"), file_name="thresholds_summary.csv")

if st.button("Analyze"):
    txt = clean_text(text)
    results = {}
    for label, (vec, model) in models.items():
        x = vec.transform([txt])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            p = float(proba[0, 1])
        else:
            score = model.decision_function(x)
            p = float(score[0]) if hasattr(score, "shape") else float(score)
        results[label] = p

    df = pd.DataFrame(list(results.items()), columns=["label", "probability"]).sort_values("probability", ascending=False)
    st.bar_chart(df.set_index("label").probability)

    st.write("Probabilities:")
    st.table(df)

    # binary predictions using thresholds
    thresh_rows = []
    for label, prob in results.items():
        thr = THRESHOLDS.get(label, 0.4)
        thresh_rows.append({"label": label, "probability": prob, "threshold": thr, "pred": int(prob >= thr)})
    st.subheader("Predictions (thresholded)")
    st.table(pd.DataFrame(thresh_rows).sort_values("probability", ascending=False))

    # show top features if linear
    for label, (vec, model) in models.items():
        if hasattr(model, "coef_"):
            st.subheader(f"Top features for {label}")
            x = vec.transform([txt])
            coef = model.coef_.ravel()
            feat_names = vec.get_feature_names_out()
            # contribution = coef * x
            contrib = x.toarray().ravel() * coef
            top_idx = np.argsort(contrib)[-10:][::-1]
            feats = [(feat_names[i], float(contrib[i])) for i in top_idx if contrib[i] != 0]
            if feats:
                st.table(pd.DataFrame(feats, columns=["feature", "contribution"]))

st.write("Models loaded:", list(models.keys()))
