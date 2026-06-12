import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

import config as cfg
from utils import clean_text

OUT = Path("reports")
OUT.mkdir(exist_ok=True)


def find_best_thresholds():
    df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    df["comment_text"] = df["comment_text"].apply(clean_text)

    thresholds = {}
    summary_rows = []

    for label in cfg.LABELS:
        print(f"Tuning {label}")
        X_train, X_val, y_train, y_val = train_test_split(
            df["comment_text"], df[label], test_size=0.1, random_state=42, stratify=df[label]
        )

        vec_model_path = cfg.MODEL_DIR / f"{label}_model.joblib"
        if not vec_model_path.exists():
            print("Model not found", vec_model_path)
            continue

        vectorizer, model = joblib.load(vec_model_path)
        X_val_vec = vectorizer.transform(X_val)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val_vec)[:, 1]
        else:
            proba = model.decision_function(X_val_vec)

        best = {"thr": 0.5, "f1": 0.0}
        ths = np.linspace(0.01, 0.99, 99)
        f1s = []
        for t in ths:
            preds = (proba >= t).astype(int)
            f = f1_score(y_val, preds, zero_division=0)
            f1s.append(f)
            if f > best["f1"]:
                best = {"thr": float(t), "f1": float(f)}

        thresholds[label] = best
        summary_rows.append({"label": label, "best_threshold": best["thr"], "best_f1": best["f1"]})

        # plot
        plt.figure()
        plt.plot(ths, f1s)
        plt.xlabel("threshold")
        plt.ylabel("f1")
        plt.title(f"Threshold vs F1 — {label}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT / f"{label}_threshold.png")
        plt.close()

    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    pd.DataFrame(summary_rows).to_csv(OUT / "thresholds_summary.csv", index=False)
    print("Saved thresholds and plots to reports/")


if __name__ == "__main__":
    find_best_thresholds()
