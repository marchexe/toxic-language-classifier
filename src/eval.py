import pandas as pd
import joblib
import os
from pathlib import Path

import config as cfg
from utils import clean_text

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


OUT_DIR = Path("reports")
FIG_DIR = OUT_DIR / "figs"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def eval_label(df, label):
    print(f"Evaluating {label}...")
    X_train, X_val, y_train, y_val = train_test_split(
        df["comment_text"], df[label], test_size=0.1, random_state=42, stratify=df[label]
    )

    vec_model_path = cfg.MODEL_DIR / f"{label}_model.joblib"
    if not vec_model_path.exists():
        print(f"Model not found: {vec_model_path}")
        return None

    vectorizer, model = joblib.load(vec_model_path)

    X_val_vec = vectorizer.transform(X_val)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val_vec)[:, 1]
    else:
        proba = model.decision_function(X_val_vec)

    preds = (proba >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
        "accuracy": accuracy_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, proba) if len(set(y_val)) > 1 else None,
        "pr_auc": average_precision_score(y_val, proba) if len(set(y_val)) > 1 else None,
    }

    # ROC curve
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_val, proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC={metrics['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve — {label}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{label}_roc.png")
        plt.close()

    # PR curve
    if metrics["pr_auc"] is not None:
        prec, rec, _ = precision_recall_curve(y_val, proba)
        plt.figure()
        plt.plot(rec, prec, label=f"PR AUC={metrics['pr_auc']:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall — {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{label}_pr.png")
        plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{label}_cm.png")
    plt.close()

    # Top features (for linear models with coef_)
    top_feats = None
    try:
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            feat_names = vectorizer.get_feature_names_out()
            top_pos_idx = coefs.argsort()[::-1][:20]
            top_neg_idx = coefs.argsort()[:20]
            top_feats = {
                "top_positive": [(feat_names[i], float(coefs[i])) for i in top_pos_idx],
                "top_negative": [(feat_names[i], float(coefs[i])) for i in top_neg_idx],
            }
    except Exception:
        top_feats = None

    return {"label": label, "metrics": metrics, "top_features": top_feats}


def main():
    df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    df["comment_text"] = df["comment_text"].apply(clean_text)

    results = []
    for label in cfg.LABELS:
        res = eval_label(df, label)
        if res:
            results.append(res)

    # Write summary CSV
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({"label": r["label"], **m})

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
        print(f"Saved metrics to {OUT_DIR / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
