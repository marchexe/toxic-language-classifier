# Toxic Language Classifier

A short guide to the project, analytics, and demo.

Project structure:
- `data/` — contains `train.csv` and `test.csv`.
- `models/` — saved `(vectorizer, model)` pairs in `joblib` files.
- `src/` — scripts: `train.py`, `predict.py`, `eval.py`, `threshold_tuning.py`, `generate_main_plots.py`, `app.py`.
- `reports/` — generated reports and plot images.

Quick start (recommended in a virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit
```

1) Run evaluation and generate ROC / PR / confusion matrix plots:

```bash
python3 src/eval.py
```

Output files: `reports/metrics_summary.csv` and `reports/figs/*.png`.

2) Tune thresholds (F1 optimization on validation) and generate threshold plots:

```bash
python3 src/threshold_tuning.py
```

Output files: `reports/thresholds.json`, `reports/thresholds_summary.csv`, and `reports/*_threshold.png`.

3) Generate two main summary plots for presentation:

```bash
python3 src/generate_main_plots.py
```

Output files: `reports/metrics_summary_main.png` and `reports/thresholds_summary_main.png`.

4) Run the local demo (Streamlit):

```bash
streamlit run src/app.py
```

The demo shows:
- two main summary plots,
- download buttons for `metrics_summary.csv` and `thresholds_summary.csv`,
- an input field for live text classification,
- probabilities and binary predictions using the tuned thresholds.

Suggested improvements for the future:
- Fine-tune a transformer model (e.g. `distilbert` / `roberta`) to improve precision on rare labels.
- Add stratified k-fold cross-validation, ensemble stacking, and probability calibration.
- Add explainability with SHAP or Integrated Gradients and perform fairness analysis.

If needed, I can also prepare a `Dockerfile` and `Makefile` for reproducibility.
