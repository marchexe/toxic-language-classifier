import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("reports")
OUT.mkdir(exist_ok=True)


def plot_metrics_summary():
    mpath = OUT / "metrics_summary.csv"
    if not mpath.exists():
        print("metrics_summary.csv not found in reports/")
        return
    df = pd.read_csv(mpath)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    x = df['label']
    # bar width
    w = 0.25
    idx = range(len(x))

    plt.bar([i - w for i in idx], df['f1'], width=w, label='F1')
    plt.bar(idx, df['pr_auc'], width=w, label='PR AUC')
    plt.bar([i + w for i in idx], df['roc_auc'], width=w, label='ROC AUC')

    plt.xticks(idx, x, rotation=45)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title('Per-label summary: F1 / PR AUC / ROC AUC')
    plt.tight_layout()
    plt.savefig(OUT / 'metrics_summary_main.png')
    plt.close()


def plot_thresholds_summary():
    tpath = OUT / 'thresholds_summary.csv'
    if not tpath.exists():
        print('thresholds_summary.csv not found in reports/')
        return
    td = pd.read_csv(tpath)

    plt.figure(figsize=(8, 4))
    sns.barplot(x='label', y='best_threshold', data=td)
    plt.ylim(0, 1)
    plt.title('Optimal threshold per label')
    plt.tight_layout()
    plt.savefig(OUT / 'thresholds_summary_main.png')
    plt.close()


def main():
    plot_metrics_summary()
    plot_thresholds_summary()
    print('Saved main plots to reports/')


if __name__ == '__main__':
    main()
