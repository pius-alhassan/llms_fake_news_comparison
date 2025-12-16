import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------ CONFIG ------------------

EVAL_ROOT = Path("data/outputs/video_evaluations")
LATEST_ONLY = True  # analyze most recent evaluation run


# ------------------ LOAD RESULTS ------------------

def get_latest_eval_dir():
    evals = sorted(EVAL_ROOT.glob("eval_*"))
    if not evals:
        raise RuntimeError("No evaluation folders found")
    return evals[-1]


def load_results(eval_dir):
    return pd.read_csv(eval_dir / "results.csv")


# ------------------ METRICS ------------------

def compute_metrics(df):
    metrics = {}

    for model in df["model"].unique():
        subset = df[df["model"] == model]

        y_true = subset["true_label"]
        y_pred = subset["predicted_label"]

        metrics[model] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label="fake"),
            "recall": recall_score(y_true, y_pred, pos_label="fake"),
            "f1": f1_score(y_true, y_pred, pos_label="fake"),
            "avg_latency_ms": subset["latency_ms"].mean()
        }

    return metrics


# ------------------ PLOTS ------------------

def plot_accuracy(metrics, out_dir):
    models = list(metrics.keys())
    acc = [metrics[m]["accuracy"] for m in models]

    plt.figure()
    plt.bar(models, acc)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per LLM")
    plt.savefig(out_dir / "accuracy_per_model.png")
    plt.close()


def plot_latency(metrics, out_dir):
    models = list(metrics.keys())
    latency = [metrics[m]["avg_latency_ms"] for m in models]

    plt.figure()
    plt.bar(models, latency)
    plt.ylabel("Latency (ms)")
    plt.title("Average Latency per LLM")
    plt.savefig(out_dir / "latency_per_model.png")
    plt.close()


def plot_accuracy_by_type(df, out_dir):
    grouped = (
        df.groupby(["model", "video_type"])["correct"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=grouped, x="video_type", y="correct", hue="model")
    plt.xticks(rotation=30)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Deepfake Type")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_by_video_type.png")
    plt.close()


def plot_confusion(df, model, out_dir):
    subset = df[df["model"] == model]
    cm = confusion_matrix(
        subset["true_label"],
        subset["predicted_label"],
        labels=["real", "fake"]
    )

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["real", "fake"],
                yticklabels=["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model}")
    plt.savefig(out_dir / f"confusion_matrix_{model}.png")
    plt.close()


# ------------------ MAIN ------------------

def main():
    eval_dir = get_latest_eval_dir()
    df = load_results(eval_dir)

    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics = compute_metrics(df)

    # Save metrics
    with open(eval_dir / "summary_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_accuracy(metrics, plots_dir)
    plot_latency(metrics, plots_dir)
    plot_accuracy_by_type(df, plots_dir)

    for model in df["model"].unique():
        plot_confusion(df, model, plots_dir)

    print(f"Analysis complete for: {eval_dir.name}")


if __name__ == "__main__":
    main()
