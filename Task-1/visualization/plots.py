import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_training_curves(history: Dict[str, list], model_name: str) -> None:
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{model_name} Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{model_name.lower()}_training_curves.png", dpi=200)
    plt.close(fig)


def save_model_comparison(results: Dict[str, Dict[str, float]]) -> None:
    if not results:
        return

    model_names = list(results.keys())
    metrics = ["accuracy", "f1_score_macro", "balanced_accuracy"]
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, metric in enumerate(metrics):
        values = [results[name].get(metric, 0.0) for name in model_names]
        ax.bar(x + (idx - 1) * width, values, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
