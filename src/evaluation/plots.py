"""Plotting helpers for evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, save_path: str | Path, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    save_path = Path(save_path)
    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, save_path: str | Path, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save_path = Path(save_path)
    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: dict, save_path: str | Path, title: str = "Training Curves"):
    """Works for both Keras history.history dicts and the PyTorch history dict we build."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    loss_keys = [k for k in history if "loss" in k]
    for k in loss_keys:
        axes[0].plot(history[k], label=k)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    acc_keys = [k for k in history if "acc" in k or "accuracy" in k]
    for k in acc_keys:
        axes[1].plot(history[k], label=k)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    save_path = Path(save_path)
    _ensure_dir(save_path)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
