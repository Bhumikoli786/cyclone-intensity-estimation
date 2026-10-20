"""Matplotlib figures saved as PNG files (headless backend, safe on servers and Colab)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_loss_curves(history: dict, path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (standardised wind units)")
    ax.set_title("Training and validation loss"); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_scatter(y_true, y_pred, path) -> str:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lim = [0, max(float(y_true.max()), float(y_pred.max())) * 1.05]
    ax.plot(lim, lim, "r--", label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Best-track wind speed (kt)"); ax.set_ylabel("Predicted wind speed (kt)")
    ax.set_title("Predicted vs observed (test set)"); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_category_mae(per_category: dict, path) -> str:
    names = list(per_category)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(names)), [per_category[n]["mae_kt"] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(n={per_category[n]['n']})" for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE (kt)"); ax.set_title("Error by true intensity category"); ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_confusion(cm, labels, path) -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)
