# logging_utils/plots.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

# ============================
# Learning curves
# ============================

def plot_training_curves(history, model_name, out_dir="plots"):
    """
    Plots loss and accuracy by epoch.
    history: dict with keys "train_loss", "val_loss", "train_acc", "val_acc"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_training_curves.png")
    plt.close()


def plot_metric_dynamics(history, model_name, out_dir="plots"):
    """
    Plots F1, Precision, and Recall dynamics by epoch.
    history: dict with keys "train_f1", "val_f1", "train_precision", "val_precision", "train_recall", "val_recall"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_f1"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_f1"], label="Train F1")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.plot(epochs, history["train_precision"], label="Train Precision")
    plt.plot(epochs, history["val_precision"], label="Val Precision")
    plt.plot(epochs, history["train_recall"], label="Train Recall")
    plt.plot(epochs, history["val_recall"], label="Val Recall")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name} Metrics Dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_metrics_dynamics.png")
    plt.close()


# ============================
# Confusion Matrix
# ============================

def plot_confusion_matrix(y_true, y_pred, classes, model_name, split, normalize=False, out_dir="plots"):
    """
    Constructs a confusion matrix.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} Confusion ({split}) {'(normalized)' if normalize else ''}")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else cm[i, j],
                    ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_confusion_{split}{'_norm' if normalize else ''}.png")
    plt.close()


# ============================
# ROC Curve
# ============================

def plot_roc(y_true, y_probs, classes, model_name, split, out_dir="plots"):
    """
    Plots ROC curves for each class.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_roc_{split}.png")
    plt.close()


# ============================
# Precision-Recall Curve
# ============================

def plot_pr_curves(y_true, y_probs, classes, model_name, split, out_dir="plots"):
    """
    Plots Precision-Recall curves for each class.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        plt.plot(recall, precision, label=f"{cname}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} PR Curves ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_pr_{split}.png")
    plt.close()


# ============================
# Calibration Curve
# ============================

def plot_calibration(y_true, y_probs, classes, model_name, split, out_dir="plots", n_bins=10):
    """
    Constructs calibration curves (how well the model's probabilities match reality).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(classes):
        prob_true, prob_pred = calibration_curve(y_true == i, y_probs[:, i], n_bins=n_bins)
        plt.plot(prob_pred, prob_true, marker="o", label=cname)

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title(f"{model_name} Calibration ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_calibration_{split}.png")
    plt.close()
