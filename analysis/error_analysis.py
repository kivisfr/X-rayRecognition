# analysis/error_analysis.py

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_misclassified_images(images, targets, preds, classes, model_name, split, out_dir, dls):
    """
    Saves examples of incorrect classifications in separate folders.

    Parameters
    ----------
        images : torch.Tensor [B,C,H,W]
            Batch of images.
        targets : torch.Tensor [B]
           True class labels.
        preds : torch.Tensor [B]
           Predicted class labels.
        classes : list[str]
           List of class names.
        model_name : str
           Model name (e.g., "ResNeXt50").
        split : str
           Dataset partition ("val" or "test").
        out_dir : Path
            Folder to save results.
    """
    out_dir = Path(out_dir) / f"errors_{model_name}_{split}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    misclassified_idx = (preds != targets).nonzero(as_tuple=True)[0].cpu().numpy()

    for idx in misclassified_idx:
        true_label = classes[targets[idx].item()]
        pred_label = classes[preds[idx].item()]

        # создаём подпапку для пары (true → pred)
        pair_dir = out_dir / f"{true_label}_as_{pred_label}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        img_path, _ = images[idx]
        img = plt.imread(img_path)
        img = (img - img.min()) / (img.max() - img.min())

        plt.imsave(pair_dir / f"sample_{idx}.png", img)


def plot_misclassification_summary(targets, preds, classes, model_name, split, out_dir):
    """
    Builds a heatmap showing which classes are most frequently confused.
    """
    out_dir = Path(out_dir)
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Reds")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"Ошибки классификации: {model_name} ({split})")

    # signatures inside the cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(out_dir / f"errors_summary_{model_name}_{split}.png")
    plt.close(fig)
