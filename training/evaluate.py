# training/evaluate.py

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


@torch.no_grad()
def evaluate_model(model, dataloaders, split="val", device="cuda"):
    """
    Running a model on validation or test.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloaders : dict
        Dictionary with DataLoaders (train/val/test).
    split : str
        "val" or "test".
    device : str
        Device ("cuda" or "cpu").

    Returns
    -------
    probs : torch.Tensor [N, num_classes]
        Predicted probabilities.
    targets : torch.Tensor [N]
        True marks.
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []

    for xb, yb in dataloaders[split]:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)

        if isinstance(outputs, tuple):  # для Inception
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())
        all_targets.append(yb.cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return probs, targets


def compute_metrics(probs, targets, num_classes):
    """
    Calculates accuracy, precision, recall, specificity, f1 for each class and macro-averages.

    Parameters
    ----------
    probs : torch.Tensor [N, num_classes]
        Predicted probabilities.
    targets : torch.Tensor [N]
        True marks.
    num_classes : int
        Number of classes.

    Returns
    -------
    acc : float
        Accuracy across all classes.
    metrics : list[dict]
        Metrics for each class.
    macro : dict
        Macro-average metrics.
    """
   # preds = probs.argmax(dim=1).numpy()
   # targets_np = targets.numpy()

   # acc = (preds == targets_np).mean()

   # metrics = []

    preds = probs.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    metrics = []


    for c in range(num_classes):



      #  # binary labels for class c
      #  y_true = (targets_np == c).astype(int)
      #  y_pred = (preds == c).astype(int)
#
#        precision = precision_score(y_true, y_pred, zero_division=0)
 #       recall = recall_score(y_true, y_pred, zero_division=0)
  #      f1 = f1_score(y_true, y_pred, zero_division=0)
#
 #       # specificity = TN / (TN+FP)
  #      cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
   #     tn, fp, fn, tp = cm.ravel()
 #       specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        tn = ((preds != c) & (targets != c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append({
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1
        })

    # macro-average
    macro = {
        "precision": np.mean([m["precision"] for m in metrics]),
        "recall": np.mean([m["recall"] for m in metrics]),
        "specificity": np.mean([m["specificity"] for m in metrics]),
        "f1": np.mean([m["f1"] for m in metrics]),
    }

    return acc, metrics, macro
