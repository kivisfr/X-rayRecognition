# train_covid_ensemble_with_checks.py
import os
import time
import csv
import shutil
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image, UnidentifiedImageError

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import json
import csv
import pandas as pd

from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

# ------------------ Config ------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE = 48
DEFAULT_NUM_WORKERS = 4
PIN_MEMORY = False
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = CHECKPOINT_DIR / "training_log.txt"
DATASET_REPORT = CHECKPOINT_DIR / "report_dataset.csv"
EPOCHS_CSV = CHECKPOINT_DIR / "epochs.csv"

STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 10
LR_STAGE1 = 1e-6
LR_STAGE2 = 1e-7
AUX_LOSS_WEIGHT = 0.4
FOCAL_GAMMA = 2.0

SMALL_RUN = False  # True -> quick smoke-test
DO_DATASET_CHECK = False  # run dataset pre-scan before training
AUTO_RESAVE_BAD = False  # if True, attempt to re-save corrupt images (may help)

PLOTS_DIR = CHECKPOINT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------ Utilities ------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def log(s: str, to_file: bool = True):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {s}"
    print(line)
    if to_file:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# CSV logging for epochs


def init_epochs_csv(path: Path = EPOCHS_CSV):
    header = ["timestamp","model","stage","local_epoch","lr","train_loss","train_acc","val_acc","f1","epoch_time_s","ckpt_path"]
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

def append_epoch_csv(row: List, path: Path = EPOCHS_CSV):
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def append_epoch_xlsx(row: dict, path: Path):
    """
    Append one epoch's metrics into an Excel file.
    row: dict with keys matching header
    """
    if path.exists():
        df = pd.read_excel(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_excel(path, index=False)


def append_metrics_xlsx(metrics: List[Dict[str, float]],
                        classes: List[str],
                        model_name: str,
                        split: str,
                        path_xlsx: Path,
                        path_csv: Path):
    rows = []
    for cname, m in zip(classes, metrics):
        rows.append({
            "model": model_name,
            "split": split,
            "class": cname,
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"]
        })
    df_new = pd.DataFrame(rows)

    if path_csv.exists():
        df_old = pd.read_csv(path_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path_csv, index=False)

    if path_xlsx.exists():
        df_old = pd.read_excel(path_xlsx)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_excel(path_xlsx, index=False)


# ------------------ Dataset scanner ------------------
def check_dataset(root: str, scenario: str, report_path: Path = DATASET_REPORT, auto_resave: bool = AUTO_RESAVE_BAD):
    """
    Scans all images under root/scenario/{train,val,test} and writes report CSV:
    filepath,class,split,status,error_msg,width,height,mode
    If auto_resave True, attempts to open and re-save images successfully (may fix minor issues).
    """
    base = Path(root) / scenario
    if not base.exists():
        raise RuntimeError(f"Dataset path {base} does not exist")

    rows = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            log(f"Warning: split directory missing: {split_dir}")
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for img_path in sorted(class_dir.rglob("*")):
                if not img_path.is_file():
                    continue
                try:
                    with Image.open(img_path) as im:
                        mode = im.mode
                        size = im.size
                        # try verify (some formats require reopen after verify)
                        im.verify()
                    status = "ok"
                    error_msg = ""
                    # optionally try to re-open & resave as RGB JPEG to fix header issues
                    if auto_resave:
                        try:
                            with Image.open(img_path) as im:
                                rgb = im.convert("RGB")
                                tmp = img_path.with_suffix(img_path.suffix + ".tmp")
                                rgb.save(tmp, format="JPEG", quality=90)
                                tmp.replace(img_path)
                                status += ";resaved"
                        except Exception as e:
                            status = "warn"
                            error_msg = f"resave_failed:{e}"
                except Exception as e:
                    mode = ""
                    size = ("", "")
                    status = "error"
                    error_msg = str(e)
                    log(f"Dataset check: failed image {img_path} | {e}")
                rows.append({
                    "filepath": str(img_path),
                    "class": class_dir.name,
                    "split": split,
                    "status": status,
                    "error_msg": error_msg,
                    "width": size[0] if isinstance(size, tuple) else size,
                    "height": size[1] if isinstance(size, tuple) else "",
                    "mode": mode
                })

    # write CSV
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filepath","class","split","status","error_msg","width","height","mode"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log(f"Dataset scan finished. Report saved to {report_path}")
    # return summary
    total = len(rows)
    errors = sum(1 for r in rows if r["status"].startswith("error"))
    warns = sum(1 for r in rows if r["status"].startswith("warn"))
    oks = total - errors - warns
    log(f"Dataset summary: total={total} ok={oks} errors={errors} warns={warns}")
    return {"total": total, "ok": oks, "errors": errors, "warns": warns}


# ------------------ Focal Loss ------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_factor = (1 - pt) ** self.gamma
        ce = -(targets_one_hot * log_probs).sum(dim=1)
        if self.weight is not None:
            w = self.weight[targets]
            loss = w * focal_factor * ce
        else:
            loss = focal_factor * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ------------------ Safe ImageFolder ------------------
class SafeImageFolder(datasets.ImageFolder):
    def loader(self, path: str):
        with open(path, "rb") as f:
            try:
                img = Image.open(f)
                return img.convert("RGB")
            except UnidentifiedImageError as e:
                raise IOError(f"Cannot identify image file {path}: {e}")

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            log(f"Warning: failed to load image {path}. Returning zeros. Error: {e}")
            # best-effort zero tensor size inference
            default_size = (3, 224, 224)
            if self.transform is not None:
                # attempt to infer numeric size from transforms
                t = self.transform
                if hasattr(t, "transforms"):
                    for tr in reversed(t.transforms):
                        if hasattr(tr, "size"):
                            sz = tr.size
                            if isinstance(sz, int):
                                default_size = (3, sz, sz)
                            elif isinstance(sz, tuple):
                                default_size = (3, sz[0], sz[1])
                            break
            return torch.zeros(default_size, dtype=torch.float32), target


# ------------------ Transforms / Dataloaders ------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(img_size: int, is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def build_dataloaders(root: str, scenario: str, batch_size: int = DEFAULT_BATCH_SIZE,
                      num_workers: int = DEFAULT_NUM_WORKERS, pin_memory: bool = PIN_MEMORY):
    base = Path(root) / scenario
    if not base.exists():
        raise RuntimeError(f"Dataset path {base} does not exist: {base}")

    tf_train_224 = build_transforms(224, True)
    tf_val_224 = build_transforms(224, False)
    train_224 = SafeImageFolder(base / "train", transform=tf_train_224)
    val_224 = SafeImageFolder(base / "val", transform=tf_val_224)
    test_224 = SafeImageFolder(base / "test", transform=tf_val_224)

    tf_train_299 = build_transforms(299, True)
    tf_val_299 = build_transforms(299, False)
    train_299 = SafeImageFolder(base / "train", transform=tf_train_299)
    val_299 = SafeImageFolder(base / "val", transform=tf_val_299)
    test_299 = SafeImageFolder(base / "test", transform=tf_val_299)

    dl_train_224 = DataLoader(train_224, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    dl_val_224 = DataLoader(val_224, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    dl_test_224 = DataLoader(test_224, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    dl_train_299 = DataLoader(train_299, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    dl_val_299 = DataLoader(val_299, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    dl_test_299 = DataLoader(test_299, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    classes = train_224.classes
    if classes != train_299.classes:
        log("Warning: class orders differ between 224 and 299 pipelines")

    return {
        "classes": classes,
        "224": {"train": dl_train_224, "val": dl_val_224, "test": dl_test_224},
        "299": {"train": dl_train_299, "val": dl_val_299, "test": dl_test_299},
    }


# ------------------ Models ------------------
class DenseNet161WithDropout(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.base = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        feats = self.base(x)
        return self.classifier(feats)

class ResNeXt50WithDropout(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.base = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        feats = self.base(x)
        return self.classifier(feats)

class InceptionV3Head(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.base = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)
        if hasattr(self.base, "AuxLogits") and self.base.AuxLogits is not None:
            in_aux = self.base.AuxLogits.fc.in_features
            self.base.AuxLogits.fc = nn.Linear(in_aux, num_classes)
    def forward(self, x):
        return self.base(x)


# ------------------ Optim/LR ------------------
def build_optimizer(model: nn.Module, lr: float = LR_STAGE1):
    return torch.optim.Adam(model.parameters(), lr=lr)

def adjust_lr(optimizer, new_lr: float):
    for g in optimizer.param_groups:
        g['lr'] = new_lr


# ------------------ Checkpointing ------------------
def save_checkpoint(state: dict, filename: Path):
    torch.save(state, str(filename))
    log(f"Saved checkpoint: {filename}")

def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], path: Path):
    ckpt = torch.load(str(path), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val = ckpt.get('best_val', None)
    log(f"Loaded checkpoint {path}. Resuming from epoch {start_epoch}")
    return start_epoch, best_val


# ------------------ Train / Eval loops ------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    aux_loss_weight: float = AUX_LOSS_WEIGHT) -> Tuple[float, float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    start = time.time()
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            main_logits, aux_logits = outputs
            loss_main = criterion(main_logits, targets)
            loss_aux = criterion(aux_logits, targets)
            loss = loss_main + aux_loss_weight * loss_aux
            logits_for_acc = main_logits
        else:
            loss = criterion(outputs, targets)
            logits_for_acc = outputs
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        preds = logits_for_acc.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += bs
    epoch_time = time.time() - start
    avg_loss = total_loss / total_samples if total_samples>0 else 0.0
    acc = total_correct / total_samples if total_samples>0 else 0.0
    return avg_loss, acc, epoch_time

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    total_correct, total = 0, 0
    all_probs, all_targets = [], []
    start = time.time()
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        if isinstance(outputs, tuple):
            main_logits, _ = outputs
            logits = main_logits
        else:
            logits = outputs
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total += images.size(0)
        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
    eval_time = time.time() - start
    acc = total_correct / total if total>0 else 0.0
    probs = torch.cat(all_probs, dim=0) if len(all_probs)>0 else torch.empty(0)
    targets = torch.cat(all_targets, dim=0) if len(all_targets)>0 else torch.empty(0, dtype=torch.long)
    return acc, probs, targets, eval_time


# ------------------ Graphics ------------------
def plot_training_curves(history, model_name="model"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()

    fname = PLOTS_DIR / f"{model_name}_training_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved training curves to {fname}")


def plot_confusion_matrix(y_true, y_pred, classes, model_name="model", split="val", normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} {split} Confusion Matrix" + (" (normalized)" if normalize else ""))

    fname = PLOTS_DIR / f"{model_name}_{split}_confusion{'_norm' if normalize else ''}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved confusion matrix to {fname}")


def plot_roc(y_true, y_probs, classes, model_name="model", split="val"):
    """
       Save ROC curves with proper class names in legend.
    """
    num_classes = len(classes)
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} {split} ROC Curves")
    plt.legend()
    fname = PLOTS_DIR / f"{model_name}_{split}_roc.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved ROC curves to {fname}")


def plot_pr_curves(y_true, y_probs, classes, model_name="model", split="val"):
    plt.figure(figsize=(8,6))
    for i, cname in enumerate(classes):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_probs[:, i])
        plt.plot(recall, precision, label=f"{cname}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} {split} Precision-Recall Curves")
    plt.legend()
    fname = PLOTS_DIR / f"{model_name}_{split}_pr.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved PR curves to {fname}")


def plot_calibration(y_true, y_probs, classes, model_name="model", split="val"):
    plt.figure(figsize=(8,6))
    for i, cname in enumerate(classes):
        prob_true, prob_pred = calibration_curve((y_true == i).astype(int), y_probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=cname)
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{model_name} {split} Calibration Curves")
    plt.legend()
    fname = PLOTS_DIR / f"{model_name}_{split}_calibration.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved calibration curves to {fname}")


def plot_metric_dynamics(history, model_name="model"):
    """
    Save line plots for precision, recall, F1 dynamics over epochs.
    Requires that history dict contains val_acc and optionally val_macro_f1.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12,5))
    # Accuracy
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Dynamics")
    plt.legend()
    fname = PLOTS_DIR / f"{model_name}_accuracy_dynamics.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved accuracy dynamics to {fname}")


def save_misclassified_images(images, y_true, y_pred, classes, model_name="model", split="val", max_samples=20):
    errors = (y_true != y_pred)
    idxs = np.where(errors)[0][:max_samples]
    for i in idxs:
        img = images[i].permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"True: {classes[y_true[i]]}, Pred: {classes[y_pred[i]]}")
        fname = PLOTS_DIR / f"{model_name}_{split}_error_{i}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()


def ensemble_diversity(preds_list: List[np.ndarray], classes: List[str], split="val"):
    """
    Compute disagreement between models.
    preds_list: list of predictions from each model (numpy arrays)
    """
    n_models = len(preds_list)
    disagreements = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            disagree = np.mean(preds_list[i] != preds_list[j])
            disagreements.append((i,j,disagree))
            log(f"Disagreement between model {i} and {j} on {split}: {disagree:.3f}")
    return disagreements


# ------------------ Metrics ------------------
def compute_metrics(probs: torch.Tensor, targets: torch.Tensor, num_classes: int):
    preds = probs.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    metrics = []
    for c in range(num_classes):
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

        macro = {
            "precision": np.mean([m["precision"] for m in metrics]),
            "recall": np.mean([m["recall"] for m in metrics]),
            "specificity": np.mean([m["specificity"] for m in metrics]),
            "f1": np.mean([m["f1"] for m in metrics])
        }

    return acc, metrics, macro


# ------------------ Training wrapper ------------------
def train_model_full(model_name: str, model: nn.Module, dataloaders: Dict[str, DataLoader],
                     num_classes: int, resume_path: Optional[Path] = None, save_every_epoch: bool = True):
    log("="*60)
    log(f"Start training model: {model_name}")
    model = model.to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = build_optimizer(model, lr=LR_STAGE1)
    start_epoch = 1
    best_val = None
    if resume_path:
        if resume_path.exists():
            start_epoch, best_val = load_checkpoint(model, optimizer, resume_path)
        else:
            log(f"Resume path {resume_path} not found. Starting from scratch.")

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    model_start = time.time()
    # Stage1
    log(f"{model_name} STAGE1: lr={LR_STAGE1} epochs={STAGE1_EPOCHS}")
    adjust_lr(optimizer, LR_STAGE1)
    epoch_ptr = start_epoch
    init_epochs_csv()
    for e in range(1, STAGE1_EPOCHS + 1):
        if SMALL_RUN and e > 1:
            break
        log(f"[{model_name}] STAGE1 epoch {epoch_ptr}/{STAGE1_EPOCHS}")
        train_loss, train_acc, epoch_time = train_one_epoch(model, dataloaders["train"], criterion, optimizer)
        val_acc, _, _, eval_time = evaluate(model, dataloaders["val"])
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        ckpt_path = CHECKPOINT_DIR / f"{model_name}_epoch{epoch_ptr}.pth"
        if save_every_epoch:
            save_checkpoint({'epoch': epoch_ptr, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'best_val': best_val}, ckpt_path)

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "stage": "stage1",
            "epoch": epoch_ptr,
            "lr": LR_STAGE1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": "",
            "epoch_time_s": epoch_time,
            "ckpt_path": str(ckpt_path)
        }
        append_epoch_xlsx(row, CHECKPOINT_DIR / "epochs.xlsx")

        # append CSV
        append_epoch_csv([time.strftime("%Y-%m-%d %H:%M:%S"), model_name, "stage1", epoch_ptr, LR_STAGE1,
                          f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_acc:.6f}", "", f"{epoch_time:.2f}", str(ckpt_path)])
        log(f"[{model_name}] STAGE1 epoch {epoch_ptr} done. train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s eval_time={eval_time:.1f}s")
        epoch_ptr += 1

    # Stage2
    log(f"{model_name} STAGE2: lr={LR_STAGE2} epochs={STAGE2_EPOCHS}")
    adjust_lr(optimizer, LR_STAGE2)
    epoch_ptr = start_epoch
    for e in range(1, STAGE2_EPOCHS + 1):
        if SMALL_RUN and e > 1:
            break
        log(f"[{model_name}] STAGE2 epoch {epoch_ptr}/{STAGE2_EPOCHS}")
        train_loss, train_acc, epoch_time = train_one_epoch(model, dataloaders["train"], criterion, optimizer)
        val_acc, _, _, eval_time = evaluate(model, dataloaders["val"])
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        ckpt_path = CHECKPOINT_DIR / f"{model_name}_epoch{epoch_ptr}.pth"
        if save_every_epoch:
            save_checkpoint({'epoch': epoch_ptr, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'best_val': best_val}, ckpt_path)

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "stage": "stage2",
            "epoch": epoch_ptr,
            "lr": LR_STAGE2,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": "",
            "epoch_time_s": epoch_time,
            "ckpt_path": str(ckpt_path)
        }
        append_epoch_xlsx(row, CHECKPOINT_DIR / "epochs.xlsx")

        append_epoch_csv([time.strftime("%Y-%m-%d %H:%M:%S"), model_name, "stage2", epoch_ptr, LR_STAGE2,
                          f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_acc:.6f}", "", f"{epoch_time:.2f}", str(ckpt_path)])
        log(f"[{model_name}] STAGE2 epoch {epoch_ptr} done. train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s eval_time={eval_time:.1f}s")
        epoch_ptr += 1

    total_time = time.time() - model_start
    log(f"Finished training {model_name}. Total time {total_time/60:.2f} minutes")
    return {"model": model, "history": history, "total_time_sec": total_time}


# ------------------ Ensemble predict ------------------
@torch.no_grad()
def ensemble_predict(models: List[nn.Module], dataloaders: Dict, num_classes: int, split: str = "test"):
    probs_list = []
    targets_ref = None
    order = [("resnext","224"), ("densenet","224"), ("inception","299")]
    for idx, (name, pipeline) in enumerate(order):
        mdl = models[idx]
        acc, probs, targets, eval_time = evaluate(mdl, dataloaders[pipeline][split])
        log(f"[Ensemble] {name} {split} acc={acc:.4f} eval_time={eval_time:.1f}s")
        probs_list.append(probs)
        if targets_ref is None:
            targets_ref = targets
        else:
            if targets_ref.size(0) != targets.size(0):
                log(f"Warning: target size mismatch between models for split={split} ({targets_ref.size(0)} vs {targets.size(0)})")
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return avg_probs, targets_ref


# ----------------------------------------------------
def evaluate_model(model: nn.Module,
                   dls_for_size: Dict[str, DataLoader],
                   num_classes: int,
                   split: str = "val") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model on the given dataloader split and collect probabilities + targets.
    model: trained torch model
    dls_for_size: dictionary with dataloaders for a given input size (e.g. dls["224"])
    num_classes: number of classes
    split: "val" or "test"
    Returns:
        probs: torch.Tensor [N, num_classes]
        targets: torch.Tensor [N]
    """
    model.eval()
    all_probs, all_targets = [], []
    dl = dls_for_size[split]

    with torch.no_grad():
        for images, targets in dl:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return probs, targets


# ------------------ Run experiment ------------------
def run_experiment(data_root: str,
                   scenario: str = "three-class",
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   num_workers: int = DEFAULT_NUM_WORKERS,
                   resume: Optional[Dict[str, Path]] = None):
    """
    Main experiment runner:
    - optional dataset pre-scan
    - train three models (ResNeXt, DenseNet, Inception)
    - save training curves + metric dynamics
    - evaluate each model individually (val/test): confusion (raw+normalized), ROC, PR, calibration
    - save per-class metrics into CSV/XLSX
    - evaluate ensemble (val/test) with the same plots and metrics
    - error analysis: save misclassified samples (small subset)
    - ensemble diversity metrics (disagreement between models)
    """

    log(f"Experiment start. data_root={data_root} scenario={scenario} batch_size={batch_size} num_workers={num_workers}")

    # --- Dataset pre-check ---
    if DO_DATASET_CHECK:
        log("Running dataset check before training...")
        check_dataset(data_root, scenario, report_path=DATASET_REPORT, auto_resave=AUTO_RESAVE_BAD)

    # --- Build dataloaders ---
    dls = build_dataloaders(data_root, scenario, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=PIN_MEMORY)
    classes = dls["classes"]
    num_classes = len(classes)
    log(f"Classes ({num_classes}): {classes}")

    # --- Initialize models ---
    resnext = ResNeXt50WithDropout(num_classes=num_classes, dropout_p=0.3)
    densenet = DenseNet161WithDropout(num_classes=num_classes, dropout_p=0.3)
    inception = InceptionV3Head(num_classes=num_classes)

    trained = {}

    # Convenience paths for structured metrics
    metrics_xlsx = CHECKPOINT_DIR / "metrics.xlsx"
    metrics_csv = CHECKPOINT_DIR / "metrics.csv"

    # Helper to fetch a small batch for error visualization
    def get_split_batch(dls, split: str, input_key: str, max_samples: int = 16):
        dl = dls[input_key][split]
        images_list, targets_list = [], []
        for imgs, targs in dl:
            images_list.append(imgs)
            targets_list.append(targs)
            if sum(x.size(0) for x in images_list) >= max_samples:
                break
        images = torch.cat(images_list, dim=0)[:max_samples]
        targets = torch.cat(targets_list, dim=0)[:max_samples]
        return images, targets

    # ============== Train & Evaluate ResNeXt ==============
    resnext_resume = resume.get("resnext") if resume else None
    resnext_res = train_model_full("resnext50", resnext, dls["224"], num_classes,
                                   resume_path=resnext_resume)
    trained["resnext"] = resnext_res["model"]
    plot_training_curves(resnext_res["history"], "ResNeXt50")
    plot_metric_dynamics(resnext_res["history"], "ResNeXt50")

    # Validation
    val_probs_r, val_targets_r = evaluate_model(trained["resnext"], dls["224"], num_classes, split="val")
    val_preds_r = val_probs_r.argmax(dim=1).numpy()
    plot_confusion_matrix(val_targets_r.numpy(), val_preds_r, classes, "ResNeXt50", "val", normalize=False)
    plot_confusion_matrix(val_targets_r.numpy(), val_preds_r, classes, "ResNeXt50", "val", normalize=True)
    plot_roc(val_targets_r.numpy(), val_probs_r.numpy(), classes, "ResNeXt50", "val")
    plot_pr_curves(val_targets_r.numpy(), val_probs_r.numpy(), classes, "ResNeXt50", "val")
    plot_calibration(val_targets_r.numpy(), val_probs_r.numpy(), classes, "ResNeXt50", "val")
    val_acc_r, val_metrics_r, val_macro_r = compute_metrics(val_probs_r, val_targets_r, num_classes)
    append_metrics_xlsx(val_metrics_r, classes, "ResNeXt50", "val", metrics_xlsx, metrics_csv)

    # Test
    test_probs_r, test_targets_r = evaluate_model(trained["resnext"], dls["224"], num_classes, split="test")
    test_preds_r = test_probs_r.argmax(dim=1).numpy()
    plot_confusion_matrix(test_targets_r.numpy(), test_preds_r, classes, "ResNeXt50", "test", normalize=False)
    plot_confusion_matrix(test_targets_r.numpy(), test_preds_r, classes, "ResNeXt50", "test", normalize=True)
    plot_roc(test_targets_r.numpy(), test_probs_r.numpy(), classes, "ResNeXt50", "test")
    plot_pr_curves(test_targets_r.numpy(), test_probs_r.numpy(), classes, "ResNeXt50", "test")
    plot_calibration(test_targets_r.numpy(), test_probs_r.numpy(), classes, "ResNeXt50", "test")
    test_acc_r, test_metrics_r, test_macro_r = compute_metrics(test_probs_r, test_targets_r, num_classes)
    append_metrics_xlsx(test_metrics_r, classes, "ResNeXt50", "test", metrics_xlsx, metrics_csv)

    # Error analysis (small batch, visualization only)
    try:
        imgs_val_224, targs_val_224 = get_split_batch(dls, "val", "224", max_samples=16)
        with torch.no_grad():
            logits_small = trained["resnext"](imgs_val_224.to(DEVICE))
            preds_small = logits_small.argmax(dim=1).cpu().numpy()
        save_misclassified_images(imgs_val_224.cpu(), targs_val_224.cpu().numpy(), preds_small,
                                  classes, model_name="ResNeXt50", split="val", max_samples=16)
    except Exception as e:
        log(f"ResNeXt50 error analysis skipped: {e}")

    # ============== Train & Evaluate DenseNet ==============
    densenet_resume = resume.get("densenet") if resume else None
    densenet_res = train_model_full("densenet161", densenet, dls["224"], num_classes,
                                    resume_path=densenet_resume)
    trained["densenet"] = densenet_res["model"]
    plot_training_curves(densenet_res["history"], "DenseNet161")
    plot_metric_dynamics(densenet_res["history"], "DenseNet161")

    # Validation
    val_probs_d, val_targets_d = evaluate_model(trained["densenet"], dls["224"], num_classes, split="val")
    val_preds_d = val_probs_d.argmax(dim=1).numpy()
    plot_confusion_matrix(val_targets_d.numpy(), val_preds_d, classes, "DenseNet161", "val", normalize=False)
    plot_confusion_matrix(val_targets_d.numpy(), val_preds_d, classes, "DenseNet161", "val", normalize=True)
    plot_roc(val_targets_d.numpy(), val_probs_d.numpy(), classes, "DenseNet161", "val")
    plot_pr_curves(val_targets_d.numpy(), val_probs_d.numpy(), classes, "DenseNet161", "val")
    plot_calibration(val_targets_d.numpy(), val_probs_d.numpy(), classes, "DenseNet161", "val")
    val_acc_d, val_metrics_d, val_macro_d = compute_metrics(val_probs_d, val_targets_d, num_classes)
    append_metrics_xlsx(val_metrics_d, classes, "DenseNet161", "val", metrics_xlsx, metrics_csv)

    # Test
    test_probs_d, test_targets_d = evaluate_model(trained["densenet"], dls["224"], num_classes, split="test")
    test_preds_d = test_probs_d.argmax(dim=1).numpy()
    plot_confusion_matrix(test_targets_d.numpy(), test_preds_d, classes, "DenseNet161", "test", normalize=False)
    plot_confusion_matrix(test_targets_d.numpy(), test_preds_d, classes, "DenseNet161", "test", normalize=True)
    plot_roc(test_targets_d.numpy(), test_probs_d.numpy(), classes, "DenseNet161", "test")
    plot_pr_curves(test_targets_d.numpy(), test_probs_d.numpy(), classes, "DenseNet161", "test")
    plot_calibration(test_targets_d.numpy(), test_probs_d.numpy(), classes, "DenseNet161", "test")
    test_acc_d, test_metrics_d, test_macro_d = compute_metrics(test_probs_d, test_targets_d, num_classes)
    append_metrics_xlsx(test_metrics_d, classes, "DenseNet161", "test", metrics_xlsx, metrics_csv)

    # Error analysis
    try:
        imgs_val_224, targs_val_224 = get_split_batch(dls, "val", "224", max_samples=16)
        with torch.no_grad():
            logits_small = trained["densenet"](imgs_val_224.to(DEVICE))
            preds_small = logits_small.argmax(dim=1).cpu().numpy()
        save_misclassified_images(imgs_val_224.cpu(), targs_val_224.cpu().numpy(), preds_small,
                                  classes, model_name="DenseNet161", split="val", max_samples=16)
    except Exception as e:
        log(f"DenseNet161 error analysis skipped: {e}")

    # ============== Train & Evaluate Inception ==============
    inception_resume = resume.get("inception") if resume else None
    inception_res = train_model_full("inception_v3", inception, dls["299"], num_classes,
                                     resume_path=inception_resume)
    trained["inception"] = inception_res["model"]
    plot_training_curves(inception_res["history"], "InceptionV3")
    plot_metric_dynamics(inception_res["history"], "InceptionV3")

    # Validation
    val_probs_i, val_targets_i = evaluate_model(trained["inception"], dls["299"], num_classes, split="val")
    val_preds_i = val_probs_i.argmax(dim=1).numpy()
    plot_confusion_matrix(val_targets_i.numpy(), val_preds_i, classes, "InceptionV3", "val", normalize=False)
    plot_confusion_matrix(val_targets_i.numpy(), val_preds_i, classes, "InceptionV3", "val", normalize=True)
    plot_roc(val_targets_i.numpy(), val_probs_i.numpy(), classes, "InceptionV3", "val")
    plot_pr_curves(val_targets_i.numpy(), val_probs_i.numpy(), classes, "InceptionV3", "val")
    plot_calibration(val_targets_i.numpy(), val_probs_i.numpy(), classes, "InceptionV3", "val")
    val_acc_i, val_metrics_i, val_macro_i = compute_metrics(val_probs_i, val_targets_i, num_classes)
    append_metrics_xlsx(val_metrics_i, classes, "InceptionV3", "val", metrics_xlsx, metrics_csv)

    # Test
    test_probs_i, test_targets_i = evaluate_model(trained["inception"], dls["299"], num_classes, split="test")
    test_preds_i = test_probs_i.argmax(dim=1).numpy()
    plot_confusion_matrix(test_targets_i.numpy(), test_preds_i, classes, "InceptionV3", "test", normalize=False)
    plot_confusion_matrix(test_targets_i.numpy(), test_preds_i, classes, "InceptionV3", "test", normalize=True)
    plot_roc(test_targets_i.numpy(), test_probs_i.numpy(), classes, "InceptionV3", "test")
    plot_pr_curves(test_targets_i.numpy(), test_probs_i.numpy(), classes, "InceptionV3", "test")
    plot_calibration(test_targets_i.numpy(), test_probs_i.numpy(), classes, "InceptionV3", "test")
    test_acc_i, test_metrics_i, test_macro_i = compute_metrics(test_probs_i, test_targets_i, num_classes)
    append_metrics_xlsx(test_metrics_i, classes, "InceptionV3", "test", metrics_xlsx, metrics_csv)

    # Error analysis
    try:
        imgs_val_299, targs_val_299 = get_split_batch(dls, "val", "299", max_samples=16)
        with torch.no_grad():
            logits_small = trained["inception"](imgs_val_299.to(DEVICE))
            preds_small = logits_small.argmax(dim=1).cpu().numpy()
        save_misclassified_images(imgs_val_299.cpu(), targs_val_299.cpu().numpy(), preds_small,
                                  classes, model_name="InceptionV3", split="val", max_samples=16)
    except Exception as e:
        log(f"InceptionV3 error analysis skipped: {e}")

    # ============== Ensemble evaluation ==============
    log("Ensemble evaluation on validation set")
    val_probs_ens, val_targets_ens = ensemble_predict(
        [trained["resnext"], trained["densenet"], trained["inception"]],
        dls, num_classes, split="val"
    )
    val_acc_ens, val_metrics_ens, val_macro_ens = compute_metrics(val_probs_ens, val_targets_ens, num_classes)
    log(f"Ensemble VAL acc={val_acc_ens:.4f} macro_f1={val_macro_ens['f1']:.4f}")
    append_metrics_xlsx(val_metrics_ens, classes, "Ensemble", "val", metrics_xlsx, metrics_csv)
    val_preds_ens = val_probs_ens.argmax(dim=1).numpy()
    plot_confusion_matrix(val_targets_ens.numpy(), val_preds_ens, classes, "Ensemble", "val", normalize=False)
    plot_confusion_matrix(val_targets_ens.numpy(), val_preds_ens, classes, "Ensemble", "val", normalize=True)
    plot_roc(val_targets_ens.numpy(), val_probs_ens.numpy(), classes, "Ensemble", "val")
    plot_pr_curves(val_targets_ens.numpy(), val_probs_ens.numpy(), classes, "Ensemble", "val")
    plot_calibration(val_targets_ens.numpy(), val_probs_ens.numpy(), classes, "Ensemble", "val")

    log("Ensemble evaluation on test set")
    test_probs_ens, test_targets_ens = ensemble_predict(
        [trained["resnext"], trained["densenet"], trained["inception"]],
        dls, num_classes, split="test"
    )
    test_acc_ens, test_metrics_ens, test_macro_ens = compute_metrics(test_probs_ens, test_targets_ens, num_classes)
    log(f"Ensemble TEST acc={test_acc_ens:.4f} macro_f1={test_macro_ens['f1']:.4f}")
    append_metrics_xlsx(test_metrics_ens, classes, "Ensemble", "test", metrics_xlsx, metrics_csv)
    test_preds_ens = test_probs_ens.argmax(dim=1).numpy()
    plot_confusion_matrix(test_targets_ens.numpy(), test_preds_ens, classes, "Ensemble", "test", normalize=False)
    plot_confusion_matrix(test_targets_ens.numpy(), test_preds_ens, classes, "Ensemble", "test", normalize=True)
    plot_roc(test_targets_ens.numpy(), test_probs_ens.numpy(), classes, "Ensemble", "test")
    plot_pr_curves(test_targets_ens.numpy(), test_probs_ens.numpy(), classes, "Ensemble", "test")
    plot_calibration(test_targets_ens.numpy(), test_probs_ens.numpy(), classes, "Ensemble", "test")

    # ============== Ensemble diversity (disagreement) ==============
    try:
        # Collect predictions for diversity on validation
        val_probs_r_np = val_probs_r.numpy()
        val_probs_d_np = val_probs_d.numpy()
        val_probs_i_np = val_probs_i.numpy()
        preds_list_val = [
            val_probs_r_np.argmax(axis=1),
            val_probs_d_np.argmax(axis=1),
            val_probs_i_np.argmax(axis=1),
        ]
        ensemble_diversity(preds_list_val, classes, split="val")

        # Collect predictions for diversity on test
        test_probs_r_np = test_probs_r.numpy()
        test_probs_d_np = test_probs_d.numpy()
        test_probs_i_np = test_probs_i.numpy()
        preds_list_test = [
            test_probs_r_np.argmax(axis=1),
            test_probs_d_np.argmax(axis=1),
            test_probs_i_np.argmax(axis=1),
        ]
        ensemble_diversity(preds_list_test, classes, split="test")
    except Exception as e:
        log(f"Ensemble diversity computation skipped: {e}")

    # --- Detailed per-class metrics for ensemble (test) ---
    for idx, cname in enumerate(classes):
        m = test_metrics_ens[idx]
        log(f"[{cname}] precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"specificity={m['specificity']:.4f} f1={m['f1']:.4f}")

    # --- Structured summary dump ---
    res = {
        "models": trained,
        "val": {
            "resnext": {"acc": val_acc_r, "macro": val_macro_r},
            "densenet": {"acc": val_acc_d, "macro": val_macro_d},
            "inception": {"acc": val_acc_i, "macro": val_macro_i},
            "ensemble": {"acc": val_acc_ens, "macro": val_macro_ens},
        },
        "test": {
            "resnext": {"acc": test_acc_r, "macro": test_macro_r},
            "densenet": {"acc": test_acc_d, "macro": test_macro_d},
            "inception": {"acc": test_acc_i, "macro": test_macro_i},
            "ensemble": {"acc": test_acc_ens, "macro": test_macro_ens},
        }
    }
    log(f"Summary: {res}")

    # Save structured summary JSON for reproducibility
    try:
        with open(CHECKPOINT_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "val": {
                    "resnext": res["val"]["resnext"],
                    "densenet": res["val"]["densenet"],
                    "inception": res["val"]["inception"],
                    "ensemble": res["val"]["ensemble"],
                },
                "test": {
                    "resnext": res["test"]["resnext"],
                    "densenet": res["test"]["densenet"],
                    "inception": res["test"]["inception"],
                    "ensemble": res["test"]["ensemble"],
                },
                "classes": classes,
                "scenario": scenario,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        log("Saved summary.json")
    except Exception as e:
        log(f"Failed to save summary.json: {e}")

    log("Experiment finished.")
    return res




# ------------------ Entry ------------------
if __name__ == "__main__":
    DATA_ROOT = "Datasets"
    SCENARIO = "5-classes"  # or "5-classes"
    resume_checkpoints = {
         "resnext": CHECKPOINT_DIR / "resnext50_epoch10.pth",
         "densenet": CHECKPOINT_DIR / "densenet161_epoch10.pth",
         "inception": CHECKPOINT_DIR / "inception_v3_epoch10.pth",
    }
    init_epochs_csv()
    res = run_experiment(DATA_ROOT, scenario=SCENARIO, batch_size=DEFAULT_BATCH_SIZE, num_workers=DEFAULT_NUM_WORKERS, resume=resume_checkpoints)
    log(f"Summary: {res}")

