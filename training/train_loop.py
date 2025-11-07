# training/train_loop.py
import time

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from logging_utils.logger import log, append_epoch_csv, append_epoch_xlsx
from training.checkpointing import save_checkpoint, resume_training
from training.evaluate import evaluate_model, compute_metrics

from project_root.config import TRAINING_CONFIG
from training.checkpointing import load_checkpoint

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training a model in one epoch.
    """
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        if isinstance(outputs, tuple):  # for Inception
            outputs = outputs[0]

        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * xb.size(0)
        running_corrects += (preds == yb).sum().item()
        total += xb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validation of the model for one epoch.
    """
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, yb)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * xb.size(0)
            running_corrects += (preds == yb).sum().item()
            total += xb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc

def train_model_staged(model_name, model, dataloaders, num_classes,
                       device="cuda", out_dir="checkpoints", resume_path = None):
    """
    Двухэтапное обучение: сначала только голова, потом fine-tuning всей сети.
    """

    num_epochs_stage1 = TRAINING_CONFIG["stage1"]["epochs"],
    num_epochs_stage2 = TRAINING_CONFIG["stage2"]["epochs"],
    lr_stage1 = TRAINING_CONFIG["stage1"]["lr"],
    lr_stage2 = TRAINING_CONFIG["stage2"]["lr"]

    # Checking for cortèges
    if isinstance(num_epochs_stage1, tuple):
        num_epochs_stage1 = num_epochs_stage1[0]
    if isinstance(num_epochs_stage2, tuple):
        num_epochs_stage2 = num_epochs_stage2[0]
    if isinstance(lr_stage1, tuple):
        lr_stage1 = lr_stage1[0]
    if isinstance(lr_stage2, tuple):
        lr_stage2 = lr_stage2[0]

    # --- Stage 1: обучаем только классификатор ---
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "base_model"):
        base = model.base_model
    else:
        base = model

    if hasattr(base, "fc"):  # ResNet / ResNeXt / Inception
        for param in base.fc.parameters():
            param.requires_grad = True
    elif hasattr(base, "classifier"):  # DenseNet
        for param in base.classifier.parameters():
            param.requires_grad = True
    elif hasattr(base, "head"):  # вдруг кастомная реализация
        for param in base.head.parameters():
            param.requires_grad = True
    else:
        raise AttributeError(f"Не найден классификатор у модели {model_name}")


    log(f"=== Stage 1: обучение головы ({model_name}) ===")
    train_model_full(model_name + "_stage1", model, dataloaders, num_classes,
                     num_epochs=num_epochs_stage1, lr=lr_stage1,
                     device=device, out_dir=out_dir)

    # --- Stage 2: размораживаем всю модель ---
    for param in model.parameters():
        param.requires_grad = True

    log(f"=== Stage 2: fine-tuning всей модели ({model_name}) ===")
    result = train_model_full(model_name + "_stage2", model, dataloaders, num_classes,
                              num_epochs=num_epochs_stage2, lr=lr_stage2,
                              device=device, out_dir=out_dir)

    return result

def train_model_full(model_name: str, model: nn.Module, dataloaders,
                     num_classes: int,
                     num_epochs : int, lr : float, resume_path = None,
                     device="cuda", out_dir="checkpoints"):
    """
    Full model training cycle with logging and checkpoints.
    """
    log("=" * 60)
    log(f"Start training model: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if resume_path:
        if resume_path.exists():
            start_epoch, best_val = load_checkpoint(model, optimizer, resume_path)
        else:
            log(f"Resume path {resume_path} not found. Starting from scratch.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # restoration of training
    start_epoch, history = resume_training(model, optimizer, resume_path, device=device)

    # initialize history if new
    if not history:
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": [],
            "epoch_time": []
        }

    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        log(f"=== Epoch {epoch+1}/{num_epochs} ({model_name}) ===")

        time_start = time.time()

        # training
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        # validation
        val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, device)

        # calculation of validation metrics
        probs, targets = evaluate_model(model, dataloaders, num_classes, split="val", device=device)
        _, metrics, macro = compute_metrics(probs, targets, num_classes)

        time_end = time.time()
        epoch_time = time_end - time_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(macro["f1"])
        history["val_f1"].append(macro["f1"])
        history["train_precision"].append(macro["precision"])
        history["val_precision"].append(macro["precision"])
        history["train_recall"].append(macro["recall"])
        history["val_recall"].append(macro["recall"])
        history["epoch_time"].append(epoch_time)

        log(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={macro['f1']:.4f} |"
            f"epoch_time={epoch_time:.4f}")

        # logging to CSV/XLSX
        metrics_dict = {
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "val_f1": macro["f1"],
            "val_precision": macro["precision"], "val_recall": macro["recall"],
            "epoch_time": epoch_time
        }
        append_epoch_csv(epoch+1, metrics_dict, out_dir / f"{model_name}_epochs.csv")
        append_epoch_xlsx(epoch+1, metrics_dict, out_dir / f"{model_name}_epochs.xlsx")

        # saving a checkpoint
        save_checkpoint(model, optimizer, epoch+1, history,
                        out_dir / f"{model_name}_epoch{epoch+1}.pth")

    return {"model": model, "history": history}
