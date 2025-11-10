# training/train_loop.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F

from logging_utils.logger import log, append_epoch_csv, append_epoch_xlsx
from logging_utils.plots import plot_training_curves, plot_metric_dynamics
from losses.focal_loss import FocalLoss
from training.checkpointing import save_checkpoint, resume_training
from training.evaluate import evaluate_model, compute_metrics

from project_root.config import TRAINING_CONFIG, DEVICE, FOCAL_GAMMA, AUX_LOSS_WEIGHT
from training.checkpointing import load_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device = DEVICE, aux_loss_weight = AUX_LOSS_WEIGHT):
    """
    Training a model in one epoch.
    """
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):  # for Inception
            main_logits, aux_logits = outputs
            loss_main = criterion(main_logits, targets)
            loss_aux = criterion(aux_logits, targets)
            loss = loss_main + aux_loss_weight * loss_aux
            logits = main_logits
        else:
            loss = criterion(outputs, targets)
            logits = outputs

        loss.backward()
        optimizer.step()

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total  if total>0 else 0.0
    epoch_acc = running_corrects / total  if total>0 else 0.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, optimizer, device=DEVICE, aux_loss_weight = AUX_LOSS_WEIGHT):
    """
    Validation of the model for one epoch.
    """
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0


    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            main_logits, aux_logits = outputs
            loss_main = criterion(main_logits, targets)
            loss_aux = criterion(aux_logits, targets)
            loss = loss_main + aux_loss_weight * loss_aux
            logits = main_logits
        else:
            loss = criterion(outputs, targets)
            logits = outputs

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total if total>0 else 0.0
    epoch_acc = running_corrects / total  if total>0 else 0.0
    return epoch_loss, epoch_acc

def train_model_staged(model_name, model, dataloaders, num_classes,
                       device="cuda", out_dir="Checkpoints"):
    """
    Two-stage training: first only the head, then fine-tuning the entire network.
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

    # --- Stage 1: Train only the classifier ---
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
    elif hasattr(base, "head"):  # for a custom implementation
        for param in base.head.parameters():
            param.requires_grad = True
    else:
        raise AttributeError(f"No classifier found for the model {model_name}")

    model_start_time = time.time()
    log(f"=== Stage 1: head training ({model_name}) ===")
    train_model_full(model_name + "_stage1", model, dataloaders, num_classes,
                     num_epochs=num_epochs_stage1, lr=lr_stage1,
                     device=device, out_dir=out_dir)

    # --- Stage 2: we defrost the entire model ---
    for param in model.parameters():
        param.requires_grad = True

    log(f"=== Stage 2: fine-tuning the entire model ({model_name}) ===")
    result = train_model_full(model_name + "_stage2", model, dataloaders, num_classes,
                              num_epochs=num_epochs_stage2, lr=lr_stage2,
                              device=device, out_dir=out_dir)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    log(f"=== {model_time/60} minutes for training ({model_name}) ===")

    return result

def train_model_full(model_name: str, model: nn.Module, dataloaders,
                     num_classes: int,
                     num_epochs : int, lr : float, resume_path = None,
                     device="cuda", out_dir="Checkpoints"):
    """
    Full model training cycle with logging and checkpoints.
    """
    log("=" * 60)
    log(f"Start training model: {model_name}")
    model = model.to(DEVICE)

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    for g in optimizer.param_groups:
        g['lr'] = lr

    for epoch in range(start_epoch, num_epochs):
        log(f"=== Epoch {epoch+1}/{num_epochs} ({model_name}) ===")

        time_start = time.time()

        # training
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device, aux_loss_weight=AUX_LOSS_WEIGHT)
        # validation
        val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, optimizer, device, aux_loss_weight=AUX_LOSS_WEIGHT)

        # calculation of validation metrics
        probs, targets = evaluate_model(model, dataloaders, split="val", device=device)
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
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={macro['f1']:.4f} | "
            f"epoch_time={epoch_time:.4f}")

        # logging to CSV/XLSX
        metrics_dict = {
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "val_f1": macro["f1"],
            "val_precision": macro["precision"], "val_recall": macro["recall"],
            "epoch_time": epoch_time
        }
        append_epoch_csv(epoch+1, metrics_dict, out_dir / "csv" / f"{model_name}_epochs.csv")
        append_epoch_xlsx(epoch+1, metrics_dict, out_dir / "xlsx" / f"{model_name}_epochs.xlsx")


        # saving a checkpoint
        save_checkpoint(model, optimizer, epoch+1, history,
                        out_dir / f"{model_name}_epoch{epoch+1}.pth")

    plot_training_curves(history, model_name=model_name, out_dir=out_dir)
    plot_metric_dynamics(history, model_name=model_name, out_dir=out_dir)


    return {"model": model, "history": history}
