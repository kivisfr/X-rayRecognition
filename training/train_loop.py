# training/train_loop.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from logging_utils.logger import log, append_epoch_csv, append_epoch_xlsx
from training.checkpointing import save_checkpoint, resume_training
from training.evaluate import evaluate_model, compute_metrics


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


def train_model_full(model_name, model, dataloaders, num_classes,
                     num_epochs=10, lr=1e-4, resume_path=None,
                     device="cuda", out_dir="checkpoints"):
    """
    Full model training cycle with logging and checkpoints.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # restoration of training
    start_epoch, history = resume_training(model, optimizer, resume_path, device=device)

    # initialize history if new
    if not history:
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": []
        }

    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        log(f"=== Epoch {epoch+1}/{num_epochs} ({model_name}) ===")

        # training
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        # validation
        val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, device)

        # calculation of validation metrics
        probs, targets = evaluate_model(model, dataloaders, num_classes, split="val", device=device)
        _, metrics, macro = compute_metrics(probs, targets, num_classes)

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

        log(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={macro['f1']:.4f}")

        # logging to CSV/XLSX
        metrics_dict = {
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "val_f1": macro["f1"], "val_precision": macro["precision"], "val_recall": macro["recall"]
        }
        append_epoch_csv(epoch+1, metrics_dict, out_dir / f"{model_name}_epochs.csv")
        append_epoch_xlsx(epoch+1, metrics_dict, out_dir / f"{model_name}_epochs.xlsx")

        # saving a checkpoint
        save_checkpoint(model, optimizer, epoch+1, history,
                        out_dir / f"{model_name}_epoch{epoch+1}.pth")

    return {"model": model, "history": history}
