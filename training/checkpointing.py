# training/checkpointing.py

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, history, out_path):
    """
    Saves the model and optimizer checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
       Current model.
    optimizer : torch.optim.Optimizer
       Optimizer.
    epoch : int
        Current epoch number.
    history : dict
       Training history (loss, acc, metrics).
    out_path : Path or str
       Path to save the .pth file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, out_path)


def load_checkpoint(model, optimizer, in_path, device="cuda"):
    """
   Loads the model checkpoint and optimizer.

    Parameters
    ----------
    model : torch.nn.Module
       The model into which we load weights.
    optimizer : torch.optim.Optimizer
       The optimizer into which we load the state.
    in_path : Path or str
       Path to the .pth file.
    device : str
        Device ("cuda" or "cpu").

    Returns
    -------
    epoch : int
       The last epoch.
    history : dict
        History of training.
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")

    checkpoint = torch.load(in_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", 0)
    history = checkpoint.get("history", {})

    return epoch, history


def resume_training(model, optimizer, resume_path, device="cuda"):
    """
    Convenient wrapper: if the path to a checkpoint is specified, it loads it.

    Returns
    -------
    start_epoch : int
       From what era should I continue training.
    history : dict
        History of training.
    """
    if resume_path is None:
        return 0, {}

    epoch, history = load_checkpoint(model, optimizer, resume_path, device=device)
    print(f"Resumed training from epoch {epoch}")
    return epoch, history
