# models/ensemble.py
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from logging_utils.logger import log
from training.train_loop import validate_one_epoch


class EnsembleModel(nn.Module):
    """
    Model ensemble: averages probabilities (softmax) from multiple models.
    """

    def __init__(self, models: dict, device="cuda"):
        """
        Parameters
        ----------
        models : dict[str, nn.Module]
            Dictionary of trained models {"resnext": model1, "densenet": model2, ...}
        device : str
            Device ("cuda" или "cpu").
        """
        super().__init__()
        self.models = models
        self.device = device

        # We translate all models into eval and onto the required device.
        for m in self.models.values():
            m.to(self.device)
            m.eval()

    def forward(self, x):
        """
        Run the input through all models and average the probabilities.
        """
        probs_list = []
        for name, model in self.models.items():
            outputs = model(x)
            if isinstance(outputs, tuple):  # для Inception
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            probs_list.append(probs)

        # averaging of probabilities
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return avg_probs

    @torch.no_grad()
    def ensemble_predict(self, models: List[nn.Module], dataloaders: Dict, num_classes: int, split: str = "test"):
        probs_list = []
        targets_ref = None
        order = [("resnext","224"), ("densenet","224"), ("inception","299")]
        for idx, (name, pipeline) in enumerate(order):
            mdl = models[idx]
            acc, probs, targets, eval_time = validate_one_epoch(mdl, dataloaders[pipeline][split])
            log(f"[Ensemble] {name} {split} acc={acc:.4f} eval_time={eval_time:.1f}s")
            probs_list.append(probs)
            if targets_ref is None:
                targets_ref = targets
            else:
                if targets_ref.size(0) != targets.size(0):
                    log(f"Warning: target size mismatch between models for split={split} ({targets_ref.size(0)} vs {targets.size(0)})")
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return avg_probs, targets_ref

    def ensemble_diversity(self, preds_list: List[np.ndarray], classes: List[str], split="val"):
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
