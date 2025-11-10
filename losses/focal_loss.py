from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch

from project_root.config import FOCAL_GAMMA


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