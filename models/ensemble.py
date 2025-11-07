# models/ensemble.py

import torch
import torch.nn as nn


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
