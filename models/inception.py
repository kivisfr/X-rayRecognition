# models/inception.py

import torch
import torch.nn as nn
import torchvision.models as models


class InceptionV3Head(nn.Module):
    """
    Inception-V3 with a custom classification head.
    Takes into account the presence of an auxiliary head (aux_logits).
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.5, pretrained: bool = True):
        super().__init__()

        # Loading the base model
        self.base_model = models.inception_v3(
            weights="IMAGENET1K_V1" if pretrained else None,
            aux_logits=True  # включаем aux для совместимости
        )

        # Main head
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

        # Auxiliary head (aux_logits)
        if self.base_model.AuxLogits is not None:
            in_features_aux = self.base_model.AuxLogits.fc.in_features
            self.base_model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features_aux, num_classes)
            )

    def forward(self, x):
        """
        Inception returns (output, aux_output) in training mode.
        For compatibility, we return only the main output.
        """
        if self.training and self.base_model.AuxLogits is not None:
            outputs, aux_outputs = self.base_model(x)
            return outputs, aux_outputs
        else:
            return self.base_model(x)
