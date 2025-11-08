# models/resnext.py

import torch.nn as nn
import torchvision.models as models


class ResNeXt50WithDropout(nn.Module):
    """
    ResNeXt-50 32x4d with custom classification head and dropout.
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.5, pretrained: bool = True):
        super().__init__()

        # Loading the base model
        self.base_model = models.resnext50_32x4d(weights="IMAGENET1K_V1" if pretrained else None)

        # The size of the last block output
        in_features = self.base_model.fc.in_features

        # Replacing the classification head
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
