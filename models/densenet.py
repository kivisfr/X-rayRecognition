# models/densenet.py

import torch.nn as nn
import torchvision.models as models


class DenseNet161WithDropout(nn.Module):
    """
    DenseNet-161 with custom classification head and dropout.
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.base = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1) if pretrained else models.densenet161()
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
