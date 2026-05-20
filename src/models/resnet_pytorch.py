"""
ResNet50 in PyTorch — the cross-framework comparison model.

Why ResNet here, not VGG: keeping the architecture different from the Keras
side makes the comparison more informative. If we used VGG in both frameworks
we'd mostly be measuring framework quirks; using ResNet adds a second
architectural axis for the comparison table.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet_pytorch(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    if num_classes == 2:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1),
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes),
        )
    return model
