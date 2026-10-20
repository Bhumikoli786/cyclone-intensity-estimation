"""PyTorch models. Every model maps (B, C, H, W) images -> (B,) standardised wind speed.

Standardised means (wind_kt - train_mean) / train_std; train.py stores mean/std inside the checkpoint
and predict.py converts back to knots.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_NAMES = ("cnn", "resnet18")


class SimpleCNN(nn.Module):
    """Small CNN trained from scratch: 4 x (conv-BN-ReLU-pool) -> global average pool -> dense head."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        chans = [in_channels, 16, 32, 64, 128]
        layers: list[nn.Module] = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(1)


class ResNetRegressor(nn.Module):
    """ResNet-18 with a 1-channel first layer and a single-output regression head.

    With pretrained=True the ImageNet weights are used (RGB filters are summed into one grey channel).
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision import models  # imported here so the CNN works without torchvision
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        old = net.conv1
        new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                new.weight.copy_(old.weight.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels)
        net.conv1 = new
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, 1))
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(name: str, in_channels: int = 1, pretrained: bool = False, dropout: float = 0.3) -> nn.Module:
    if name == "cnn":
        return SimpleCNN(in_channels, dropout)
    if name == "resnet18":
        return ResNetRegressor(in_channels, pretrained, dropout)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 20):
    """Monte-Carlo dropout: run n stochastic passes with dropout ON (BatchNorm stays in eval mode).

    Returns (mean, std) in standardised units. The std is a rough uncertainty indicator, not a
    calibrated confidence interval.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    draws = torch.stack([model(x) for _ in range(n_samples)])
    model.eval()
    return draws.mean(0), draws.std(0)
