from __future__ import annotations

import torch
import torch.nn as nn


class RSDF(nn.Module):
    """Reconstruction-Space Denoising Front-end from Eq. (5)."""

    def __init__(self, residual_std: float = 1e-3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=residual_std)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class CTClassifier3(nn.Module):
    """Three-block binary classifier specified in the manuscript (23,426 params)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTClassifier5(nn.Module):
    """Five-block reference CNN matching the paper's reported 392,834 parameters.

    The paper reports the parameter count but not the layer-by-layer CNN5 table.
    Channels 1->16->32->64->128->256 plus a 256->2 head reproduce that count exactly.
    Pooling after the first four blocks is an explicit reconstruction choice.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiagnosticPipeline(nn.Module):
    """Optional RSDF followed by a classifier backbone."""

    def __init__(self, classifier: nn.Module | None = None, use_rsdf: bool = False):
        super().__init__()
        self.rsdf = RSDF() if use_rsdf else nn.Identity()
        self.classifier = classifier if classifier is not None else CTClassifier3()

    def forward(self, recon: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.rsdf(recon))
