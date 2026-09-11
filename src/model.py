"""Convolutional neural network used by every experiment."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class CNN(nn.Module):
    """Binary classifier for normalized 128×128 grayscale images.

    Three convolutional blocks reduce the image to 128 feature maps of size
    8×8. Two fully connected layers then produce one classification logit per
    image. Inputs have shape ``(batch, 1, 128, 128)`` and outputs have shape
    ``(batch, 1)``.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize model layers from dropout-related configuration values.

        ``regularization`` toggles Dropout modules and defaults to ``True``;
        ``dropout_rate`` defaults to 0.25. Reported configurations set the rate
        to zero, so their Dropout modules are effective no-ops.
        """
        super().__init__()

        self.use_dropout = config.get('regularization', True)
        dropout_rate = config.get('dropout_rate', 0.25)
        self.dropout = nn.Dropout(dropout_rate) if self.use_dropout else nn.Identity()

        self.layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # downsample: 128 -> 64
            nn.ELU(),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized binary-classification logit per image."""
        x = self.layers(x)
        x = self.fc_layers(x)
        return x