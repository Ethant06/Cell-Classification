"""Model training loop and final training-accuracy calculation."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import CNN


def train(
    model: CNN, dataloader: DataLoader, config: Mapping[str, Any]
) -> float:
    """Optimize a binary classifier and return final training-loader accuracy.

    Optimization uses ``BCEWithLogitsLoss`` and SGD. Epoch loss is the mean of
    batch losses, and predictions use a sigmoid threshold of 0.5. The final
    accuracy pass reuses the training dataset, so stochastic image augmentation
    remains active even though the model itself is in evaluation mode.

    Args:
        model: CNN whose parameters will be updated in place.
        dataloader: Training samples, including any configured augmentation.
        config: Mapping containing ``epochs``, ``lr``, and ``momentum``.

    Returns:
        Fraction of correctly classified samples during a final evaluation pass.
    """
    optimizer = optim.SGD(model.parameters(), lr = config['lr'], momentum = config['momentum'])
    criterion = nn.BCEWithLogitsLoss()
    print("Train samples:", len(dataloader.dataset))
    model.train()
    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for data in dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)
            labels = labels.float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            predicted = (probs >= 0.5).long()

            epoch_total += labels.size(0)
            epoch_loss += loss.item()
            epoch_correct += (predicted == labels.long()).sum().item()


        epoch_loss_sum = epoch_loss / len(dataloader)
        epoch_accuracy = epoch_correct / epoch_total
        print(
            f"  Epoch {epoch + 1}/{n_epochs}  loss={epoch_loss_sum:.4f}  train_acc={epoch_accuracy:.4f}",
            flush=True,
        )

    model.eval()
    final_train_correct = 0
    final_train_total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            logits = model(inputs).squeeze(1)
            probs = torch.sigmoid(logits)
            predicted = (probs >= 0.5).long()
            final_train_total += labels.size(0)
            final_train_correct += (predicted == labels).sum().item()

    return final_train_correct / final_train_total
