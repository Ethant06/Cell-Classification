"""Evaluate trained models and optionally persist a classification report."""

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader

from .model import CNN


def evaluate(
    model: CNN,
    dataloader: DataLoader,
    config: Mapping[str, Any],
    save_report: bool,
) -> float:
    """Measure test performance and optionally save a detailed text report.

    Args:
        model: Trained binary classifier.
        dataloader: Deterministic test loader.
        config: Experiment mapping containing ``report_path``.
        save_report: Whether to write balanced accuracy, macro-F1, and the
            scikit-learn classification report.

    Returns:
        Ordinary (unbalanced) accuracy: the fraction of test samples classified
        correctly. Balanced accuracy and macro-F1 are printed but not returned.
    """

    model.eval()
    all_preds = []
    all_labels = []
    print("Test samples:", len(dataloader.dataset))

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            logits = model(inputs).squeeze(1)
            probs = torch.sigmoid(logits)
            predicted = (probs >= 0.5).long()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_preds, dtype=int)
    final_accuracy = float(np.mean(y_pred == y_true))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if save_report:
        imbalance_header = (
            "Imbalance-aware summary (use alongside accuracy):\n"
            f"  balanced_accuracy: {bal_acc:.6f}\n"
            f"  macro_f1: {macro_f1:.6f}\n\n"
        )
        report = imbalance_header + classification_report(
            y_true, y_pred, digits=3, zero_division=0
        )
        os.makedirs(os.path.dirname(config['report_path']), exist_ok= True)
        with open(config['report_path'], 'w') as f:
            f.write(report)

    print(
        f"Balanced accuracy: {bal_acc:.6f} | Macro-F1: {macro_f1:.6f}"
    )
    return final_accuracy
