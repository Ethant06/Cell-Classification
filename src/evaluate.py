import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
import numpy as np
import os

def evaluate(model, dataloader, config, save_report) -> float:
    """
    Evaluates trained CNN model on test dataset.

    - Runs prediction on the test DataLoader
    - Collects predictions and true labels
    - Saves a classification report for the first seed
    - Returns overall test accuracy
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
