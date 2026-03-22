from .model import CNN
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
import numpy as np
import os

def evaluate(model, dataloader, config, save_plot) -> float:
    """
    Evaluates trained CNN model on test dataset.

    - Runs prediction on the test DataLoader
    - Computes loss per batch and cumulative accuracy after each batch
    - Collects predictions and true labels
    - saves:
        * Loss per batch and cumulative-accuracy plots
        * Confusion matrix
        * Classification report
    - Returns overall test accuracy
    """

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    batch_losses = []
    batch_accuracy = []
    all_preds = []
    all_labels = []
    seen = 0
    correct_so_far = 0
    print("Test samples:", len(dataloader.dataset))

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            logits = model(inputs).squeeze(1)
            labels = labels.float()
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            predicted = (probs >= 0.5).long()

            batch_n = labels.size(0)
            batch_correct = (predicted == labels.long()).sum().item()
            seen += batch_n
            correct_so_far += batch_correct
            batch_accuracy.append(correct_so_far / seen)

            batch_losses.append(loss.item())

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_preds, dtype=int)
    final_accuracy = float(np.mean(y_pred == y_true))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    #save_plot is true if the experiments are running on the first seed only.
    #plots are saved if and only if experiment is running in the first seed.
    if save_plot == True:
        os.makedirs(os.path.dirname(config['plot_path_test']), exist_ok=True)
        os.makedirs(os.path.dirname(config['plot_confusion']), exist_ok=True)
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(12, 4))
        ax1.plot(batch_losses)
        ax1.set_title("Loss Per Batch")
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax2.plot(batch_accuracy)
        ax2.set_title("Cumulative accuracy (after each batch)")
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(config['plot_path_test'])
        plt.close(fig)

        # plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        disp.plot(cmap='Blues', ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(config['plot_confusion'])
        plt.close(fig_cm)

        # report of performance for all_plots/experiment/report.txt for first seed
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
