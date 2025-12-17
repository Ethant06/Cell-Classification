from .model import CNN
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os

def evaluate(model, dataloader, config, save_plot) -> float:
    """
    Evaluates trained CNN model on test dataset.

    - Runs prediction on the test DataLoader
    - Computes loss and accuracy per batch
    - Collects predictions and true labels
    - saves:
        * Loss and accuracy per batch plots
        * Confusion matrix
        * Classification report
    - Returns overall test accuracy
    """


    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_losses = []
    batch_accuracy = []
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    print("Test samples:", len(dataloader.dataset))

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            print(f"Test Batch {i}") # i is for testing purposes
            inputs, labels = data
            predictions = model(inputs)
            loss = criterion(predictions, labels)

            _, predicted = torch.max(predictions.detach(), 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = (correct / total)

            batch_losses.append(loss.item())
            batch_accuracy.append(accuracy)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
        ax2.set_title("Accuracy Per Batch")
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
        report = classification_report(all_labels, all_preds, digits = 3)
        os.makedirs(os.path.dirname(config['report_path']), exist_ok= True)
        with open(config['report_path'], 'w') as f:
            f.write(report)

    #final accuracy is recorded for each seed for all experimet
    final_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return final_accuracy
