from .model import CNN
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os

def evaluate(model, dataloader, config) -> None:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_losses = []
    batch_accuracy = []
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
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
    
    os.makedirs(os.path.dirname(config['plot_path_test']), exist_ok=True)
    os.makedirs(os.path.dirname(config['plot_confusion']), exist_ok=True)
    fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 4))
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
    
    # plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config['plot_confusion'])
    
    # report of performance
    report = classification_report(all_labels, all_preds, digits = 3)
    os.makedirs(os.path.dirname(config['report_path']), exist_ok= True)
    with open(config['report_path'], 'w') as f:
        f.write(report)