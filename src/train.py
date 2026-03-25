from .model import CNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os


def train(model, dataloader, config, save_plots) -> float:
    """
    - Iterates over the training dataset for a fixed number of epochs
    - Performs forward and backward passes
    - Updates model parameters using SGD
    - Tracks loss and accuracy per epoch
    - Optionally saves training loss and accuracy plots depending on parameter save_plots
    """
    optimizer = optim.SGD(model.parameters(), lr = config['lr'], momentum = config['momentum'])
    criterion = nn.BCEWithLogitsLoss()
    train_loss = []
    train_acc = []
    print("Train samples:", len(dataloader.dataset))
    model.train()
    for _ in range(config['epochs']):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, data in enumerate(dataloader):
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
        epoch_accuracy = (epoch_correct / epoch_total)
        train_loss.append(epoch_loss_sum)
        train_acc.append(epoch_accuracy)
    
    # save plots if the experiment is running in the first seed. Recorded in all_plots/experiment_type/
    if save_plots == True:
        fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 8))
        ax1.plot(train_loss)
        ax1.set_title("Loss Per Epoch")
        ax2.plot(train_acc)
        ax2.set_title("Training Accuracy Per Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        plt.tight_layout()
        plot_dir = os.path.dirname(config['plot_path_train'])
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(config['plot_path_train'])
        plt.close(fig)


    model.eval()
    final_train_correct = 0
    final_train_total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            predictions = model(inputs)
            _, predicted = torch.max(predictions, 1)
            final_train_total += labels.size(0)
            final_train_correct += (predicted == labels).sum().item()

    final_train_accuracy = final_train_correct / final_train_total
    return final_train_accuracy
