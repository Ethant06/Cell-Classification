from .model import CNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os


def train(model, dataloader, config) -> None:
    optimizer = optim.SGD(model.parameters(), lr = config['lr'], momentum = config['momentum'])
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for i, data in enumerate(dataloader, 0): #block will iterate through total data/ #batch_size
            print(f"Batch {i}")
            inputs, labels = data # inputs is batch of images: [32, 1 channel, 128, 128]
            optimizer.zero_grad() # labels is a tensor [0, 1, 0, 0, 1, 1, 0] for the labels
            prediction = model(inputs) # output shape [32, 2]
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(prediction.detach(), 1)
            epoch_total += labels.size(0)
            epoch_loss += loss.item()
            epoch_correct += (predicted == labels).sum().item()
              
        epoch_loss_sum = epoch_loss / len(dataloader) #dataloader is a tuple (image: [batch_size, H, W], label)
        epoch_accuracy = (epoch_correct / epoch_total)
        train_loss.append(epoch_loss_sum)
        train_acc.append(epoch_accuracy)
        
    # plots
    fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 4))
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
        
    
        