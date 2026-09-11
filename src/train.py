import torch.optim as optim
import torch
import torch.nn as nn


def train(model, dataloader, config) -> float:
    """
    - Iterates over the training dataset for a fixed number of epochs
    - Performs forward and backward passes
    - Updates model parameters using SGD
    - Reports loss and accuracy per epoch
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
        epoch_accuracy = (epoch_correct / epoch_total)
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

    final_train_accuracy = final_train_correct / final_train_total
    return final_train_accuracy
