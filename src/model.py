import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
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
        
    def forward(self, x):
        x = self.layers(x)
        x = self.fc_layers(x)
        return x