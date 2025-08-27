import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        self.use_dropout = config.get('regularization')
        dropout_rate = config.get('dropout_rate')
        self.dropout = nn.Dropout(dropout_rate) if self.use_dropout else nn.Identity()
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, out_channels = 32, 
                kernel_size = 3, stride = 1, padding = 1
            ), # output dimension = [128 - 3 + (2*1)]/1 + 1 = 128
              # After first conv layer: 32(channels) x 128 x 128 pixels total
            nn.ELU(),
            self.dropout,
            nn.MaxPool2d(kernel_size = 2, stride = 2), # 32(channels) x 64 x 64 total pixels
            nn.Conv2d(
                in_channels = 32, out_channels = 64, 
                kernel_size = 3, stride = 1, padding = 1
                ),# output dimension = [64 - 3 + (2 * 1)]/1 + 1 = 64(channel) x 64 x 64
            nn.ELU(),
            self.dropout,
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # output dimension = (64 - 2)/2 + 1 = 64(channels) x 32 x 32
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 32),
            nn.ReLU(),
            self.dropout,
            nn.Linear(32, 2),
            
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = self.fc_layers(x)
        return x
        
    