import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input shape: (batch_size, 3, 224, 224)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # Shape: (batch_size, 32, 224, 224) 
            # Formula: output_size = (input_size - kernel_size + 2*padding)/stride + 1
            # (224 - 3 + 2*1)/1 + 1 = 224
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Shape: (batch_size, 32, 112, 112)
            # Formula: output_size = (input_size)/stride
            # 224/2 = 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # Shape: (batch_size, 64, 112, 112)
            # (112 - 3 + 2*1)/1 + 1 = 112
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Shape: (batch_size, 64, 56, 56)
            # 112/2 = 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # Shape: (batch_size, 128, 56, 56)
            # (56 - 3 + 2*1)/1 + 1 = 56
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Shape: (batch_size, 128, 28, 28)
            # 56/2 = 28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # Shape: (batch_size, 256, 28, 28)
            # (28 - 3 + 2*1)/1 + 1 = 28
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # Shape: (batch_size, 256, 14, 14)
            # 28/2 = 14
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # Input shape: (batch_size, 256 * 14 * 14)
            # Flatten operation: 256 channels * 14 * 14 = 50176 features
            nn.Linear(256 * 14 * 14, 512),
            # Shape: (batch_size, 512)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            # Shape: (batch_size, 1)
            nn.Sigmoid()
            # Final Shape: (batch_size, 1)
            # Output is a single probability value between 0 and 1
        )

    def forward(self, x):
        # Input shape: (batch_size, 3, 224, 224)
        x = self.conv_layers(x)
        # Shape after conv_layers: (batch_size, 256, 14, 14)
        x = x.view(x.size(0), -1)
        # Shape after flatten: (batch_size, 256 * 14 * 14)
        x = self.fc_layers(x)
        # Final shape: (batch_size, 1)
        return x

