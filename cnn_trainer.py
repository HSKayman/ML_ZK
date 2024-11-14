
#!pip install torch 
#!pip install torchsummary
#!pip install torchvision


import torch
from datetime import datetime
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        # Input: 1x224x224
        self.features = nn.Sequential(
            # Conv1: (224+4-5)/1 + 1 = 224
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # Output: 6x224x224
            nn.ReLU(inplace=True),
            # MaxPool1: (224-2)/2 + 1 = 112
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 6x112x112
            
            # Conv2: (112+4-5)/1 + 1 = 112
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),  # Output: 16x112x112
            nn.ReLU(inplace=True),
            # MaxPool2: (112-2)/2 + 1 = 56
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x56x56

            # Conv3: (56+2-3)/2 + 1 = 28
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 32x28x28
            nn.ReLU(inplace=True),
            # MaxPool3: (28-2)/2 + 1 = 14
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32x14x14
            
            # Conv4: (14+2-3)/2 + 1 = 7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64x7x7
            nn.ReLU(inplace=True),
            # MaxPool4: (7-2)/2 + 1 = 3
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x3x3
        )
        
        # Calculate flattened size: 64 * 3 * 3 = 576
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 3 * 3, 256),  # 3136 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.features(x)

        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.classifier(x)
        return x

if '__main__' == __name__: # for importing this module in other modules

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if device.type == 'cuda':
        print(f'GPU Device: {torch.cuda.get_device_name(0)}')
        print(f'Memory Usage:')
        print(f'Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB')
        print(f'Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB')

    model = CNN().to(device)
    summary(model, (1, 224, 224))
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert input picture to tensor
    matrix_converter = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((224, 224)),                  # Resize to desired dimensions
        transforms.ToTensor()    
    ])

    # Set data set directory
    data_dir = 'data_for_model_1/'

    # Load data set
    dataset = datasets.ImageFolder(data_dir,transform=matrix_converter)

    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Noof classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    print(f"Total samples: {len(dataset)}")

    num_epochs = 100 

    # Training loop
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0.0
        correct = 0
        epoch_start_time = time.time()
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Forward pass
            predictions = model(images)
            loss = loss_func(predictions, targets)
            
            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            correct += (predicted == targets).sum().item()

            # Backward pass and optimization
            optimizer.zero_grad()  # clear gradients
            loss.backward()        # compute gradients
            optimizer.step()       # update parameters
            
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
                
        # print average loss for the epoch
        avg_loss = total_loss / len(dataset)
        accuracy = 100 * correct / len(dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-" * 30)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    times = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_accuracy': accuracy,
    }, f'CNN_model_final_{times}.pth')

    print(f"Best accuracy achieved: {accuracy:.2f}%")
