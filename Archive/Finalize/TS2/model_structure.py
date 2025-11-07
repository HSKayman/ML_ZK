import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from collections import defaultdict

# ResNet Architecture
def get_resnet_model(class_count=2):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# SqueezeNet Architecture
def get_squ_model(class_count=2):
    model = models.squeezenet1_1(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Conv2d(512, class_count, kernel_size=1),  
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1))
    )

    return model

# Data Preprocessing
def get_preprocessing_transforms(input_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                    std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                    std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Metric Tracker
class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metric_dict):
        for key, value in metric_dict.items():
            self.metrics[key].append(value)
    
    def get_metrics(self):
        return self.metrics

# Plotting Functions
def plot_metrics(tracker,model_number):
    metrics = tracker.get_metrics()
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(metrics['train_loss'], label='Train Loss')
    ax1.plot(metrics['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy and F1 score
    ax2.plot(metrics['train_acc'], label='Train Accuracy')
    ax2.plot(metrics['val_acc'], label='Validation Accuracy')
    ax2.plot(metrics['train_f1'], label='Train F1')
    ax2.plot(metrics['val_f1'], label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy and F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_M{model_number}.png')
    plt.close()

# Confusion Matrix
def plot_confusion_matrices(train_cm, val_cm, model_number):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training confusion matrix
    sns.heatmap(train_cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Training Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Plot validation confusion matrix
    sns.heatmap(val_cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('Validation Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrices_M{model_number}.png')
    plt.close()

# Evaluation Function
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1) 
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    f1 = f1_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    
    return accuracy, f1, avg_loss, cm

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', model_number=1):
    best_val_acc = 0.0
    tracker = MetricTracker()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_cm = confusion_matrix(train_labels, train_preds)
        train_acc = (train_cm[0,0] + train_cm[1,1]) / np.sum(train_cm)
        train_f1 = f1_score(train_labels, train_preds)
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_acc, val_f1, val_loss, val_cm = evaluate_model(model, val_loader, device)
        
        # Update metrics tracker
        tracker.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1
        })
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Training Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}')
        print(f'Training F1: {train_f1:.4f}, Validation F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{model_number}_EP{epoch+1}.pth')
            
        # # Plot metrics every 5 epochs
        # if (epoch + 1) % 5 == 0:
        #     plot_metrics(tracker, model_number)
        #     plot_confusion_matrices(train_cm, val_cm, model_number)
    
    # Final plots
    plot_metrics(tracker,model_number)
    plot_confusion_matrices(train_cm, val_cm,model_number)
    
    return tracker
