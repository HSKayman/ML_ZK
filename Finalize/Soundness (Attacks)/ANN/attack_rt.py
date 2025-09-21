# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
data = load_iris()

# %%
# Generate synthetic dataset
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
#                          n_redundant=5, n_classes=3, random_state=42)

X = data.data
y = data.target

# %%
X,y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
class ANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.relu = nn.ReLU()    
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
verified_model = ANN(input_size=4, hidden_sizes=[64, 32], num_classes=3)

# %%
# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(verified_model.parameters(), lr=0.001)

# %%
# Training loop
epochs = 30
for epoch in range(epochs):
    verified_model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = verified_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}')

# %%
# Evaluation
verified_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = verified_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

verified_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = verified_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Train Accuracy: {100 * correct / total:.2f}%')


# %%
activations = {}
def get_activation(name, storage_dict):
    def hook(model, input, output):
        storage_dict[name] = output.detach()
    return hook

verified_model.fc1.register_forward_hook(get_activation('fc1', activations))
verified_model.fc2.register_forward_hook(get_activation('fc2', activations))
verified_model.fc3.register_forward_hook(get_activation('fc3', activations))

# %%
calibration_data = X_train[0]

# Get verified activations from original model
verified_model.eval()
with torch.no_grad():
    _ = verified_model(calibration_data)
    target_activations = {k: v.clone() for k, v in activations.items()}

real_activations = target_activations.copy()

# %% [markdown]
# # ADVERSARY SETUP INVERSE TRANSFORM

# %%
# %%
def crack_input(target_output, model, learning_rate=0.001, iterations=10000, method='pseudo_inverse', return_all_activations=True):
    model.eval()
    
    # Extract model parameters
    W1 = model.fc1.weight.data
    b1 = model.fc1.bias.data
    W2 = model.fc2.weight.data
    b2 = model.fc2.bias.data
    W3 = model.fc3.weight.data
    b3 = model.fc3.bias.data
    
    # Dictionary to store predicted activations
    predicted_activations = {}
    
    if method == 'pseudo_inverse':
        # Layer 3 inverse: output -> h2
        W3_inv = torch.pinverse(W3)
        h2 = W3_inv @ (target_output.squeeze() - b3)
        h2 = torch.clamp(h2, min=0)  # ReLU constraint
        predicted_activations['fc2'] = h2.unsqueeze(0)  # Store fc2 output
        
        # Layer 2 inverse: h2 -> h1
        W2_inv = torch.pinverse(W2)
        h1 = W2_inv @ (h2 - b2)
        h1 = torch.clamp(h1, min=0)  # ReLU constraint
        predicted_activations['fc1'] = h1.unsqueeze(0)  # Store fc1 output
        
        # Layer 1 inverse: h1 -> input
        W1_inv = torch.pinverse(W1)
        x_reconstructed = W1_inv @ (h1 - b1)
        
    elif method == 'svd':
        # Layer 3 inverse using SVD
        U3, S3, V3 = torch.svd(W3)
        S3_inv = torch.where(S3 > 1e-6, 1.0/S3, torch.zeros_like(S3))
        W3_inv = V3 @ torch.diag(S3_inv) @ U3.t()
        h2 = W3_inv @ (target_output.squeeze() - b3)
        h2 = torch.clamp(h2, min=0)
        predicted_activations['fc2'] = h2.unsqueeze(0)
        
        # Layer 2 inverse using SVD
        U2, S2, V2 = torch.svd(W2)
        S2_inv = torch.where(S2 > 1e-6, 1.0/S2, torch.zeros_like(S2))
        W2_inv = V2 @ torch.diag(S2_inv) @ U2.t()
        h1 = W2_inv @ (h2 - b2)
        h1 = torch.clamp(h1, min=0)
        predicted_activations['fc1'] = h1.unsqueeze(0)
        
        # Layer 1 inverse using SVD
        U1, S1, V1 = torch.svd(W1)
        S1_inv = torch.where(S1 > 1e-6, 1.0/S1, torch.zeros_like(S1))
        W1_inv = V1 @ torch.diag(S1_inv) @ U1.t()
        x_reconstructed = W1_inv @ (h1 - b1)
        
    elif method == 'regularized':
        # Regularized inverse (Ridge regression style)
        lambda_reg = 1e-4
        
        # Layer 3 inverse
        W3_reg_inv = torch.inverse(W3.t() @ W3 + lambda_reg * torch.eye(W3.shape[1], device=device)) @ W3.t()
        h2 = W3_reg_inv @ (target_output.squeeze() - b3)
        h2 = torch.clamp(h2, min=0)
        predicted_activations['fc2'] = h2.unsqueeze(0)
        
        # Layer 2 inverse
        W2_reg_inv = torch.inverse(W2.t() @ W2 + lambda_reg * torch.eye(W2.shape[1], device=device)) @ W2.t()
        h1 = W2_reg_inv @ (h2 - b2)
        h1 = torch.clamp(h1, min=0)
        predicted_activations['fc1'] = h1.unsqueeze(0)
        
        # Layer 1 inverse
        W1_reg_inv = torch.inverse(W1.t() @ W1 + lambda_reg * torch.eye(W1.shape[1], device=device)) @ W1.t()
        x_reconstructed = W1_reg_inv @ (h1 - b1)
    
    # Store fc3 output (which is the target output)
    predicted_activations['fc3'] = target_output
    
    x_reconstructed = x_reconstructed.unsqueeze(0)
    
    if return_all_activations:
        return x_reconstructed, predicted_activations
    else:
        return x_reconstructed


# %%
#testing
pred_inputs = crack_input(real_activations['fc3'], verified_model, learning_rate=0.001, iterations=1)

# %%
# %%
ROUND = 2
N_INPUTS = 120
results = pd.DataFrame(columns=[
    'input_id', 'round_id', 
    'fc1_min_abs_diff', 'fc1_max_abs_diff', 'fc1_mean_abs_diff',
    'fc2_min_abs_diff', 'fc2_max_abs_diff', 'fc2_mean_abs_diff',
    'fc3_min_abs_diff', 'fc3_max_abs_diff', 'fc3_mean_abs_diff',
    'all_layers_max_diff', 'all_layers_min_of_max',
    'real_input', 'pred_input', 'inverse_method'
])

for i in range(min(N_INPUTS, len(X_train))):
    print(f"Input {i+1}")
    for j in range(ROUND):
        print(f"Round {j+1}")
        
        # Try different inverse methods
        inverse_method = np.random.choice(['pseudo_inverse', 'svd', 'regularized'])

        # Registering hooks to capture activations
        activations = {}
        hooks = []
        hooks.append(verified_model.fc1.register_forward_hook(get_activation('fc1', activations)))
        hooks.append(verified_model.fc2.register_forward_hook(get_activation('fc2', activations)))
        hooks.append(verified_model.fc3.register_forward_hook(get_activation('fc3', activations)))

        calibration_data = X_train[i]

        # Get verified activations from original model
        verified_model.eval()
        with torch.no_grad():
            _ = verified_model(calibration_data)
            target_activations = {k: v.clone() for k, v in activations.items()}

        real_activations = target_activations.copy()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()

        # INVERSE TRANSFORM INPUT GENERATION WITH PREDICTED ACTIVATIONS
        pred_inputs, pred_activations = crack_input(
            real_activations['fc3'], 
            verified_model, 
            method=inverse_method,
            return_all_activations=True
        )

        round_results = {'input_id': i+1, 'round_id': j+1, 'inverse_method': inverse_method}
        
        # Compare real and predicted activations (from inverse transform)
        all_layer_max_diffs = []
        for layer in real_activations.keys():
            # Calculate differences between real and inverse-predicted activations
            abs_diff = torch.abs(real_activations[layer] - pred_activations[layer])
            mean_abs_diff = abs_diff.mean().item()
            max_abs_diff = abs_diff.max().item()
            min_abs_diff = abs_diff.min().item()

            # Store in results dictionary
            round_results[f'{layer}_min_abs_diff'] = min_abs_diff
            round_results[f'{layer}_max_abs_diff'] = max_abs_diff
            round_results[f'{layer}_mean_abs_diff'] = mean_abs_diff
            
            all_layer_max_diffs.append(max_abs_diff)
        
        # Store the maximum difference across ALL layers
        round_results['all_layers_max_diff'] = max(all_layer_max_diffs)
        round_results['all_layers_min_of_max'] = min(all_layer_max_diffs)

        # Append results to DataFrame
        round_results['real_input'] = X_train[i].cpu().flatten().numpy().tolist()
        round_results['pred_input'] = pred_inputs.cpu().flatten().numpy().tolist()
        results = pd.concat([results, pd.DataFrame([round_results])], ignore_index=True)
        
results.to_csv('activation_diff_results_inverse_transform.csv', index=False)



