# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
from scipy.special import kl_div
import os
from model_structure import get_preprocessing_transforms,get_resnet_model, train_model, evaluate_model

# %%
INPUT_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "pretrained" # "my" or "pretrained"
MODEL_1_DIR = 'best_model_1.pth'
MODEL_2_DIR = 'best_model_2.pth'
K_ROUND = 250  # Number of rounds to monitor
SAVE_FREQUENCY = 250  # Save every 5 steps, adjust as needed
THE_CLASS = 1 # 0 cat or squirrel, 1 dog

# %%
# Get transforms
train_transform, val_transform = get_preprocessing_transforms(INPUT_SIZE)
master_file_path = '.'
test_data_1_dir = f'{master_file_path}/data4model_1/test/'
test_data_2_dir = f'{master_file_path}/data4model_2/test/'
train_data_1_dir = f'{master_file_path}/data4model_1/train/'
train_data_2_dir = f'{master_file_path}/data4model_2/train/'

# Load data set
dataset_test_1 = datasets.ImageFolder(test_data_1_dir,transform=val_transform)
dataset_train_1 = datasets.ImageFolder(train_data_1_dir,transform=val_transform)
dataset_test_2 = datasets.ImageFolder(test_data_2_dir,transform=val_transform)
dataset_train_2 = datasets.ImageFolder(train_data_2_dir,transform=val_transform)
additional_set = datasets.ImageFolder(f'{master_file_path}/data4model_1/for_extra_test/',transform=val_transform)

additional_loader = DataLoader(additional_set, shuffle=False, batch_size=BATCH_SIZE)
test_loader_1 = DataLoader(dataset_test_1, shuffle=False, batch_size=BATCH_SIZE)
train_loader_1 = DataLoader(dataset_train_1, shuffle=False, batch_size=BATCH_SIZE)
test_loader_2 = DataLoader(dataset_test_2, shuffle=False, batch_size=BATCH_SIZE)
train_loader_2 = DataLoader(dataset_train_2, shuffle=False, batch_size=BATCH_SIZE)

dataSets =  {"Model_1:Train": train_loader_1,
        "Model_1:Test": test_loader_1,
        "Model_2:Train": train_loader_2,
        "Model_2:Test": test_loader_2,
        "Model_1:additional_set": additional_loader
        }

# %%
# Load the model
# model_1 = BCNN().to(DEVICE)
model_1 = get_resnet_model(2).to(DEVICE)
weights = torch.load(MODEL_1_DIR, map_location=torch.device(DEVICE))
model_1.load_state_dict(weights)
model_1.eval()

#model_2 = BCNN().to(DEVICE)
model_2 = get_resnet_model(2).to(DEVICE)
weights = torch.load(MODEL_2_DIR, map_location=torch.device(DEVICE))
model_2.load_state_dict(weights)
model_2.eval()

# %%
for key in dataSets:
    print(key)
    if "Model_1" in key: # Verified Model
        model = model_1
    else:
        model = model_2
    
    results = evaluate_model(model.to(DEVICE), dataSets[key], DEVICE)
    print("Accuracy: ", results[0])
    print("F1 Score: ", results[1])
    print("confusion Matrix: \n",results[3])
    print("\n")

# %% [markdown]
# # TEST STARTS HERE

# %%
activation_values = {"Model_1": {}, 
                     "Model_2": {}}

# Register hooks to capture activation values for each layer
# This function will register hooks to capture the input, output, weights, 
# and biases of each layer in the model and store them in the `activation_values` dictionary.
# Pytorch backend will call this function when the model is run in the forward pass.
def register_hooks(model, model_name):
    activation_values[model_name] = {}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            # Store input, output, weights and biases for each layer with keeping their shape 
            activation_values[model_name][name] = {
                'input': input[0].detach(),
                'output': output.detach(),
                'weight': module.weight.detach() if hasattr(module, 'weight') else None,
                'bias': module.bias.detach() if hasattr(module, 'bias') and module.bias is not None else None,
            }
            if isinstance(module, nn.Conv2d):   # For Conv2d layers, also store kernel size, padding, stride, and dilation because 
                                                # This information to calculate contributions of each neuron in the layer required.
                activation_values[model_name][name]['conv_params'] = {
                    'kernel_size': module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                    'padding': module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                    'stride': module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                    'dilation': module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
                }
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)): # Just register hooks for Conv2d and Linear layers not for all layers.
            hooks.append(module.register_forward_hook(get_hook(name)))
    
    return hooks

# %%
def select_random_neuron_indices(activations, n_neurons=1,randomness=False):
    indices = {}
    
    for layer_name, layer_data in activations.items():
        # skipping if layer without weight such as ReLU, Dropout, etc.
        if not isinstance(layer_data, dict) or 'output' not in layer_data:
            continue
            
        # activation output and flatten it
        activation = layer_data['output']
        flattened = activation.view(activation.shape[0], -1)
        flattened_np = flattened.cpu().numpy().squeeze()
        # positive (activated) indices
        positive_indices = np.where(flattened_np > 0)[0]
        
        # skipping if no activated neurons
        if len(positive_indices) == 0:
            continue

        if randomness:    
            # random neurons from activated ones
            selected_indices = np.random.choice(
                positive_indices,
                size=min(n_neurons, len(positive_indices)),
                replace=False
            )
        else:
            positive_values = flattened_np[positive_indices]
            min_value_idx = positive_indices[np.argmin(positive_values)]
            
            # Select the minimum value index
            selected_indices = np.array([min_value_idx])
        
        # storing selected indices
        indices[layer_name] = {
            'neuron_idx': selected_indices,
            'all_activated_indices': positive_indices,
            'original_shape': activation.shape
        }
    
    return indices

# %%
def compare_model_activations(model1_activations, model2_activations, selected_indices):
    comparison_results = {}
    
    for layer_name, indices_info in selected_indices.items():
        if layer_name not in model1_activations or layer_name not in model2_activations:
            continue
        
        data1 = model1_activations[layer_name]
        data2 = model2_activations[layer_name]
        
        if not isinstance(data1, dict) or not isinstance(data2, dict):
            continue
        
        selected_indices = indices_info['neuron_idx']
        original_shape = indices_info.get('original_shape', data1['output'].shape)
        
        # get inputs, weights, biases, and stored outputs
        # CHANGED: Now comparing different inputs with same weights
        input1 = data1['input']  # Input from model 1
        input2 = data2['input']  # Input from model 2
        weights1 = data1['weight']  # Using same weights (model 1)
        bias1 = data1.get('bias')
        stored_output1 = data1['output']
    
        layer_results = {
            'neuron_comparisons': [],
            'model1_verification_errors': [],
            'cross_model_differences': [],
            'model1_activation_value':[],
            'model2_activation_value':[]
        }
        
        for idx in selected_indices:
            if len(weights1.shape) == 4:
                # dimensions: batch_size, channels, height, width
                batch_size, channels, height, width = original_shape
                
                # converting flat idx to corresponding coordinates
                channel = idx // (height * width)
                pos_in_channel = idx % (height * width)
                h_idx = pos_in_channel // width
                w_idx = pos_in_channel % width
                
                # get stored activation for this position
                stored_activation = stored_output1[0, channel, h_idx, w_idx].item()
                
                # get convolution parameters
                kernel_size = weights1.shape[2]
                padding = kernel_size // 2  # symmetric padding
                
                # get input patch that corresponds to this position
                conv_params = data1.get('conv_params', {
                    'kernel_size': 3, 
                    'padding': 1,
                    'stride': 1,
                    'dilation': 1
                })
                
                kernel_size = conv_params['kernel_size']
                padding = conv_params['padding'] 
                stride = conv_params['stride']
                dilation = conv_params['dilation']
                
                # Calculate input patch position
                # With padding=1, we need to adjust position getting correct patch
                in_h_start = h_idx * stride - padding
                in_w_start = w_idx * stride - padding
                in_h_end = in_h_start + kernel_size
                in_w_end = in_w_start + kernel_size
                
                # Extract input patch accounting for padding
                input_patch = torch.zeros(input1.shape[1], kernel_size, kernel_size, device=input1.device)
                
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        h_pos = in_h_start + i * dilation
                        w_pos = in_w_start + j * dilation
                        
                        # If position is within bounds of input
                        if 0 <= h_pos < input1.shape[2] and 0 <= w_pos < input1.shape[3]:
                            input_patch[:, i, j] = input1[0, :, h_pos, w_pos]

                # Extract input patch from model 2
                input_patch2 = torch.zeros(input2.shape[1], kernel_size, kernel_size, device=input2.device)
                
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        h_pos = in_h_start + i * dilation
                        w_pos = in_w_start + j * dilation
                        
                        # If position is within bounds of input
                        if 0 <= h_pos < input2.shape[2] and 0 <= w_pos < input2.shape[3]:
                            input_patch2[:, i, j] = input2[0, :, h_pos, w_pos]
                
                # Compute activation with model1 input and model1 weights
                if channel < weights1.shape[0]:
                    kernel1 = weights1[channel]
                    # Classic convolution calculation
                    calculated_activation1 = (input_patch * kernel1).sum()
                    
                    if bias1 is not None:
                        calculated_activation1 += bias1[channel]

                    calculated_activation1 = max(0, calculated_activation1.item())  # ReLU
                else:
                    calculated_activation1 = 0
                
                # Compute activation using model2 input with same model1 weights
                if channel < weights1.shape[0]:
                    kernel1 = weights1[channel]
                    calculated_activation2 = (input_patch2 * kernel1).sum()
                    if bias1 is not None:
                        calculated_activation2 += bias1[channel]
                    calculated_activation2 = max(0, calculated_activation2.item())  # ReLU
                else:
                    calculated_activation2 = 0
                
                neuron_data = {
                    'type': 'conv',
                    'position': {'channel': int(channel), 'height': int(h_idx), 'width': int(w_idx)},
                    'input_patch_model1': {
                        'data': input_patch.detach().cpu().numpy(),
                        'coords': {'h_start': in_h_start, 'h_end': in_h_end, 
                                  'w_start': in_w_start, 'w_end': in_w_end}
                    },
                    'input_patch_model2': {
                        'data': input_patch2.detach().cpu().numpy(),
                        'coords': {'h_start': in_h_start, 'h_end': in_h_end, 
                                  'w_start': in_w_start, 'w_end': in_w_end}
                    }
                }
            else:
                
                flattened_output = stored_output1.view(stored_output1.shape[0], -1)
                
                # check if it is correct when i remove this
                if idx >= flattened_output.shape[1] or idx >= weights1.shape[0]:
                    continue
                
                # get stored activation
                stored_activation = flattened_output[0, idx].item()
                
                # unrolling input for dot product
                flattened_input1 = input1.flatten(start_dim=1)
                flattened_input2 = input2.flatten(start_dim=1)
                
                # check if it is correct when i remove this
                if flattened_input1.shape[1] != weights1.shape[1]:
                    continue
                
                # calculate model1 activation using input1 and weights1
                calculated_activation1 = torch.matmul(flattened_input1, weights1[idx])
                if bias1 is not None and idx < bias1.shape[0]:
                    calculated_activation1 += bias1[idx]
                calculated_activation1 = max(0, calculated_activation1.item())  # ReLU
                
                # Calculate using model2 input with same model1 weights
                calculated_activation2 = torch.matmul(flattened_input2, weights1[idx])
                if bias1 is not None and idx < bias1.shape[0]:
                    calculated_activation2 += bias1[idx]
                calculated_activation2 = max(0, calculated_activation2.item())  # ReLU
                
                neuron_data = {
                    'type': 'linear',
                    'position': {'neuron': int(idx)},
                    'input_values_model1': flattened_input1.detach().cpu().numpy(),
                    'input_values_model2': flattened_input2.detach().cpu().numpy()
                }
            
            # Calculate verification error (how accurate our calculation is for model1's input)
            verification_error = abs(stored_activation - calculated_activation1)
            
            # Calculate cross-input difference (same weights, different inputs)
            cross_model_difference = abs(calculated_activation1 - calculated_activation2)
            
            # Store comparison for this neuron
            neuron_comparison = {
                'neuron_idx': int(idx),
                'stored_activation': stored_activation,
                'calculated_activation1': calculated_activation1,
                'calculated_activation2': calculated_activation2,
                'verification_error': verification_error,
                'cross_model_difference': cross_model_difference,
                'neuron_data': neuron_data
            }
            
            layer_results['neuron_comparisons'].append(neuron_comparison)
            layer_results['model1_verification_errors'].append(verification_error)
            layer_results['cross_model_differences'].append(cross_model_difference)
            layer_results['model1_activation_value'].append(stored_activation)
            layer_results['model2_activation_value'].append(calculated_activation2)
        
        # Calculate statistics
        if layer_results['neuron_comparisons']:
            layer_results['mean_verification_error'] = sum(layer_results['model1_verification_errors']) / len(layer_results['model1_verification_errors'])
            layer_results['mean_cross_model_difference'] = sum(layer_results['cross_model_differences']) / len(layer_results['cross_model_differences'])
        else:
            layer_results['mean_verification_error'] = float('nan')
            layer_results['mean_cross_model_difference'] = float('nan')
        
        comparison_results[layer_name] = layer_results
    
    return comparison_results

# %%
images = [(img, idx) for idx, (img, label) in enumerate(dataSets["Model_1:additional_set"].dataset) 
            if label == THE_CLASS  and model_1(img.unsqueeze(0).to(DEVICE))
                                          .argmax().item() == THE_CLASS]


# %%
activation_values = {"Model_1":{},
                    "Model_2":{}}
CLASSES = { 
    0: 'cat',
    1: 'dog'
}

# %%
layer_stats_columns = [
    'image_id', 'round', 'layer_name', 
    'mean_verification_error', 'mean_cross_model_difference', 'model1_activation_value', 'model2_activation_value'
]

output_file = f'{MODEL_NAME}_model_random_paths_for_{CLASSES[THE_CLASS]}_and_{K_ROUND}_rounds.csv'

try:
    existing_df = pd.read_csv(output_file)
    last_image = existing_df['image_id'].max()
    last_round = existing_df[existing_df['image_id'] == last_image]['round'].max()
    print(f"Resuming from image {last_image}, round {last_round}")
    file_exists = True
except FileNotFoundError:
    # Create a new file with headers
    with open(output_file, 'w') as f:
        f.write(','.join(layer_stats_columns) + '\n')
    last_image = -1
    last_round = -1
    file_exists = True
    print("Starting new run")


results_buffer = []
for image, pic_index in images:
    # Skip already processed images
    if pic_index < last_image:
        continue
        
    # register hooks
    hooks1 = register_hooks(model_1, "Model_1")
    hooks2 = register_hooks(model_2, "Model_2")
    
    # clearing
    activation_values["Model_1"] = {}
    activation_values["Model_2"] = {}
    
    # Forward pass
    with torch.no_grad():
        pred1 = model_1(image.unsqueeze(0).to(DEVICE))
        pred2 = model_2(image.unsqueeze(0).to(DEVICE))

    for i in range(K_ROUND):
        # Skip already processed rounds for the last image
        if pic_index == last_image and i <= last_round:
            continue

        # Select random neurons and compare
        selected_indices = select_random_neuron_indices(
            activation_values["Model_1"], 
            n_neurons=1,
            randomness=True
        )
        result = compare_model_activations(
            activation_values["Model_1"], 
            activation_values["Model_2"], 
            selected_indices
        )
        
        # Process results for each layer
        for layer_name, layer_data in result.items():
            results_buffer.append({
                'image_id': pic_index,
                'round': i,
                'layer_name': layer_name,
                'mean_verification_error': layer_data.get('mean_verification_error', np.nan),
                'mean_cross_model_difference': layer_data.get('mean_cross_model_difference', np.nan),
                'model1_activation_value': layer_data.get('model1_activation_value', np.nan),
                'model2_activation_value': layer_data.get('model2_activation_value', np.nan)
            })
        
        print('\r ',f"pic_index: {pic_index:02d}", f"round: {i:02d}",end="\n")
            
        if len(results_buffer) >= SAVE_FREQUENCY:
            temp_df = pd.DataFrame(results_buffer)
            temp_df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"Appended batch at image {pic_index}, round {i}")
            # Clear buffer
            results_buffer = []
    # Remove hooks
    for hook in hooks1 + hooks2:
        hook.remove()

# Save any remaining results in the buffer
if results_buffer:
    temp_df = pd.DataFrame(results_buffer)
    temp_df.to_csv(output_file, mode='a', header=False, index=False)
print("Processing complete. Final results saved.")



