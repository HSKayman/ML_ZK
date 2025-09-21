# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pickle

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

# %%
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map='auto' if torch.cuda.is_available() else None
)
model.eval()


# %%
model.parameters

# %%
# Helper function to capture activations
def get_activation(name, activations_dict):
    def hook(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            activations_dict[name] = output[0].detach().clone()
        else:
            activations_dict[name] = output.detach().clone()
    return hook

# %%
# Generate sample inputs for analysis
def generate_sample_inputs(tokenizer, seq_length=32):    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world of technology.",
        "In a hole in the ground there lived a hobbit.",
        "To be or not to be, that is the question Shakespeare posed.",
        "Machine learning models require large datasets for training.",
        "The mitochondria is the powerhouse of the cell in biology.",
        "Climate change is causing unprecedented shifts in global weather patterns.",
        "Mozart composed his first symphony at the age of eight years old.",
        "The stock market experienced significant volatility during the pandemic crisis.",
        "Quantum physics reveals the strange behavior of particles at subatomic levels.",
        "Professional chefs recommend using fresh herbs to enhance flavor profiles.",
        "Ancient Egyptian pyramids were built using sophisticated engineering techniques.",
        "Regular exercise and proper nutrition are essential for maintaining good health.",
        "The International Space Station orbits Earth approximately every ninety minutes.",
        "Cryptocurrency markets operate twenty-four hours a day across global exchanges.",
        "Vincent van Gogh painted Starry Night while staying at an asylum.",
        "Professional athletes must maintain strict training regimens throughout their careers.",
        "The Amazon rainforest produces twenty percent of the world's oxygen supply.",
        "Modern architecture emphasizes clean lines and functional design principles.",
        "Forensic scientists use DNA analysis to solve complex criminal investigations.",
        "Traditional Japanese tea ceremonies follow centuries-old ritualistic practices.",
        "Marine biologists study coral reef ecosystems threatened by ocean acidification.",
        "The Renaissance period marked a cultural rebirth in European art and science.",
        "Cybersecurity experts work tirelessly to protect digital infrastructure from threats.",
        "Sustainable agriculture practices help preserve soil quality for future generations."
    ]
    
    inputs = []
    for i in range(len(sample_texts)):
        # Cycle through sample texts and add variations
        base_text = sample_texts[i % len(sample_texts)]

        # Tokenize
        tokenized = tokenizer(
            base_text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=seq_length
        )
        inputs.append(tokenized.input_ids.to(model.device))
    
    return inputs

# %%
# Generate activation differences for Llama-2 layers
def generate_activation_differences_llama(model, X_data, n_samples=50, n_reconstructions=3):
    results = []
    
    # Select specific layers to analyze (first few transformer layers)
    # layer_names = [
    #     'model.layers.0',  # First transformer layer
    #     'model.layers.1',  # Second transformer layer  
    #     'model.layers.2',  # Third transformer layer
    # ]
    layer_names = [f'model.layers.{i}' for i in range(len(model.model.layers))]
    for sample_idx in tqdm(range(min(n_samples, len(X_data))), desc="Processing samples"):
        original_input = X_data[sample_idx]
        
        # Get original activations
        original_activations = {}
        hooks = []
        
       # Register hooks for all layers
        for layer_name in layer_names:
            layer_module = model
            for attr in layer_name.split('.'):
                layer_module = getattr(layer_module, attr)
            hooks.append(layer_module.register_forward_hook(
                get_activation(layer_name, original_activations)
            ))
        
        # Get original output and activations
        with torch.no_grad():
            original_output = model(original_input).logits
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Multiple reconstruction attempts
        for recon_idx in range(n_reconstructions):
            # Initialize random input embeddings for reconstruction
            seq_length = original_input.shape[1]
            embedding_dim = model.config.hidden_size
            
            # Use embeddings instead of token IDs for gradient-based optimization
            reconstructed_embeddings = torch.randn(
                1, seq_length, embedding_dim,
                device=model.device,
                dtype=torch.float32,
                requires_grad=True
            )
            
            optimizer = optim.Adam([reconstructed_embeddings], lr=0.001)
            
            # Reconstruction optimization
            for iteration in range(10000): 
                optimizer.zero_grad()
                
                # Forward pass with embeddings
                embeddings_model_dtype = reconstructed_embeddings.to(model.dtype)
                output = model(inputs_embeds=embeddings_model_dtype).logits
                
                # Loss: match original output
                loss = nn.functional.mse_loss(output.float(), original_output.float())
                
                # Add regularization
                reg_loss = 0.001 * torch.mean(reconstructed_embeddings ** 2)
                total_loss = loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                if total_loss.item() < 1e-3:
                    break
                if iteration % 1000 == 0:
                    print(f"Sample {sample_idx}, Recon {recon_idx}, Iter {iteration}, Loss: {total_loss.item():.6f}",end="")
            
            # Get reconstructed activations
            reconstructed_activations = {}
            hooks = []
            
            for layer_name in layer_names:
                layer_module = model
                for attr in layer_name.split('.'):
                    layer_module = getattr(layer_module, attr)
                hooks.append(layer_module.register_forward_hook(
                    get_activation(layer_name, reconstructed_activations)
                ))
            
            with torch.no_grad():
                embeddings_model_dtype = reconstructed_embeddings.to(model.dtype)
                _ = model(inputs_embeds=embeddings_model_dtype)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Calculate differences for each layer
            row = {'sample_idx': sample_idx, 'reconstruction_idx': recon_idx}
            
            # Store individual layer metrics
            all_layer_max_diffs = []
            
            for layer_name in layer_names:
                if layer_name in original_activations and layer_name in reconstructed_activations:
                    orig_act = original_activations[layer_name].flatten().float()
                    recon_act = reconstructed_activations[layer_name].flatten().float()
                    
                    abs_diff = torch.abs(orig_act - recon_act)
                    
                    layer_short = layer_name.split('.')[-1]  # Get layer number
                    row[f'layer_{layer_short}_min_abs_diff'] = abs_diff.min().item()
                    row[f'layer_{layer_short}_mean_abs_diff'] = abs_diff.mean().item()
                    row[f'layer_{layer_short}_max_abs_diff'] = abs_diff.max().item()
                    
                    all_layer_max_diffs.append(abs_diff.max().item())
            
            # Store the maximum difference across ALL layers
            if all_layer_max_diffs:
                row['all_layers_max_diff'] = max(all_layer_max_diffs)
                row['all_layers_min_of_max'] = min(all_layer_max_diffs)
            
            results.append(row)
    
    return pd.DataFrame(results)

# %%
# Generate sample data
print("Generating sample inputs...")
X_data = generate_sample_inputs(tokenizer, seq_length=15) 
print(f"Generated {len(X_data)} samples")

# %%
# Generate results
print("Generating activation differences for Llama-2 layers...")
results = generate_activation_differences_llama(model, X_data, n_samples=25, n_reconstructions=25)
#406m 40.9s

# %%
# Save results
results.to_csv('llama2_activation_diff_results.csv', index=False)
print(f"Results saved. Shape: {results.shape}")
print("\nFirst few rows:")
print(results.head())


