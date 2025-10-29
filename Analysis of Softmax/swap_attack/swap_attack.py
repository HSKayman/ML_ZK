# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import pickle

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

# %%
# Load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

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
def create_malicious_output(tokenizer, original_logits):
    # the very last token in the sequence
    last_token_logits = original_logits[0, -1, :].clone()
    
    # Find index of highest logit
    correct_token_idx = torch.argmax(last_token_logits).item()
    
    # a very unlikely token
    incorrect_token_idx = torch.argmin(last_token_logits).item()
    
    print("--- Logit Swap Attack ---")
    print(f"Original top prediction: '{tokenizer.decode(correct_token_idx)}' (ID: {correct_token_idx})")
    print(f"Target swap token:     '{tokenizer.decode(incorrect_token_idx)}' (ID: {incorrect_token_idx})")
    
    # malicious target by swapping the values
    malicious_target_logits = last_token_logits.clone()
    correct_value = malicious_target_logits[correct_token_idx]
    incorrect_value = malicious_target_logits[incorrect_token_idx]
    
    malicious_target_logits[correct_token_idx] = incorrect_value
    malicious_target_logits[incorrect_token_idx] = correct_value
    
    print(f"New top prediction after swap: '{tokenizer.decode(torch.argmax(malicious_target_logits))}'\n")
    
    return malicious_target_logits.detach()
# %%
# Generate sample inputs for analysis
def generate_sample_inputs(tokenizer, seq_length=8):    
    sample_texts = [
        "The capital of France is",
        "The largest mammal on Earth is",
        "The process of photosynthesis occurs in"
    ]
    
    inputs = []
    for i in range(len(sample_texts)):
        tokenized = tokenizer(
            sample_texts[i], 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=seq_length
        )
        inputs.append(tokenized.input_ids.to(model.device))
    
    return inputs

# %%
def generate_activation_differences_llama(model, X_data, n_samples=5, n_reconstructions=3):
    results = []
    
    layer_names = [f'model.layers.{i}' for i in range(len(model.model.layers))]
    
    for sample_idx in tqdm(range(min(n_samples, len(X_data))), desc="Processing samples"):
        original_input = X_data[sample_idx]
        
        # Get original activations
        original_activations = {}
        hooks = []
        for layer_name in layer_names:
            layer_module = model.get_submodule(layer_name)
            hooks.append(layer_module.register_forward_hook(get_activation(layer_name, original_activations)))
        
        with torch.no_grad():
            original_output = model(original_input).logits
        
        for hook in hooks:
            hook.remove()
        
        # --- ATTACK STEP ---
        # Create the malicious target for reconstruction
        malicious_target_logits = create_malicious_output(tokenizer, original_output)
        
        # Multiple reconstruction attempts
        for recon_idx in range(n_reconstructions):
            seq_length = original_input.shape[1]
            embedding_dim = model.config.hidden_size
            
            reconstructed_embeddings = torch.randn(
                1, seq_length, embedding_dim,
                device=model.device,
                dtype=torch.float32,
                requires_grad=True
            )
            
            optimizer = optim.Adam([reconstructed_embeddings], lr=0.01)
            
            # Reconstruction optimization
            for iteration in tqdm(range(5000), desc=f"Recon {recon_idx+1}/{n_reconstructions}", leave=False): 
                optimizer.zero_grad()
                
                embeddings_model_dtype = reconstructed_embeddings.to(model.dtype)
                output = model(inputs_embeds=embeddings_model_dtype).logits
                
                # MODIFIED: Loss now matches the MALICIOUS target
                loss = nn.functional.mse_loss(output.float(), malicious_target_logits.float())
                
                reg_loss = 0.001 * torch.mean(reconstructed_embeddings ** 2)
                total_loss = loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                if total_loss.item() < 1e-4:
                    break
            
            print(f"\nSample {sample_idx}, Recon {recon_idx}, Final Loss: {total_loss.item():.6f}")

            # Get reconstructed activations
            reconstructed_activations = {}
            hooks = []
            for layer_name in layer_names:
                layer_module = model.get_submodule(layer_name)
                hooks.append(layer_module.register_forward_hook(get_activation(layer_name, reconstructed_activations)))
            
            with torch.no_grad():
                embeddings_model_dtype = reconstructed_embeddings.to(model.dtype)
                _ = model(inputs_embeds=embeddings_model_dtype)
            
            for hook in hooks:
                hook.remove()
            
            # Calculate differences for each layer
            row = {'sample_idx': sample_idx, 'reconstruction_idx': recon_idx}
            all_layer_max_diffs = []
            
            for layer_name in layer_names:
                if layer_name in original_activations and layer_name in reconstructed_activations:
                    orig_act = original_activations[layer_name].flatten().float()
                    recon_act = reconstructed_activations[layer_name].flatten().float()
                    
                    abs_diff = torch.abs(orig_act - recon_act)
                    
                    layer_num = layer_name.split('.')[-1]
                    row[f'layer_{layer_num}_min_abs_diff'] = abs_diff.min().item()
                    row[f'layer_{layer_num}_mean_abs_diff'] = abs_diff.mean().item()
                    row[f'layer_{layer_num}_max_abs_diff'] = abs_diff.max().item()
                    
                    all_layer_max_diffs.append(abs_diff.max().item())
            
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
# Using fewer samples/reconstructions for a quicker demonstration
results_df = generate_activation_differences_llama(model, X_data, n_samples=3, n_reconstructions=3)

# %%
# Save results
results_df.to_csv('llama2_swap_attack_results.csv', index=False)
print(f"\nResults saved to 'llama2_swap_attack_results.csv'. Shape: {results_df.shape}")
print("\nFirst few rows:")
print(results_df.head())

# %%



