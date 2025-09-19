# %%
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from typing import Dict, List, Tuple, Optional
import json


# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_1_PATH = "meta-llama/Llama-2-7b-chat-hf" 
MODEL_2_PATH = "meta-llama/Llama-2-7b-hf"       

print(DEVICE)

# %%
tokenizer = LlamaTokenizer.from_pretrained(MODEL_1_PATH)
tokenizer.pad_token = tokenizer.eos_token

model_1 = LlamaForCausalLM.from_pretrained(
    MODEL_1_PATH,
    torch_dtype=torch.float32, 
    device_map=DEVICE
)

model_2 = LlamaForCausalLM.from_pretrained(
    MODEL_2_PATH,
    torch_dtype=torch.float32, 
    device_map=DEVICE
)

# %%
model_1 = model_1.to(DEVICE)
model_2 = model_2.to(DEVICE)

# %%
def get_model_weights(model, move_to_cpu=False,layer_range=None):
    weights = {}
    
    # Embedding weights
    embed_weight = model.model.embed_tokens.weight
    weights['embed_tokens'] = embed_weight.detach().cpu() if move_to_cpu else embed_weight.detach()
    
    # Layer-specific weights
    if layer_range is None:
        layer_range = range(len(model.model.layers))
    
    for i in layer_range:
        if i >= len(model.model.layers):
            continue
            
        layer = model.model.layers[i]
        layer_prefix = f"layer_{i}"
        
        # Self-attention weights
        weights[f"{layer_prefix}_q_proj"] = layer.self_attn.q_proj.weight.detach().cpu() if move_to_cpu else layer.self_attn.q_proj.weight.detach()
        weights[f"{layer_prefix}_k_proj"] = layer.self_attn.k_proj.weight.detach().cpu() if move_to_cpu else layer.self_attn.k_proj.weight.detach()
        weights[f"{layer_prefix}_v_proj"] = layer.self_attn.v_proj.weight.detach().cpu() if move_to_cpu else layer.self_attn.v_proj.weight.detach()
        weights[f"{layer_prefix}_o_proj"] = layer.self_attn.o_proj.weight.detach().cpu() if move_to_cpu else layer.self_attn.o_proj.weight.detach()
        
        # MLP weights
        weights[f"{layer_prefix}_gate_proj"] = layer.mlp.gate_proj.weight.detach().cpu() if move_to_cpu else layer.mlp.gate_proj.weight.detach()
        weights[f"{layer_prefix}_up_proj"] = layer.mlp.up_proj.weight.detach().cpu() if move_to_cpu else layer.mlp.up_proj.weight.detach()
        weights[f"{layer_prefix}_down_proj"] = layer.mlp.down_proj.weight.detach().cpu() if move_to_cpu else layer.mlp.down_proj.weight.detach()
        
        # Layer norm weights
        weights[f"{layer_prefix}_input_layernorm"] = layer.input_layernorm.weight.detach().cpu() if move_to_cpu else layer.input_layernorm.weight.detach()
        weights[f"{layer_prefix}_post_attention_layernorm"] = layer.post_attention_layernorm.weight.detach().cpu() if move_to_cpu else layer.post_attention_layernorm.weight.detach()
    
    # Final layer norm and LM head
    weights['final_norm'] = model.model.norm.weight.detach().cpu() if move_to_cpu else model.model.norm.weight.detach()
    weights['lm_head'] = model.lm_head.weight.detach().cpu() if move_to_cpu else model.lm_head.weight.detach()
    
    return weights

def calculate_weight_differences(weights_1, weights_2):
    differences = {}
    
    common_keys = set(weights_1.keys()) & set(weights_2.keys())
    print(f"Comparing {len(common_keys)} weight matrices...")
    
    for i, key in enumerate(common_keys):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(common_keys)}: {key}")
            
        w1 = weights_1[key]
        w2 = weights_2[key]
        
        if w1.shape != w2.shape:
            print(f"Warning: Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
            continue
        
        # Calculate difference matrix
        diff_matrix = w1 - w2
        
        # Calculate various norms and statistics
        frobenius_norm = torch.norm(diff_matrix, p='fro').item()
        frobenius_norm_relative = frobenius_norm / (torch.norm(w1, p='fro').item() + 1e-10)
        
        spectral_norm = torch.norm(diff_matrix, p=2).item()
        spectral_norm_relative = spectral_norm / (torch.norm(w1, p=2).item() + 1e-10)
        
        # Element-wise statistics
        abs_diff = torch.abs(diff_matrix)
        mean_abs_diff = torch.mean(abs_diff).item()
        max_abs_diff = torch.max(abs_diff).item()
        std_diff = torch.std(diff_matrix).item()
        
        # Percentage of significantly different weights (threshold = 1e-6)
        significant_diff_ratio = (abs_diff > 1e-6).float().mean().item()
        
        # Cosine similarity
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()
        cosine_sim = F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()
        
        differences[key] = {
            'frobenius_norm': frobenius_norm,
            'frobenius_norm_relative': frobenius_norm_relative,
            'spectral_norm': spectral_norm,
            'spectral_norm_relative': spectral_norm_relative,
            'mean_abs_difference': mean_abs_diff,
            'max_abs_difference': max_abs_diff,
            'std_difference': std_diff,
            'significant_diff_ratio': significant_diff_ratio,
            'cosine_similarity': cosine_sim,
            'weight_shape': w1.shape,
            'total_parameters': w1.numel()
        }
    
    return differences

def analyze_weight_patterns(weight_differences):
    analysis = {
        'by_component_type': defaultdict(list),
        'by_layer_depth': defaultdict(list),
        'summary_stats': {}
    }
    
    # Group by component type
    for layer_name, diff_data in weight_differences.items():
        if any(x in layer_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            component_type = 'attention'
        elif any(x in layer_name for x in ['gate_proj', 'up_proj', 'down_proj']):
            component_type = 'mlp'
        elif 'layernorm' in layer_name or 'norm' in layer_name:
            component_type = 'normalization'
        elif 'embed' in layer_name:
            component_type = 'embedding'
        elif 'lm_head' in layer_name:
            component_type = 'output'
        else:
            component_type = 'other'
        
        analysis['by_component_type'][component_type].append({
            'layer_name': layer_name,
            'frobenius_norm': diff_data['frobenius_norm'],
            'frobenius_norm_relative': diff_data['frobenius_norm_relative'],
            'significant_diff_ratio': diff_data['significant_diff_ratio'],
            'cosine_similarity': diff_data['cosine_similarity']
        })
    
    # Group by layer depth
    for layer_name, diff_data in weight_differences.items():
        if 'layer_' in layer_name:
            try:
                layer_num = int(layer_name.split('_')[1])
                analysis['by_layer_depth'][layer_num].append({
                    'layer_name': layer_name,
                    'frobenius_norm': diff_data['frobenius_norm'],
                    'frobenius_norm_relative': diff_data['frobenius_norm_relative'],
                    'cosine_similarity': diff_data['cosine_similarity']
                })
            except:
                continue
    
    # Calculate summary statistics
    all_frobenius = [data['frobenius_norm'] for data in weight_differences.values()]
    all_frobenius_rel = [data['frobenius_norm_relative'] for data in weight_differences.values()]
    all_significant_ratios = [data['significant_diff_ratio'] for data in weight_differences.values()]
    all_cosine_sims = [data['cosine_similarity'] for data in weight_differences.values()]
    
    analysis['summary_stats'] = {
        'total_layers_compared': len(weight_differences),
        'mean_frobenius_norm': np.mean(all_frobenius),
        'std_frobenius_norm': np.std(all_frobenius),
        'max_frobenius_norm': np.max(all_frobenius),
        'min_frobenius_norm': np.min(all_frobenius),
        'mean_frobenius_norm_relative': np.mean(all_frobenius_rel),
        'mean_significant_diff_ratio': np.mean(all_significant_ratios),
        'mean_cosine_similarity': np.mean(all_cosine_sims),
        'min_cosine_similarity': np.min(all_cosine_sims),
        'total_parameters_compared': sum(data['total_parameters'] for data in weight_differences.values())
    }
    
    return analysis

def print_weight_analysis_summary(analysis):
    print("="*70)
    print("LLAMA MODEL WEIGHT DIFFERENCE ANALYSIS SUMMARY")
    print("="*70)
    
    # Overall statistics
    stats = analysis['summary_stats']
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  • Total layers compared: {stats['total_layers_compared']}")
    print(f"  • Total parameters compared: {stats['total_parameters_compared']:,}")
    print(f"  • Mean Frobenius norm: {stats['mean_frobenius_norm']:.2e}")
    print(f"  • Mean relative Frobenius norm: {stats['mean_frobenius_norm_relative']:.8f}")
    print(f"  • Max Frobenius norm: {stats['max_frobenius_norm']:.2e}")
    print(f"  • Min Frobenius norm: {stats['min_frobenius_norm']:.2e}")
    print(f"  • Mean cosine similarity: {stats['mean_cosine_similarity']:.8f}")
    print(f"  • Min cosine similarity: {stats['min_cosine_similarity']:.8f}")
    print(f"  • Mean significant difference ratio: {stats['mean_significant_diff_ratio']:.4f}")
    
    # Component type analysis
    print(f"\n🔧 BY COMPONENT TYPE:")
    for comp_type, comp_data in analysis['by_component_type'].items():
        frob_norms = [item['frobenius_norm_relative'] for item in comp_data]
        cosine_sims = [item['cosine_similarity'] for item in comp_data]
        sig_ratios = [item['significant_diff_ratio'] for item in comp_data]
        
        print(f"  {comp_type.upper()}:")
        print(f"    - Count: {len(comp_data)} layers")
        print(f"    - Mean relative Frobenius: {np.mean(frob_norms):.8f} ± {np.std(frob_norms):.8f}")
        print(f"    - Mean cosine similarity: {np.mean(cosine_sims):.8f} ± {np.std(cosine_sims):.8f}")
        print(f"    - Mean sig. diff ratio: {np.mean(sig_ratios):.4f}")
    
    # Layer depth analysis (if available)
    if analysis['by_layer_depth']:
        print(f"\n📈 BY LAYER DEPTH:")
        for depth in sorted(analysis['by_layer_depth'].keys())[:10]:  # Show first 10 layers
            depth_data = analysis['by_layer_depth'][depth]
            frob_norms = [item['frobenius_norm_relative'] for item in depth_data]
            cosine_sims = [item['cosine_similarity'] for item in depth_data]
            
            print(f"  Layer {depth}: Frob={np.mean(frob_norms):.6f}, Cosine={np.mean(cosine_sims):.6f}")
    
    print("="*70)

# %%
weights_1 = get_model_weights(model_1)
weights_2 = get_model_weights(model_2)

# %%
weight_differences = calculate_weight_differences(weights_1, weights_2)

# %%
analysis = analyze_weight_patterns(weight_differences)

# %%
print_weight_analysis_summary(analysis)

# %%
activations_model_1 = {}
activations_model_2 = {}
current_hooks = []

# %%
def clear_activations():
    global activations_model_1, activations_model_2
    activations_model_1.clear()
    activations_model_2.clear()

def remove_all_hooks():
    global current_hooks
    for hook in current_hooks:
        hook.remove()
    current_hooks.clear()

def get_activation_hook(name, model_name):
    def hook(module, input, output):
        try:
            # Handle output
            if isinstance(output, tuple):
                activation = output[0] if output[0] is not None else None
            else:
                activation = output
            
            # Handle input - check for None values
            input_tensor = None
            if input is not None:
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0] if input[0] is not None else None
                else:
                    input_tensor = input if input is not None else None
            
            # Create activation data with None checks
            activation_data = {
                'output': activation.detach().cpu() if activation is not None else None,
                'input': input_tensor.detach().cpu() if input_tensor is not None else None,
                'weight': module.weight.detach().cpu() if hasattr(module, 'weight') and module.weight is not None else None,
                'bias': module.bias.detach().cpu() if hasattr(module, 'bias') and module.bias is not None else None
            }
            
            if model_name == "Model_1":
                activations_model_1[name] = activation_data
            else:
                activations_model_2[name] = activation_data
                
        except Exception as e:
            print(f"Error in hook {name}: {e}")
            # Store None data to prevent missing keys
            activation_data = {
                'output': None,
                'input': None, 
                'weight': None,
                'bias': None
            }
            
            if model_name == "Model_1":
                activations_model_1[name] = activation_data
            else:
                activations_model_2[name] = activation_data
            
    return hook

def register_llama_hooks(model, model_name, layer_range=None):
    global current_hooks
    hooks = []
    layer_info = {}
    
    # Determine layer range
    if layer_range is None:
        layer_range = range(len(model.model.layers))
    
    for i in layer_range:
        if i >= len(model.model.layers):
            continue
            
        layer = model.model.layers[i]
        layer_prefix = f"layer_{i}"
        
        # 1. Self-Attention Components
        # Query, Key, Value projections
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_attention_q", model_name)
        ))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_attention_k", model_name)
        ))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_attention_v", model_name)
        ))
        
        # Output projection
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_attention_output", model_name)
        ))
        
        # 2. MLP Components  
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_mlp_gate", model_name)
        ))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_mlp_up", model_name)
        ))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_mlp_down", model_name)
        ))
        
        # 3. Layer Norms
        hooks.append(layer.input_layernorm.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_input_norm", model_name)
        ))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            get_activation_hook(f"{layer_prefix}_post_attn_norm", model_name)
        ))
        
        # Store layer info
        layer_info[layer_prefix] = {
            'layer_idx': i,
            'components': ['attention_q', 'attention_k', 'attention_v', 
                         'attention_output', 'mlp_gate', 'mlp_up', 'mlp_down',
                         'input_norm', 'post_attn_norm']
        }
    
    # Final layer norm and LM head (optional)
    hooks.append(model.model.norm.register_forward_hook(
        get_activation_hook("final_norm", model_name)
    ))
    hooks.append(model.lm_head.register_forward_hook(
        get_activation_hook("lm_head", model_name)
    ))
    
    current_hooks.extend(hooks)
    return hooks, layer_info

# %%
def select_random_neurons_per_layer(activations, seed=42):
    np.random.seed(seed)
    selected_neurons = {}
    
    for layer_name, layer_data in activations.items():
        if not isinstance(layer_data, dict):
            continue
            
        activation = layer_data.get('output')
        
        if activation is None:
            print(f"Skipping {layer_name}: No activation data")
            continue
        
        # Handle different activation shapes
        if len(activation.shape) == 3:  # [batch, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = activation.shape
            
            if hidden_size == 0:
                continue
            
            neuron_idx = np.random.randint(0, hidden_size)
            
            selected_neurons[layer_name] = {
                'neuron_index': neuron_idx,
                'sequence_length': seq_len,
                'hidden_size': hidden_size,
                'activation_shape': activation.shape
            }
            
    return selected_neurons
    
def get_component_type(layer_name):
    if 'attention' in layer_name:
        return 'attention'
    elif 'mlp' in layer_name:
        return 'mlp'
    elif 'norm' in layer_name:
        return 'normalization'
    elif 'lm_head' in layer_name:
        return 'output'
    else:
        return 'other'

# %%
def calculate_neuron_outputs(layer_name, neuron_idx, input_tensor,
                            weights_1, weights_2, 
                            actual_output_1, actual_output_2):
    results = {
        'neuron_index': neuron_idx,
        'calculations': [],
        'layer_type': get_component_type(layer_name)
    }
    
    # Skip if essential data is missing
    if input_tensor is None or weights_1 is None or weights_2 is None:
        return results
    
    # Get weights and biases
    w1 = weights_1.get('weight')
    w2 = weights_2.get('weight')
    b1 = weights_1.get('bias')
    b2 = weights_2.get('bias')
    
    if w1 is None or w2 is None:
        return results
    
    # Handle layer normalization differently (1D weights)
    if 'norm' in layer_name:
        # Layer norm: output = weight * normalized_input + bias
        # For layer norm, we can't select individual neurons the same way
        # Instead, we'll look at the scaling factor for the selected dimension
        for token_idx in range(input_tensor.shape[1]):
            try:
                token_input = input_tensor[0, token_idx, :]  # [hidden_size]
                
                # For layer norm, weight is 1D, so we use it as element-wise multiplication
                if neuron_idx < w1.shape[0] and neuron_idx < w2.shape[0]:
                    # Get the scaling factor for this dimension
                    scale_1 = w1[neuron_idx].item()
                    scale_2 = w2[neuron_idx].item()
                    
                    # Get the normalized input value for this dimension
                    input_val = token_input[neuron_idx].item()
                    
                    # Calculate scaled outputs
                    calc_1 = scale_1 * input_val
                    calc_2 = scale_2 * input_val
                    
                    if b1 is not None and neuron_idx < b1.shape[0]:
                        calc_1 += b1[neuron_idx].item()
                    if b2 is not None and neuron_idx < b2.shape[0]:
                        calc_2 += b2[neuron_idx].item()
                    
                    # Get actual outputs
                    actual_1 = actual_output_1[0, token_idx, neuron_idx] if actual_output_1 is not None else None
                    actual_2 = actual_output_2[0, token_idx, neuron_idx] if actual_output_2 is not None else None
                    
                    results['calculations'].append({
                        'token_position': token_idx,
                        'model_1_calculated': calc_1,
                        'model_2_calculated': calc_2,
                        'difference': calc_1 - calc_2,
                        'model_1_actual': actual_1.item() if actual_1 is not None else None,
                        'model_2_actual': actual_2.item() if actual_2 is not None else None,
                        'weight_diff': scale_1 - scale_2
                    })
                    
            except Exception as e:
                continue
    
    else:
        # Handle regular linear layers (2D weights)
        for token_idx in range(input_tensor.shape[1]):
            try:
                token_input = input_tensor[0, token_idx, :]  # [hidden_size]
                
                # Check bounds
                if neuron_idx >= w1.shape[0] or neuron_idx >= w2.shape[0]:
                    continue
                
                # Model 1 calculation: input @ w1.T + b1
                calc_1 = torch.matmul(token_input, w1[neuron_idx, :])
                if b1 is not None and neuron_idx < b1.shape[0]:
                    calc_1 += b1[neuron_idx]
                
                # Model 2 calculation: input @ w2.T + b2  
                calc_2 = torch.matmul(token_input, w2[neuron_idx, :])
                if b2 is not None and neuron_idx < b2.shape[0]:
                    calc_2 += b2[neuron_idx]
                
                # Apply activation function for MLP gate/up projections
                if 'mlp_gate' in layer_name or 'mlp_up' in layer_name:
                    calc_1 = F.silu(calc_1)
                    calc_2 = F.silu(calc_2)
                
                # Get actual outputs
                actual_1 = actual_output_1[0, token_idx, neuron_idx] if actual_output_1 is not None else None
                actual_2 = actual_output_2[0, token_idx, neuron_idx] if actual_output_2 is not None else None
                
                results['calculations'].append({
                    'token_position': token_idx,
                    'model_1_calculated': calc_1.item(),
                    'model_2_calculated': calc_2.item(),
                    'difference': (calc_1 - calc_2).item(),
                    'model_1_actual': actual_1.item() if actual_1 is not None else None,
                    'model_2_actual': actual_2.item() if actual_2 is not None else None
                })
                
            except Exception as e:
                continue
    
    return results

# %%
def compare_neuron_calculations(model_1_activations, model_2_activations, 
                                selected_neurons):
    comparison_results = {}
    
    for layer_name, neuron_info in selected_neurons.items():
        neuron_idx = neuron_info['neuron_index']
        
        # Get current layer data
        layer_1_data = model_1_activations.get(layer_name, {})
        layer_2_data = model_2_activations.get(layer_name, {})
        
        # Skip if missing data
        if not isinstance(layer_1_data, dict) or not isinstance(layer_2_data, dict):
            continue
            
        # Get the input to this layer (from Model 1)
        model_1_input = layer_1_data.get('input')
        
        if model_1_input is None:
            continue
        
        # Get weights from both models
        weights_1 = layer_1_data  # Contains weight and bias
        weights_2 = layer_2_data  # Contains weight and bias
        
        # Calculate outputs for the selected neuron
        results = calculate_neuron_outputs(
            layer_name, neuron_idx, model_1_input,
            weights_1, weights_2,
            layer_1_data.get('output'), layer_2_data.get('output')
        )
        
        if results and results['calculations']:
            comparison_results[layer_name] = results
            
    return comparison_results

def run_comparison(text_input, seed=42):
    print(f"Processing: {text_input[:50]}...")
    
    # Clear previous data
    clear_activations()
    remove_all_hooks()
    
    # Tokenize input
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    
    try:
        # Register hooks for all layers
        print("Registering hooks...")
        hooks_1, layers_1 = register_llama_hooks(model_1, "Model_1")
        hooks_2, layers_2 = register_llama_hooks(model_2, "Model_2")
        
        # Run both models
        print("Running models...")
        with torch.no_grad():
            outputs_1 = model_1(**inputs)
            outputs_2 = model_2(**inputs)
        
        print(f"Captured {len(activations_model_1)} activations from Model 1")
        print(f"Captured {len(activations_model_2)} activations from Model 2")
        
        # Select random neurons (one per layer)
        print("Selecting random neurons...")
        selected_neurons = select_random_neurons_per_layer(activations_model_1, seed=seed)
        
        print(f"Selected neurons from {len(selected_neurons)} layers")
        
        # Compare activations
        print("Comparing activations...")
        comparison_results = compare_neuron_calculations(
            activations_model_1,
            activations_model_2,
            selected_neurons
        )
        
        return {
            'input_text': text_input,
            'tokenized_input': inputs,
            'model_1_output': outputs_1.logits,
            'model_2_output': outputs_2.logits,
            'layer_comparisons': comparison_results,
            'selected_neurons': selected_neurons
        }
    
    except Exception as e:
        print(f"Error in run_comparison: {e}")
        import traceback
        traceback.print_exc()
        return {
            'input_text': text_input,
            'error': str(e),
            'layer_comparisons': {},
            'selected_neurons': {}
        }
    
    finally:
        # Always cleanup hooks
        remove_all_hooks()

def save_detailed_results(comparison_results, filename="detailed_activation_comparison.csv"):
    rows = []
    
    for layer_name, layer_data in comparison_results['layer_comparisons'].items():
        if 'calculations' in layer_data:
            for calc in layer_data['calculations']:
                row = {
                    'input_text': comparison_results['input_text'][:100],
                    'layer_name': layer_name,
                    'layer_type': layer_data['layer_type'],
                    'neuron_index': layer_data['neuron_index'],
                    'token_position': calc['token_position'],
                    'model_1_calculated': calc['model_1_calculated'],
                    'model_2_calculated': calc['model_2_calculated'],
                    'difference': calc['difference'],
                    'abs_difference': abs(calc['difference']),
                    'model_1_actual': calc.get('model_1_actual'),
                    'model_2_actual': calc.get('model_2_actual')
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    
    return df

# %%
TEST_TEXTS = [
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


# %%
all_results = []

for i, text in enumerate(TEST_TEXTS):
    print(f"\n=== Processing text {i+1}/{len(TEST_TEXTS)} ===")
    
    try:
        # Use different seed for each text to get variety
        result = run_comparison(text, seed=42+i)
        
        all_results.append(result)
        
        # Save detailed results
        save_detailed_results(result, filename="all_layers_activation_comparison.csv")
        
        print(f"Completed text {i+1}")
        
    except Exception as e:
        print(f"Error processing text {i+1}: {e}")
        continue

# %%
# %%
# Print summary statistics
if all_results:
    print("\n=== SUMMARY STATISTICS (ALL LAYERS) ===")
    
    all_layer_stats = defaultdict(list)
    all_component_stats = defaultdict(list)
    
    for result in all_results:
        for layer_name, layer_data in result['layer_comparisons'].items():
            if 'calculations' in layer_data:
                diffs = [abs(calc['difference']) for calc in layer_data['calculations']]
                if diffs:
                    mean_diff = np.mean(diffs)
                    all_layer_stats[layer_name].append(mean_diff)
                    all_component_stats[layer_data['layer_type']].append(mean_diff)
    
    print("\nTop 10 layers by average difference:")
    layer_avg_diffs = [(layer, np.mean(diffs)) for layer, diffs in all_layer_stats.items()]
    layer_avg_diffs.sort(key=lambda x: x[1], reverse=True)
    
    for layer, avg_diff in layer_avg_diffs[:10]:
        std_diff = np.std(all_layer_stats[layer])
        print(f"  {layer}: {avg_diff:.6f} ± {std_diff:.6f}")
    
    print("\nAverage differences by component type:")
    for component, diffs in all_component_stats.items():
        print(f"  {component}: {np.mean(diffs):.6f} ± {np.std(diffs):.6f}")
    
    # Calculate overall statistics
    all_differences = []
    for result in all_results:
        for layer_name, layer_data in result['layer_comparisons'].items():
            if 'calculations' in layer_data:
                all_differences.extend([abs(calc['difference']) 
                                      for calc in layer_data['calculations']])
    
    if all_differences:
        print(f"\nOverall statistics:")
        print(f"  Total comparisons: {len(all_differences)}")
        print(f"  Mean absolute difference: {np.mean(all_differences):.6f}")
        print(f"  Std deviation: {np.std(all_differences):.6f}")
        print(f"  Max difference: {np.max(all_differences):.6f}")
        print(f"  Min difference: {np.min(all_differences):.6f}")
        print(f"  Median difference: {np.median(all_differences):.6f}")


# %%
def calculate_total_differences(result):
    total_diff = 0
    layer_diffs = {}
    token_diffs = {}
    
    for layer_name, layer_data in result['layer_comparisons'].items():
        if 'calculations' not in layer_data or not layer_data['calculations']:
            continue
        
        layer_sum = 0
        for calc in layer_data['calculations']:
            diff = abs(calc['difference'])
            layer_sum += diff
            
            # Track per-token differences
            token_pos = calc['token_position']
            if token_pos not in token_diffs:
                token_diffs[token_pos] = 0
            token_diffs[token_pos] += diff
        
        layer_diffs[layer_name] = layer_sum
        total_diff += layer_sum
    
    return {
        'total_difference': total_diff,
        'layer_differences': layer_diffs,
        'token_differences': token_diffs,
        'num_layers': len(layer_diffs),
        'num_tokens': len(token_diffs)
    }

def decode_and_compare_outputs(result, tokenizer, top_k=5):
    input_ids = result['tokenized_input']['input_ids']
    logits_1 = result['model_1_output']
    logits_2 = result['model_2_output']
    
    # Get predictions for the last token (next token prediction)
    last_token_logits_1 = logits_1[0, -1, :]  # [vocab_size]
    last_token_logits_2 = logits_2[0, -1, :]  # [vocab_size]
    
    # Get top-k predictions
    probs_1 = torch.softmax(last_token_logits_1, dim=-1)
    probs_2 = torch.softmax(last_token_logits_2, dim=-1)
    
    top_probs_1, top_indices_1 = torch.topk(probs_1, top_k)
    top_probs_2, top_indices_2 = torch.topk(probs_2, top_k)
    
    # Decode tokens
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    print(f"\nInput text: {input_text}")
    print(f"\n{'='*60}")
    print(f"{'Model 1 Predictions':<30} | {'Model 2 Predictions':<30}")
    print(f"{'='*60}")
    
    for i in range(top_k):
        token_1 = tokenizer.decode(top_indices_1[i])
        prob_1 = top_probs_1[i].item()
        
        token_2 = tokenizer.decode(top_indices_2[i])
        prob_2 = top_probs_2[i].item()
        
        print(f"{i+1}. '{token_1}' ({prob_1:.3f}){' '*(20-len(token_1))} | "
              f"{i+1}. '{token_2}' ({prob_2:.3f})")
    
    # Calculate Jensen-Shannon divergence between distributions
    def jensen_shannon_divergence(p, q):
        """Calculate Jensen-Shannon divergence between two probability distributions."""
        # Add small epsilon for numerical stability
        p = p + 1e-10
        q = q + 1e-10
        
        # Calculate the average distribution M = (P + Q) / 2
        m = (p + q) / 2
        
        # Calculate KL divergences: KL(P||M) and KL(Q||M)
        kl_pm = torch.nn.functional.kl_div(torch.log(m), p, reduction='sum')
        kl_qm = torch.nn.functional.kl_div(torch.log(m), q, reduction='sum')
        
        # Jensen-Shannon divergence = (KL(P||M) + KL(Q||M)) / 2
        js_div = (kl_pm + kl_qm) / 2
        
        return js_div.item()
    
    js_div = jensen_shannon_divergence(probs_1, probs_2)
    
    print(f"\nJensen-Shannon Divergence: {js_div:.4f}")
    
    return {
        'top_tokens_model_1': [tokenizer.decode(idx) for idx in top_indices_1],
        'top_probs_model_1': top_probs_1.tolist(),
        'top_tokens_model_2': [tokenizer.decode(idx) for idx in top_indices_2],
        'top_probs_model_2': top_probs_2.tolist(),
        'jensen_shannon_divergence': js_div
    }

def visualize_difference_summary(all_results, save_path='difference_summary.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sum of Activation Differences Analysis', fontsize=16)
    
    # 1. Total differences per query
    ax = axes[0, 0]
    total_diffs = []
    query_labels = []
    
    for i, result in enumerate(all_results):
        diff_analysis = calculate_total_differences(result)
        total_diffs.append(diff_analysis['total_difference'])
        query_labels.append(f"Query {i+1}")
    
    bars = ax.bar(query_labels, total_diffs, color='darkblue', alpha=0.7)
    ax.set_ylabel('Total Absolute Difference')
    ax.set_title('Total Activation Differences per Query')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, total_diffs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 2. Average difference per layer
    ax = axes[0, 1]
    avg_diffs_per_layer = []
    
    for result in all_results:
        diff_analysis = calculate_total_differences(result)
        if diff_analysis['num_layers'] > 0:
            avg_diff = diff_analysis['total_difference'] / diff_analysis['num_layers']
            avg_diffs_per_layer.append(avg_diff)
    
    ax.plot(range(len(avg_diffs_per_layer)), avg_diffs_per_layer, 'ro-', markersize=8)
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Average Difference per Layer')
    ax.set_title('Average Layer Difference by Query')
    ax.grid(True, alpha=0.3)
    
    # 3. Difference distribution across layers (for first query)
    ax = axes[0, 2]
    if all_results:
        first_result = all_results[0]
        diff_analysis = calculate_total_differences(first_result)
        
        # Get layer types and their differences
        layer_types = {'norm': 0, 'mlp': 0, 'attn': 0, 'other': 0}
        for layer_name, diff in diff_analysis['layer_differences'].items():
            if 'norm' in layer_name:
                layer_types['norm'] += diff
            elif 'mlp' in layer_name:
                layer_types['mlp'] += diff
            elif 'attn' in layer_name:
                layer_types['attn'] += diff
            else:
                layer_types['other'] += diff
        
        # Create pie chart
        non_zero_types = {k: v for k, v in layer_types.items() if v > 0}
        if non_zero_types:
            ax.pie(non_zero_types.values(), labels=non_zero_types.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Difference Distribution by Layer Type\n(Query 1)')
    
    # 4. Token-wise differences (averaged across queries)
    ax = axes[1, 0]
    max_tokens = max(len(calculate_total_differences(r)['token_differences']) 
                     for r in all_results)
    
    avg_token_diffs = []
    for token_pos in range(max_tokens):
        token_sum = 0
        count = 0
        for result in all_results:
            diff_analysis = calculate_total_differences(result)
            if token_pos in diff_analysis['token_differences']:
                token_sum += diff_analysis['token_differences'][token_pos]
                count += 1
        if count > 0:
            avg_token_diffs.append(token_sum / count)
        else:
            avg_token_diffs.append(0)
    
    ax.bar(range(len(avg_token_diffs)), avg_token_diffs, color='green', alpha=0.7)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Average Absolute Difference')
    ax.set_title('Average Difference by Token Position')
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative differences
    ax = axes[1, 1]
    for i, result in enumerate(all_results):
        diff_analysis = calculate_total_differences(result)
        
        # Sort layers by name for consistent ordering
        sorted_layers = sorted(diff_analysis['layer_differences'].items())
        layer_names = [l[0] for l in sorted_layers]
        layer_diffs = [l[1] for l in sorted_layers]
        
        # Calculate cumulative sum
        cumulative = np.cumsum(layer_diffs)
        
        # Plot every 10th layer to avoid overcrowding
        x_points = list(range(0, len(cumulative), 10))
        y_points = [cumulative[i] for i in x_points]
        
        ax.plot(x_points, y_points, '-o', label=f'Query {i+1}', 
                markersize=4, alpha=0.7)
    
    ax.set_xlabel('Layer Index (every 10th)')
    ax.set_ylabel('Cumulative Difference')
    ax.set_title('Cumulative Differences Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    
    for i, result in enumerate(all_results[:5]):  # Show first 5
        diff_analysis = calculate_total_differences(result)
        text_preview = result['input_text'][:30] + "..."
        
        summary_text += f"Query {i+1}: {text_preview}\n"
        summary_text += f"  Total Diff: {diff_analysis['total_difference']:.2f}\n"
        summary_text += f"  Layers: {diff_analysis['num_layers']}\n"
        summary_text += f"  Avg/Layer: {diff_analysis['total_difference']/diff_analysis['num_layers']:.2f}\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Difference summary visualization saved to {save_path}")


# %%
# Run the analysis with difference tracking
all_results = []
difference_summaries = []

print("\n" + "="*70)
print("RUNNING ANALYSIS WITH DIFFERENCE TRACKING")
print("="*70)

for i, text in enumerate(TEST_TEXTS):
    print(f"\n{'='*70}")
    print(f"Processing Query {i+1}/{len(TEST_TEXTS)}")
    print(f"Text: {text[:60]}...")
    print(f"{'='*70}")
    
    try:
        # Run comparison
        result = run_comparison(text, seed=42+i)
        
        # Calculate differences
        diff_analysis = calculate_total_differences(result)
        
        # Add output comparison
        output_comp = decode_and_compare_outputs(result, tokenizer, top_k=5)
        result['output_comparison'] = output_comp
        result['difference_analysis'] = diff_analysis
        
        all_results.append(result)
        difference_summaries.append(diff_analysis)
        
        # Print summary for this query
        print(f"\n{'='*50}")
        print(f"QUERY {i+1} DIFFERENCE SUMMARY")
        print(f"{'='*50}")
        print(f"Total Absolute Difference: {diff_analysis['total_difference']:.2f}")
        print(f"Number of Layers Analyzed: {diff_analysis['num_layers']}")
        print(f"Number of Tokens: {diff_analysis['num_tokens']}")
        print(f"Average Difference per Layer: {diff_analysis['total_difference']/diff_analysis['num_layers']:.2f}")
        print(f"Average Difference per Token: {diff_analysis['total_difference']/diff_analysis['num_tokens']:.2f}")
        
        # Show top 5 layers with highest differences
        sorted_layers = sorted(diff_analysis['layer_differences'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 Layers with Highest Differences:")
        for layer_name, diff in sorted_layers:
            print(f"  {layer_name}: {diff:.2f}")
        
        print(f"\nCompleted Query {i+1}")
        
    except Exception as e:
        print(f"Error processing query {i+1}: {e}")
        import traceback
        traceback.print_exc()
        continue


# Create visualizations
if all_results:
    visualize_difference_summary(all_results, save_path='difference_summary.png')


# Create a detailed comparison table with differences
print("\n" + "="*70)
print("FINAL SUMMARY TABLE WITH DIFFERENCES")
print("="*70)

summary_data = []

for i, result in enumerate(all_results):
    if 'difference_analysis' in result and 'output_comparison' in result:
        diff = result['difference_analysis']
        out = result['output_comparison']
        
        summary_data.append({
            'Query': i+1,
            'Text': result['input_text'][:40] + '...',
            'Total_Diff': f"{diff['total_difference']:.2f}",
            'Avg_Layer_Diff': f"{diff['total_difference']/diff['num_layers']:.2f}",
            'KL_Div': f"{out['kl_divergence']:.4f}",
            'Top1_Match': '✓' if out['top_tokens_model_1'][0] == out['top_tokens_model_2'][0] else '✗'
        })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Plot correlation between activation differences and output differences
if len(all_results) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    activation_diffs = [r['difference_analysis']['total_difference'] for r in all_results 
                       if 'difference_analysis' in r]
    kl_divs = [r['output_comparison']['kl_divergence'] for r in all_results 
               if 'output_comparison' in r]
    
    if len(activation_diffs) == len(kl_divs):
        ax.scatter(activation_diffs, kl_divs, s=100, alpha=0.7, c='purple')
        
        # Add labels for each point
        for i, (x, y) in enumerate(zip(activation_diffs, kl_divs)):
            ax.annotate(f'Q{i+1}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        
        # Add trend line
        z = np.polyfit(activation_diffs, kl_divs, 1)
        p = np.poly1d(z)
        ax.plot(activation_diffs, p(activation_diffs), "r--", alpha=0.8, 
                label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        
        ax.set_xlabel('Total Activation Difference')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Correlation: Activation Differences vs Output Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(activation_diffs, kl_divs)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('activation_output_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nCorrelation plot saved to activation_output_correlation.png")
        print(f"Correlation coefficient: {correlation:.3f}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)



