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
import gc

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_1_PATH = "meta-llama/Llama-2-7b-chat-hf" 
MODEL_2_PATH = "meta-llama/Llama-2-7b-hf"       

print(DEVICE)

# %%
print(f"Using device: {DEVICE}")

# %%
tokenizer = LlamaTokenizer.from_pretrained(MODEL_1_PATH)
tokenizer.pad_token = tokenizer.eos_token

model_1 = LlamaForCausalLM.from_pretrained(
    MODEL_1_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

model_2 = LlamaForCausalLM.from_pretrained(
    MODEL_2_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Models loaded successfully")

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
# Global variables for activation capture
activations_model_1 = {}
activations_model_2 = {}
current_hooks = []
hook_errors = []


# %%
def clear_activations():
    global activations_model_1, activations_model_2
    activations_model_1.clear()
    activations_model_2.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def remove_all_hooks():
    global current_hooks
    for hook in current_hooks:
        try:
            hook.remove()
        except:
            pass
    current_hooks.clear()

def get_activation_hook(name, model_name):
    def hook(module, input, output):
        global hook_errors
        try:
            # Handle output
            if isinstance(output, tuple):
                activation = output[0] if len(output) > 0 and output[0] is not None else None
            else:
                activation = output
            
            # Handle input
            input_tensor = None
            if input is not None and isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0] if input[0] is not None else None
            
            # Store activation data
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
            error_msg = f"Hook error in {name} ({model_name}): {str(e)}"
            hook_errors.append(error_msg)
            print(f"WARNING: {error_msg}")
            
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

# %%
def register_llama_hooks(model, model_name, layer_range=None, max_layers=None):
    global current_hooks, hook_errors
    hooks = []
    successful_hooks = 0
    failed_hooks = 0
    
    hook_errors.clear()
    
    total_layers = len(model.model.layers)
    if max_layers is not None:
        total_layers = min(total_layers, max_layers)
    
    if layer_range is None:
        layer_range = range(total_layers)
    
    print(f"Registering hooks for {model_name}: {len(layer_range)} layers")
    
    for i in layer_range:
        if i >= len(model.model.layers):
            continue
            
        layer = model.model.layers[i]
        layer_prefix = f"layer_{i}"
        
        components_to_hook = [
            (layer.self_attn.q_proj, f"{layer_prefix}_attention_q"),
            (layer.self_attn.k_proj, f"{layer_prefix}_attention_k"),
            (layer.self_attn.v_proj, f"{layer_prefix}_attention_v"),
            (layer.self_attn.o_proj, f"{layer_prefix}_attention_output"),
            (layer.mlp.gate_proj, f"{layer_prefix}_mlp_gate"),
            (layer.mlp.up_proj, f"{layer_prefix}_mlp_up"),
            (layer.mlp.down_proj, f"{layer_prefix}_mlp_down"),
            (layer.input_layernorm, f"{layer_prefix}_input_norm"),
            (layer.post_attention_layernorm, f"{layer_prefix}_post_attn_norm"),
        ]
        
        for module, hook_name in components_to_hook:
            try:
                hook = module.register_forward_hook(
                    get_activation_hook(hook_name, model_name)
                )
                hooks.append(hook)
                successful_hooks += 1
            except Exception as e:
                error_msg = f"Failed to register {hook_name}: {str(e)}"
                hook_errors.append(error_msg)
                failed_hooks += 1
    
    # Register final components
    try:
        hooks.append(model.model.norm.register_forward_hook(
            get_activation_hook("final_norm", model_name)
        ))
        hooks.append(model.lm_head.register_forward_hook(
            get_activation_hook("lm_head", model_name)
        ))
        successful_hooks += 2
    except Exception as e:
        error_msg = f"Failed to register final components: {str(e)}"
        hook_errors.append(error_msg)
        failed_hooks += 2
    
    current_hooks.extend(hooks)
    
    print(f"Hook registration complete for {model_name}:")
    print(f"  ✓ Successful: {successful_hooks}")
    print(f"  ✗ Failed: {failed_hooks}")
    
    return hooks

def select_neurons_per_token_position(activations1, activations2, selection_method='min_diff', seed=42):
    np.random.seed(seed)
    selected_neurons = {}
    
    print(f"Selecting neurons per token position from {len(activations1)} layers...")
    print(f"Selection method: {selection_method}")
    
    for layer_name, layer_data in activations1.items():
        if not isinstance(layer_data, dict):
            continue
            
        activation1 = layer_data.get('output')
        activation2 = activations2.get(layer_name, {}).get('output')

        if activation1 is None or activation2 is None:
            print(f"Skipping {layer_name}: Missing activation data")
            continue
        
        try:
            if len(activation1.shape) == 3:  # [batch, seq_len, hidden_size]
                batch_size, seq_len, hidden_size = activation1.shape
                
                if hidden_size == 0:
                    continue
                
                # Select neurons for EACH token position separately
                token_selections = {}
                
                for token_pos in range(seq_len):
                    # Get activations for this specific token position
                    token_act1 = activation1[0, token_pos, :]  # [hidden_size]
                    token_act2 = activation2[0, token_pos, :]  # [hidden_size]
                    
                    # Calculate differences for this token
                    diff = torch.abs(token_act1 - token_act2)
                    
                    # Select neuron based on method
                    if selection_method == 'min_diff':
                        neuron_idx = torch.argmin(diff).item()
                    elif selection_method == 'max_diff':
                        neuron_idx = torch.argmax(diff).item()
                    elif selection_method == 'random':
                        neuron_idx = np.random.randint(0, hidden_size)
                    elif selection_method == 'high_activation':
                        # Select neuron with highest activation in model 1
                        neuron_idx = torch.argmax(torch.abs(token_act1)).item()
                    else:
                        neuron_idx = torch.argmin(diff).item()  # Default to min_diff
                    
                    token_selections[token_pos] = {
                        'neuron_index': neuron_idx,
                        'difference': diff[neuron_idx].item(),
                        'activation1_value': token_act1[neuron_idx].item(),
                        'activation2_value': token_act2[neuron_idx].item(),
                        'abs_activation1': abs(token_act1[neuron_idx].item()),
                        'abs_activation2': abs(token_act2[neuron_idx].item()),
                        'selection_method': selection_method
                    }
                
                selected_neurons[layer_name] = {
                    'per_token_selections': token_selections,
                    'sequence_length': seq_len,
                    'hidden_size': hidden_size,
                    'activation_shape': list(activation1.shape),
                    'layer_type': get_component_type(layer_name)
                }
                
        except Exception as e:
            print(f"Error selecting neurons for {layer_name}: {e}")
            continue
            
    print(f"Successfully selected neurons from {len(selected_neurons)} layers")
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
    elif 'embed' in layer_name:
        return 'embedding'
    else:
        return 'other'

# %%
def calculate_single_token_neuron(layer_name, neuron_idx, token_pos, 
                                 layer_1_data, layer_2_data):

    input_tensor = layer_1_data.get('input')
    if input_tensor is None or token_pos >= input_tensor.shape[1]:
        return {'error': 'Missing or invalid input data'}
    
    # Get input for this specific token
    token_input = input_tensor[0, token_pos, :]  # [hidden_size]
    
    # Get weights
    w1 = layer_1_data.get('weight')
    w2 = layer_2_data.get('weight')
    b1 = layer_1_data.get('bias')
    b2 = layer_2_data.get('bias')
    
    if w1 is None or w2 is None:
        return {'error': 'Missing weight data'}
    
    try:
        # Calculate for this specific token and neuron
        if 'norm' in layer_name:
            # Layer norm calculation: weight * normalized_input + bias
            if neuron_idx >= w1.shape[0] or neuron_idx >= token_input.shape[0]:
                return {'error': 'Index out of bounds for layer norm'}
                
            calc_1 = w1[neuron_idx].item() * token_input[neuron_idx].item()
            calc_2 = w2[neuron_idx].item() * token_input[neuron_idx].item()
            
            if b1 is not None and neuron_idx < b1.shape[0]:
                calc_1 += b1[neuron_idx].item()
            if b2 is not None and neuron_idx < b2.shape[0]:
                calc_2 += b2[neuron_idx].item()
                
        else:
            # Linear layer calculation: input @ weight.T + bias
            if neuron_idx >= w1.shape[0]:
                return {'error': 'Neuron index out of bounds'}
                
            calc_1 = torch.matmul(token_input, w1[neuron_idx, :]).item()
            calc_2 = torch.matmul(token_input, w2[neuron_idx, :]).item()
            
            if b1 is not None and neuron_idx < b1.shape[0]:
                calc_1 += b1[neuron_idx].item()
            if b2 is not None and neuron_idx < b2.shape[0]:
                calc_2 += b2[neuron_idx].item()
            
            # Apply activation function for MLP components
            if 'mlp_gate' in layer_name or 'mlp_up' in layer_name:
                calc_1 = F.silu(torch.tensor(calc_1)).item()
                calc_2 = F.silu(torch.tensor(calc_2)).item()
        
        # Get actual outputs from the models
        actual_1 = layer_1_data.get('output')
        actual_2 = layer_2_data.get('output')
        
        actual_1_val = None
        actual_2_val = None
        
        if actual_1 is not None and token_pos < actual_1.shape[1] and neuron_idx < actual_1.shape[2]:
            actual_1_val = actual_1[0, token_pos, neuron_idx].item()
        if actual_2 is not None and token_pos < actual_2.shape[1] and neuron_idx < actual_2.shape[2]:
            actual_2_val = actual_2[0, token_pos, neuron_idx].item()
        
        # Calculate errors between our calculations and actual outputs
        calc_error_1 = abs(calc_1 - actual_1_val) if actual_1_val is not None else None
        calc_error_2 = abs(calc_2 - actual_2_val) if actual_2_val is not None else None
        
        return {
            'token_position': token_pos,
            'neuron_index': neuron_idx,
            'model_1_calculated': calc_1,
            'model_2_calculated': calc_2,
            'calculation_difference': calc_1 - calc_2,
            'model_1_actual': actual_1_val,
            'model_2_actual': actual_2_val,
            'actual_difference': (actual_1_val - actual_2_val) if (actual_1_val is not None and actual_2_val is not None) else None,
            'calculation_error_1': calc_error_1,
            'calculation_error_2': calc_error_2,
            'layer_type': get_component_type(layer_name)
        }
        
    except Exception as e:
        return {'error': f'Calculation failed: {str(e)}'}

def compare_neuron_calculations_per_token(model_1_activations, model_2_activations, 
                                        selected_neurons):
    comparison_results = {}
    
    print(f"Comparing calculations for {len(selected_neurons)} layers...")
    
    for layer_name, neuron_info in selected_neurons.items():
        if 'per_token_selections' not in neuron_info:
            continue
            
        results = {
            'layer_type': neuron_info.get('layer_type', get_component_type(layer_name)),
            'sequence_length': neuron_info['sequence_length'],
            'hidden_size': neuron_info['hidden_size'],
            'token_analyses': {},
            'summary_stats': {}
        }
        
        # Get layer data
        layer_1_data = model_1_activations.get(layer_name, {})
        layer_2_data = model_2_activations.get(layer_name, {})
        
        if not isinstance(layer_1_data, dict) or not isinstance(layer_2_data, dict):
            print(f"Skipping {layer_name}: Invalid layer data")
            continue
        
        # Analyze each token position with its selected neuron
        valid_analyses = 0
        calc_diffs = []
        actual_diffs = []
        calc_errors_1 = []
        calc_errors_2 = []
        
        for token_pos, token_data in neuron_info['per_token_selections'].items():
            neuron_idx = token_data['neuron_index']
            
            # Calculate for this specific token and neuron
            token_analysis = calculate_single_token_neuron(
                layer_name, neuron_idx, token_pos,
                layer_1_data, layer_2_data
            )
            
            # Add selection info to analysis
            if 'error' not in token_analysis:
                token_analysis.update({
                    'selected_activation1': token_data['activation1_value'],
                    'selected_activation2': token_data['activation2_value'],
                    'selection_difference': token_data['difference'],
                    'selection_method': token_data.get('selection_method', 'unknown')
                })
                
                valid_analyses += 1
                calc_diffs.append(token_analysis['calculation_difference'])
                
                if token_analysis['actual_difference'] is not None:
                    actual_diffs.append(token_analysis['actual_difference'])
                if token_analysis['calculation_error_1'] is not None:
                    calc_errors_1.append(token_analysis['calculation_error_1'])
                if token_analysis['calculation_error_2'] is not None:
                    calc_errors_2.append(token_analysis['calculation_error_2'])
            
            results['token_analyses'][token_pos] = token_analysis
        
        # Calculate summary statistics
        if valid_analyses > 0:
            results['summary_stats'] = {
                'valid_analyses': valid_analyses,
                'total_tokens': len(neuron_info['per_token_selections']),
                'mean_calc_difference': np.mean(calc_diffs) if calc_diffs else None,
                'std_calc_difference': np.std(calc_diffs) if calc_diffs else None,
                'max_abs_calc_difference': max([abs(d) for d in calc_diffs]) if calc_diffs else None,
                'mean_actual_difference': np.mean(actual_diffs) if actual_diffs else None,
                'mean_calc_error_1': np.mean(calc_errors_1) if calc_errors_1 else None,
                'mean_calc_error_2': np.mean(calc_errors_2) if calc_errors_2 else None,
                'unique_neurons_selected': len(set(td['neuron_index'] for td in neuron_info['per_token_selections'].values()))
            }
        
        comparison_results[layer_name] = results
    
    print(f"Completed comparisons for {len(comparison_results)} layers")
    return comparison_results

def save_detailed_results_per_token(comparison_results, filename="per_token_neuron_analysis.csv"):
    rows = []
    
    input_text = comparison_results.get('input_text', 'Unknown')
    
    for layer_name, layer_data in comparison_results.get('layer_comparisons', {}).items():
        if 'token_analyses' not in layer_data:
            continue
            
        for token_pos, token_analysis in layer_data['token_analyses'].items():
            if 'error' in token_analysis:
                # Save error rows too
                row = {
                    'input_text': input_text[:100],
                    'layer_name': layer_name,
                    'layer_type': layer_data.get('layer_type', 'unknown'),
                    'token_position': token_pos,
                    'neuron_index': None,
                    'error': token_analysis['error'],
                    'model_1_calculated': None,
                    'model_2_calculated': None,
                    'calculation_difference': None,
                    'model_1_actual': None,
                    'model_2_actual': None,
                    'actual_difference': None,
                    'calculation_error_1': None,
                    'calculation_error_2': None,
                    'selected_activation1': None,
                    'selected_activation2': None,
                    'selection_difference': None,
                    'selection_method': None
                }
            else:
                row = {
                    'input_text': input_text[:100],
                    'layer_name': layer_name,
                    'layer_type': token_analysis.get('layer_type', 'unknown'),
                    'token_position': token_analysis['token_position'],
                    'neuron_index': token_analysis['neuron_index'],
                    'error': None,
                    'model_1_calculated': token_analysis['model_1_calculated'],
                    'model_2_calculated': token_analysis['model_2_calculated'],
                    'calculation_difference': token_analysis['calculation_difference'],
                    'abs_calculation_difference': abs(token_analysis['calculation_difference']),
                    'model_1_actual': token_analysis['model_1_actual'],
                    'model_2_actual': token_analysis['model_2_actual'],
                    'actual_difference': token_analysis['actual_difference'],
                    'abs_actual_difference': abs(token_analysis['actual_difference']) if token_analysis['actual_difference'] is not None else None,
                    'calculation_error_1': token_analysis['calculation_error_1'],
                    'calculation_error_2': token_analysis['calculation_error_2'],
                    'selected_activation1': token_analysis.get('selected_activation1'),
                    'selected_activation2': token_analysis.get('selected_activation2'),
                    'selection_difference': token_analysis.get('selection_difference'),
                    'selection_method': token_analysis.get('selection_method')
                }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    
    print(f"Saved {len(rows)} rows to {filename}")
    return df

def run_comparison_per_token(text_input, selection_method='min_diff', seed=42, max_layers=None):
    print(f"\n{'='*60}")
    print(f"Processing: {text_input[:50]}...")
    print(f"Selection method: {selection_method}")
    print(f"{'='*60}")
    
    # Clear previous data and free memory
    clear_activations()
    remove_all_hooks()
    
    # Tokenize input
    inputs = tokenizer(
        text_input, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    
    try:
        # Register hooks
        print("\n1. Registering hooks...")
        hooks_1 = register_llama_hooks(model_1, "Model_1", max_layers=max_layers)
        hooks_2 = register_llama_hooks(model_2, "Model_2", max_layers=max_layers)
        
        if len(hooks_1) == 0 or len(hooks_2) == 0:
            raise Exception("Failed to register hooks")
        
        # Run models
        print("\n2. Running models...")
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs_1 = model_1(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs_2 = model_2(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n3. Activation capture results:")
        print(f"   Model 1: {len(activations_model_1)} layers captured")
        print(f"   Model 2: {len(activations_model_2)} layers captured")
        
        # Select neurons per token position
        print("\n4. Selecting neurons per token position...")
        selected_neurons = select_neurons_per_token_position(
            activations_model_1, activations_model_2, 
            selection_method=selection_method, seed=seed
        )
        
        # Compare activations
        print("\n5. Comparing activations...")
        comparison_results = compare_neuron_calculations_per_token(
            activations_model_1,
            activations_model_2,
            selected_neurons
        )
        
        print(f"\n6. Results summary:")
        print(f"   Layers with comparisons: {len(comparison_results)}")
        
        # Calculate overall statistics
        total_valid = sum(r['summary_stats'].get('valid_analyses', 0) for r in comparison_results.values())
        total_tokens = sum(r['summary_stats'].get('total_tokens', 0) for r in comparison_results.values())
        
        print(f"   Total valid analyses: {total_valid}")
        print(f"   Total token positions: {total_tokens}")
        
        return {
            'input_text': text_input,
            'tokenized_input': inputs,
            'model_1_output': outputs_1.logits,
            'model_2_output': outputs_2.logits,
            'layer_comparisons': comparison_results,
            'selected_neurons': selected_neurons,
            'hook_errors': hook_errors.copy(),
            'layers_captured_1': len(activations_model_1),
            'layers_captured_2': len(activations_model_2),
            'selection_method': selection_method,
            'total_valid_analyses': total_valid,
            'total_token_positions': total_tokens
        }
    
    except Exception as e:
        print(f"\nERROR in run_comparison_per_token: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'input_text': text_input,
            'error': str(e),
            'layer_comparisons': {},
            'selected_neurons': {},
            'hook_errors': hook_errors.copy(),
            'layers_captured_1': len(activations_model_1),
            'layers_captured_2': len(activations_model_2),
            'selection_method': selection_method
        }
    
    finally:
        remove_all_hooks()
        clear_activations()

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
# Run with your preferred method
PREFERRED_METHOD = 'min_diff'  # Change this to your preferred method

print(f"\n{'='*60}")
print(f"Running full analysis with method: {PREFERRED_METHOD}")
print(f"{'='*60}")

all_results = []

for i, text in enumerate(TEST_TEXTS):
    print(f"\n=== Processing text {i+1}/{len(TEST_TEXTS)} ===")
    
    try:
        result = run_comparison_per_token(
            text, 
            selection_method=PREFERRED_METHOD,
            seed=42+i,
            max_layers=None  # Use all layers
        )
        
        all_results.append(result)
        
        # Save detailed results
        save_detailed_results_per_token(
            result, 
            filename=f"all_texts_per_token_{PREFERRED_METHOD}.csv"
        )
        
        print(f"✓ Completed text {i+1}")
        
    except Exception as e:
        print(f"✗ Error processing text {i+1}: {e}")
        continue

print(f"\n{'='*60}")


