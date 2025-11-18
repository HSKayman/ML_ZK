# %%
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
from collections import defaultdict

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
    dtype=torch.float16,
    device_map="auto"
)

model_2 = LlamaForCausalLM.from_pretrained(
    MODEL_2_PATH,
    dtype=torch.float16,
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
        for depth in sorted(analysis['by_layer_depth'].keys()):  
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


