# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import os
import gc
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple


# %%
# from huggingface_hub import login
# login()  

# %%
DEVICE = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

# %%
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# %%
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map=DEVICE
)
model.eval()

# %%
# =============================================================================
# Distance Metrics for Comparing Logit Distributions
# =============================================================================

def compute_l2_distance(original_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> float:
    # apply softmax to the logits
    original_logits = F.softmax(original_logits, dim=-1)
    perturbed_logits = F.softmax(perturbed_logits, dim=-1)
    # Compute L2 (Euclidean) distance between two logit vectors
    return torch.norm(original_logits - perturbed_logits, p=2).item()

def compute_cosine_distance(original_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> float:
    # Compute cosine distance (1 - cosine_similarity) between two logit vectors
    cos_sim = F.cosine_similarity(original_logits.unsqueeze(0), perturbed_logits.unsqueeze(0))
    return (1 - cos_sim).item()

def compute_kl_divergence(original_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> float:
    # Compute KL divergence: KL(original || perturbed) after softmax
    original_probs = F.softmax(original_logits, dim=-1)
    perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * (log_P - log_Q))
    kl_div = F.kl_div(perturbed_log_probs, original_probs, reduction='sum')
    return kl_div.item()

def compute_js_divergence(original_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> float:
   # Compute Jensen-Shannon divergence: 0.5*KL(P||M) + 0.5*KL(Q||M) where M = 0.5*(P+Q).
    P = F.softmax(original_logits, dim=-1)
    Q = F.softmax(perturbed_logits, dim=-1)
    M = 0.5 * (P + Q)
    
    # KL(P || M)
    kl_pm = F.kl_div(M.log(), P, reduction='sum')
    # KL(Q || M)
    kl_qm = F.kl_div(M.log(), Q, reduction='sum')
    
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div.item()

def compute_all_distances(original_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> Dict[str, float]:
    # Compute all distance metrics between original and perturbed logits.
    return {
        'l2_distance': compute_l2_distance(original_logits, perturbed_logits),
        'cosine_distance': compute_cosine_distance(original_logits, perturbed_logits),
        'kl_divergence': compute_kl_divergence(original_logits, perturbed_logits),
        'js_divergence': compute_js_divergence(original_logits, perturbed_logits),
    }

# %%
# =============================================================================
# RMSNorm and Gradient-Based Perturbation Functions
# =============================================================================

def compute_swap_gradient(
    z: torch.Tensor,
    W: torch.Tensor,
    top1_idx: int,
    top2_idx: int,
    norm_layer: nn.Module,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:

    # Compute gradient of swap loss w.r.t. pre-norm activations z.
    # The swap loss is: L = p[top1] - p[top2] We want to minimize this (decrease top1 prob, increase top2 prob).
    z = z.clone().detach().requires_grad_(True)
    
    # Forward pass through models RMSNorm
    z_norm = norm_layer(z)
    
    # Compute logits
    logits = F.linear(z_norm, W, bias)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Swap loss: minimize p[top1] - p[top2]
    # Gradient will point in direction that INCREASES this loss
    # So we negate it to get direction that DECREASES the loss (achieves swap)
    swap_loss = probs[top1_idx] - probs[top2_idx]
    
    # Backward pass
    swap_loss.backward()
    
    # Return negative gradient
    return -z.grad.detach()

def rank_neurons_by_alignment(
    gradient: torch.Tensor,
    W: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
   # Look here to fix the ranking of neurons but not needed
    w_col_norms = torch.norm(W, dim=0)  # [hidden_size]
    projections = gradient * w_col_norms  # [hidden_size]
    
    # Scores are absolute values
    scores = torch.abs(projections)
    
    # Sort by score descending
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    # Get signs for perturbation direction
    signs = torch.sign(projections)
    
    return sorted_indices, sorted_scores, signs

def find_min_neurons_for_swap(
    z: torch.Tensor,
    W: torch.Tensor,
    epsilon: float,
    norm_layer: nn.Module,
    bias: Optional[torch.Tensor] = None,
    max_neurons: Optional[int] = None
) -> Dict[str, Any]:
    # Find minimum number of neurons to perturb to achieve a swap of top-1 and top-2 predictions.

    hidden_size = z.shape[-1]
    if max_neurons is None:
        max_neurons = hidden_size
    
    # Compute original probabilities and find top-1, top-2
    z_norm = norm_layer(z)
    original_logits = F.linear(z_norm, W, bias)
    original_probs = F.softmax(original_logits, dim=-1)
    
    top2_values, top2_indices = torch.topk(original_logits, 2)
    top1_idx = top2_indices[0].item()
    top2_idx = top2_indices[1].item()
    
    # Compute gradient direction for swap
    gradient = compute_swap_gradient(z, W, top1_idx, top2_idx, norm_layer, bias)
    
    # Rank neurons by alignment with gradient
    sorted_indices, sorted_scores, signs = rank_neurons_by_alignment(gradient, W)
    
    # Greedy selection - add neurons one by one
    z_mod = z.clone()
    perturbed_neurons = []
    perturbations = {}
    
    for i in range(min(max_neurons, hidden_size)):
        neuron_idx = sorted_indices[i].item()
        
        # sign(v · W[:,i]) * epsilon
        delta = signs[neuron_idx].item() * epsilon
        z_mod[neuron_idx] += delta
        
        perturbed_neurons.append(neuron_idx)
        perturbations[neuron_idx] = delta
        
        # Check if swap is achieved
        z_mod_norm = norm_layer(z_mod)
        new_logits = F.linear(z_mod_norm, W, bias)
        new_probs = F.softmax(new_logits, dim=-1)
        
        new_top1_idx = torch.argmax(new_logits).item()
        
        # Swap achieved if original top-2 is now top-1
        if new_top1_idx == top2_idx:
            return {
                'success': True,
                'num_neurons': len(perturbed_neurons),
                'neuron_indices': perturbed_neurons,
                'perturbations': perturbations,
                'z_modified': z_mod,
                'original_probs': original_probs,
                'final_probs': new_probs,
                'original_logits': original_logits,
                'final_logits': new_logits,
                'original_top1': top1_idx,
                'original_top2': top2_idx,
                'final_top1': new_top1_idx,
            }
    
    # Failing to achieve swap within max_neurons
    z_mod_norm = norm_layer(z_mod)
    final_logits = F.linear(z_mod_norm, W, bias)
    final_probs = F.softmax(final_logits, dim=-1)
    
    return {
        'success': False,
        'num_neurons': len(perturbed_neurons),
        'neuron_indices': perturbed_neurons,
        'perturbations': perturbations,
        'z_modified': z_mod,
        'original_probs': original_probs,
        'final_probs': final_probs,
        'original_logits': original_logits,
        'final_logits': final_logits,
        'original_top1': top1_idx,
        'original_top2': top2_idx,
        'final_top1': torch.argmax(final_logits).item(),
    }

def compute_logit_shift_for_swap(p_top1: float, p_top2: float) -> float:
    # Compute the required logit shift to swap top-1 and top-2.
    # Convert probabilities to logits (log-odds)
    # For softmax with 2 classes: logit_diff = ln(p1/p2)
    # We need to flip the sign of this difference
    
    eps = 1e-10
    p_top1 = max(min(p_top1, 1 - eps), eps)
    p_top2 = max(min(p_top2, 1 - eps), eps)
    
    # The logit gap we need to overcome
    logit_gap = np.log(p_top1 / p_top2)
    
    # Need to shift by at least this much
    return abs(logit_gap) + 0.1  # Small margin for numerical stability

def estimate_required_alpha(
    z: torch.Tensor,
    W: torch.Tensor,
    p_top1: float,
    p_top2: float
) -> float:
   
    # Compute RMS of z
    rms_z = torch.sqrt(torch.mean(z ** 2)).item()
    
    # Average weight column norm (layer gain)
    w_col_norms = torch.norm(W, dim=0)  # [hidden_size]
    avg_w_norm = w_col_norms.mean().item()
    
    # Required logit shift
    delta_logit = compute_logit_shift_for_swap(p_top1, p_top2)
    
    # Estimator from equation
    alpha_total = (rms_z / avg_w_norm) * delta_logit
    
    return alpha_total

def estimate_min_neurons(
    z: torch.Tensor,
    W: torch.Tensor,
    epsilon: float,
    p_top1: float,
    p_top2: float
) -> int:
    # Estimate the minimum number of neurons needed for swap.
    alpha_total = estimate_required_alpha(z, W, p_top1, p_top2)
    
    # Each neuron contributes at most epsilon to the total effect
    k_estimate = int(np.ceil(alpha_total / epsilon))
    
    return max(1, k_estimate)

# %%
# Global variables for detailed activation capture
captured_activations = {}
current_hooks = []
hook_errors = []

def clear_activations():
    global captured_activations
    captured_activations.clear()
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

def get_activation_hook(name):
    def hook(module, input, output):
        global hook_errors
        try:
            # Handle different output types
            if output is None:
                activation = None
            elif isinstance(output, tuple):
                activation = output[0]
            elif hasattr(output, 'last_hidden_state'):
                # Handle model output objects
                activation = output.last_hidden_state
            else:
                activation = output
            
            # Handle input
            input_tensor = input[0] if isinstance(input, tuple) and len(input) > 0 else None

            # Safely detach and move to CPU
            def safe_detach_cpu(tensor):
                if tensor is None:
                    return None
                try:
                    # Check if tensor is on meta device
                    if hasattr(tensor, 'device') and str(tensor.device) == 'meta':
                        return None
                    return tensor.detach().cpu()
                except Exception as e:
                    hook_errors.append(f"Detach error in {name}: {str(e)}")
                    return None

            captured_activations[name] = {
                'output': safe_detach_cpu(activation),
                'input': safe_detach_cpu(input_tensor),
                'weight': safe_detach_cpu(module.weight) if hasattr(module, 'weight') and module.weight is not None else None,
                'bias': safe_detach_cpu(module.bias) if hasattr(module, 'bias') and module.bias is not None else None
            }
        except Exception as e:
            error_msg = f"Hook error in {name}: {str(e)}"
            hook_errors.append(error_msg)
            captured_activations[name] = {'output': None, 'input': None, 'weight': None, 'bias': None}
    return hook

def register_llama_hooks(model):
    global current_hooks
    remove_all_hooks() # clear any old hooks first
    hook_errors.clear()

    total_layers = len(model.model.layers)

    for i in range(total_layers):
        layer = model.model.layers[i]
        layer_prefix = f"layer_{i}"
        components = [
            (layer.self_attn.q_proj, f"{layer_prefix}_attention_q"), (layer.self_attn.k_proj, f"{layer_prefix}_attention_k"),
            (layer.self_attn.v_proj, f"{layer_prefix}_attention_v"), (layer.self_attn.o_proj, f"{layer_prefix}_attention_output"),
            (layer.mlp.gate_proj, f"{layer_prefix}_mlp_gate"), (layer.mlp.up_proj, f"{layer_prefix}_mlp_up"),
            (layer.mlp.down_proj, f"{layer_prefix}_mlp_down"), (layer.input_layernorm, f"{layer_prefix}_input_norm"),
            (layer.post_attention_layernorm, f"{layer_prefix}_post_attn_norm"),
        ]
        for module, name in components:
            current_hooks.append(module.register_forward_hook(get_activation_hook(name)))
    
    current_hooks.append(model.model.norm.register_forward_hook(get_activation_hook("final_norm")))
    current_hooks.append(model.lm_head.register_forward_hook(get_activation_hook("lm_head")))
    # print(f"Registered {len(current_hooks)} hooks.")

def run_model_and_capture_activations(model, inputs=None, inputs_embeds=None):
    global hook_errors
    clear_activations()
    register_llama_hooks(model)
    
    with torch.no_grad():
        if inputs is not None:
            _ = model(**inputs)
        elif inputs_embeds is not None:
            _ = model(inputs_embeds=inputs_embeds)
        else:
            raise ValueError("Either inputs or inputs_embeds must be provided.")
            
    remove_all_hooks()
    
    # Print any hook errors that occurred
    if hook_errors:
        print(f"WARNING: {len(hook_errors)} hook errors occurred:")
        for err in hook_errors[:5]:
            print(f"  - {err}")
        if len(hook_errors) > 5:
            print(f"  ... and {len(hook_errors) - 5} more")
    
    # return a copy of the captured activations
    return captured_activations.copy()

# %%
# Neuron Perturbation Sensitivity Analysis

def get_top_k_predictions(logits: torch.Tensor, tokenizer, k: int = 3) -> Dict[str, Any]:
    probs = F.softmax(logits, dim=-1)
    top_logits, top_indices = torch.topk(logits, k)
    top_probs = probs[top_indices]
    
    result = {}
    for i in range(k):
        idx = top_indices[i].item()
        word = tokenizer.decode([idx])
        result[f'top{i+1}_word'] = word
        result[f'top{i+1}_index'] = idx
        result[f'top{i+1}_logit'] = top_logits[i].item()
        result[f'top{i+1}_softmax'] = top_probs[i].item()
    
    return result

def get_activation_stats(z: torch.Tensor) -> Dict[str, float]:
    # Compute summary statistics for activation vector.
    return {
        'z_mean': z.mean().item(),
        'z_std': z.std().item(),
        'z_min': z.min().item(),
        'z_max': z.max().item(),
        'z_l2_norm': torch.norm(z, p=2).item(),
        'z_rms': torch.sqrt(torch.mean(z ** 2)).item(),
    }

def get_top_k_probs(probs: torch.Tensor, k: int = 20) -> Dict[str, float]:
   #Get top-k probability values (without token info, just the values).
    top_probs, top_indices = torch.topk(probs, k)
    result = {}
    for i in range(k):
        result[f'prob_rank{i+1}'] = top_probs[i].item()
        result[f'prob_rank{i+1}_idx'] = top_indices[i].item()
    return result

def compute_delta_stats(perturbations: Dict[int, float]) -> Dict[str, float]:
    # Compute statistics about the perturbation deltas.
    if not perturbations:
        return {
            'delta_total': 0.0,
            'delta_mean': 0.0,
            'delta_std': 0.0,
            'delta_min': 0.0,
            'delta_max': 0.0,
            'delta_l1': 0.0,
        }
    
    deltas = list(perturbations.values())
    deltas_tensor = torch.tensor(deltas)
    
    return {
        'delta_total': sum(abs(d) for d in deltas),  # L1 norm
        'delta_mean': np.mean(deltas),
        'delta_std': np.std(deltas) if len(deltas) > 1 else 0.0,
        'delta_min': min(deltas),
        'delta_max': max(deltas),
        'delta_l2': torch.norm(deltas_tensor, p=2).item(),
    }

def get_topk_activations(z: torch.Tensor, k: int = 20) -> Dict[str, float]:
   #Get top-k activation values (by absolute magnitude) from activation vector.
    abs_z = torch.abs(z)
    top_vals, top_indices = torch.topk(abs_z, k)
    
    result = {}
    for i in range(k):
        idx = top_indices[i].item()
        result[f'act_rank{i+1}_idx'] = idx
        result[f'act_rank{i+1}_val'] = z[idx].item()  # Original value (with sign)
        result[f'act_rank{i+1}_abs'] = top_vals[i].item()  # Absolute value
    return result

def run_gradient_swap_attack(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    pre_norm_activations: torch.Tensor,
    epsilon: float,
    input_id: int = 0,
    filename: str = "gradient_swap_attack_results.csv",
    top_k_probs: int = 20,  # Number of top probabilities to record
) -> Dict[str, Any]:
    
    # Get dimensions
    seq_len = pre_norm_activations.shape[1]
    hidden_size = pre_norm_activations.shape[2]
    last_token_pos = seq_len - 1
    
    # Get the pre-norm activation for the last token
    z = pre_norm_activations[0, last_token_pos, :].float()
    
    # Get models RMSNorm layer and lm_head weights
    norm_layer = model.model.norm  # The final RMSNorm layer
    W = model.lm_head.weight.detach().float()
    bias = model.lm_head.bias.detach().float() if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None else None
    
    # Get original predictions using models actual RMSNorm
    z_norm = norm_layer(z)
    original_logits = F.linear(z_norm, W, bias)
    original_probs = F.softmax(original_logits, dim=-1)
    
    top2_values, top2_indices = torch.topk(original_logits, 2)
    top1_idx = top2_indices[0].item()
    top2_idx = top2_indices[1].item()
    p_top1 = original_probs[top1_idx].item()
    p_top2 = original_probs[top2_idx].item()
    
    # Get original top-3 predictions for logging only
    original_top3 = get_top_k_predictions(original_logits, tokenizer, k=3)
    
    # Get activation stats for original z (before perturbation)
    orig_z_stats = get_activation_stats(z)
    orig_z_stats = {f'orig_{k}': v for k, v in orig_z_stats.items()}
    
    # Get top-k activations for input to final_norm (z) and output (z_norm)
    orig_z_topk = get_topk_activations(z, k=top_k_probs)
    orig_z_topk = {f'orig_z_{k}': v for k, v in orig_z_topk.items()}
    
    orig_znorm_topk = get_topk_activations(z_norm, k=top_k_probs)
    orig_znorm_topk = {f'orig_znorm_{k}': v for k, v in orig_znorm_topk.items()}
    
    # Get top-k probability values for original distribution
    orig_topk_probs = get_top_k_probs(original_probs, k=top_k_probs)
    orig_topk_probs = {f'orig_{k}': v for k, v in orig_topk_probs.items()}
    
    # Compute analytical estimate
    estimated_alpha = estimate_required_alpha(z, W, p_top1, p_top2)
    
    print(f"  Original top-1: '{tokenizer.decode([top1_idx])}' (p={p_top1:.4f})")
    print(f"  Original top-2: '{tokenizer.decode([top2_idx])}' (p={p_top2:.4f})")
    print(f"  Estimation of the required perturbation magnitude Alpha_total {estimated_alpha:.2f}")
    
    # Run the greedy swap attack using models actual RMSNorm
    result = find_min_neurons_for_swap(z, W, epsilon, norm_layer, bias, max_neurons=hidden_size)
    
    # Compute distances
    distances = compute_all_distances(
        result['original_logits'],
        result['final_logits']
    )
    
    # Get final top-3 predictions
    final_top3 = get_top_k_predictions(result['final_logits'], tokenizer, k=3)
    
    # Get activation stats for modified z (after perturbation)
    final_z_stats = get_activation_stats(result['z_modified'])
    final_z_stats = {f'final_{k}': v for k, v in final_z_stats.items()}
    
    # Get top-k activations for final z_mod and z_mod_norm
    final_z_topk = get_topk_activations(result['z_modified'], k=top_k_probs)
    final_z_topk = {f'final_z_{k}': v for k, v in final_z_topk.items()}
    
    final_z_mod_norm = norm_layer(result['z_modified'])
    final_znorm_topk = get_topk_activations(final_z_mod_norm, k=top_k_probs)
    final_znorm_topk = {f'final_znorm_{k}': v for k, v in final_znorm_topk.items()}
    
    # Get top-k probability values for final distribution
    final_topk_probs = get_top_k_probs(result['final_probs'], k=top_k_probs)
    final_topk_probs = {f'final_{k}': v for k, v in final_topk_probs.items()}
    
    # Compute delta statistics
    delta_stats = compute_delta_stats(result['perturbations'])
    
    # Build result record
    record = {
        'input_id': input_id,
        'epsilon': epsilon,
        'success': result['success'],
        'num_neurons': result['num_neurons'],
        'total_neurons': hidden_size,
        'soundness_ratio': result['num_neurons'] / hidden_size,
        'estimated_alpha': estimated_alpha,
        # Delta stats
        **delta_stats,
        # Activation summary stats
        **orig_z_stats,
        **final_z_stats,
        # Top-3 token predictions
        **{f'orig_{k}': v for k, v in original_top3.items()},
        **{f'final_{k}': v for k, v in final_top3.items()},
        # Distance metrics
        **distances,
        # Top-k activations: input to final_norm (z)
        **orig_z_topk,
        **final_z_topk,
        # Top-k activations: output of final_norm (z_norm)
        **orig_znorm_topk,
        **final_znorm_topk,
        # Top-k probability values
        **orig_topk_probs,
        **final_topk_probs,
        # Neuron indices
        'neuron_indices': str(result['neuron_indices'][:20]) + ('...' if len(result['neuron_indices']) > 20 else ''),
    }
    
    # Save to CSV
    df = pd.DataFrame([record])
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    # Print result
    if result['success']:
        print(f"  SUCCESS: Swap achieved with {result['num_neurons']} neurons ({result['num_neurons']/hidden_size*100:.2f}%)")
        print(f"  New top-1: '{tokenizer.decode([result['final_top1']])}'")
        print(f"  Delta L2: {delta_stats['delta_l2']:.4f}, Delta total: {delta_stats['delta_total']:.4f}")
    else:
        print(f"  FAILED: Could not achieve swap with {result['num_neurons']} neurons")
        print(f"  Current top-1: '{tokenizer.decode([result['final_top1']])}'")
        print(f"  Delta L2: {delta_stats['delta_l2']:.4f}, Delta total: {delta_stats['delta_total']:.4f}")
    
    return {
        **result,
        'distances': distances,
        'estimated_alpha': estimated_alpha,
        'delta_stats': delta_stats,
        'orig_z_stats': orig_z_stats,
        'final_z_stats': final_z_stats,
        'record': record,
    }

# %%W
# =============================================================================
# Main Workflow: Gradient-Based Swap Attack (soundness.pdf Algorithm 1)
# =============================================================================

def run_swap_attack_workflow(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    string_input: List,  # [input_id, input_text]
    epsilon: float = 1.0,
    filename: str = "gradient_swap_attack_results.csv",
) -> Dict[str, Any]:

    # Finds the minimum number of neurons to perturb (with max perturbation  e each) 
    # to swap the top-1 and top-2 predictions using gradient-based neuron ranking.

    input_id, input_text = string_input
    sample_input = tokenizer(input_text, return_tensors="pt")
    inputs_on_device = {k: v.to(model.device) for k, v in sample_input.items()}
    
    print(f"\n{'='*60}")
    print(f"Input ID: {input_id}")
    print(f"Input: '{tokenizer.decode(inputs_on_device['input_ids'][0])}'")
    print(f"Epsilon (max perturbation per neuron): {epsilon}")
    print(f"{'='*60}")
    
    # --- Step 1: Capture Pre-RMSNorm Activations ---
    #print("Running forward pass to capture activations...")
    
    # The final_norm input is the output of the last transformer layer
    original_activations = run_model_and_capture_activations(model, inputs=inputs_on_device)
    
    # Get pre-norm activations (input to final RMSNorm)
    try:
        # The input to final_norm is the pre-normalized activation
        pre_norm_activations = original_activations['final_norm']['input']
        if pre_norm_activations is None:
            # Fallback: use output and un-normalize (approximation)
            print("WARNING: Using final_norm output as approximation for pre-norm activations")
            pre_norm_activations = original_activations['final_norm']['output']
        pre_norm_activations = pre_norm_activations.to(model.device).float()
    except KeyError:
        print("ERROR: Could not find 'final_norm' in activations.")
        return None
    
    hidden_size = pre_norm_activations.shape[2]
    #print(f"Pre-norm activations shape: {pre_norm_activations.shape}")
    #print(f"Hidden size (N = total neurons): {hidden_size}")
    
    # --- Step 2: Run the Gradient-Based Swap Attack ---
    #print(f"\nRunning gradient-based swap attack...")
    
    result = run_gradient_swap_attack(
        model=model,
        tokenizer=tokenizer,
        pre_norm_activations=pre_norm_activations,
        epsilon=epsilon,
        input_id=input_id,
        filename=filename,
    )
    
    # Clean up
    del original_activations, pre_norm_activations
    clear_activations()
    
    # Print summary
    print(f"\n{'='*60}")
    # print("SUMMARY")
    print(f"  Success: {result['success']}")
    print(f"  Neurons perturbed: {result['num_neurons']} / {hidden_size} ({result['num_neurons']/hidden_size*100:.2f}%)")
    print(f"  L2 distance: {result['distances']['l2_distance']:.4f}")
    print(f"  JS divergence: {result['distances']['js_divergence']:.4f}")
    print(f"{'='*60}")
    
    return result

# %%
sample_texts = [
    [1,"The capital of France is"],
    [2,"The largest mammal on Earth is"],
    [3,"The process of photosynthesis occurs in"],
    [4,"The speed of light in a vacuum is"],
    [5,"The chemical symbol for gold is"],
    [6,"The human body has how many bones"],
    [7,"The Great Wall of China was built to"],
    [8,"Water boils at what temperature"],
    [9,"The smallest unit of matter is"],
    [10,"Shakespeare wrote the play"],
    [11,"The currency of Japan is"],
    [12,"Mount Everest is located in"],
    [13,"The inventor of the telephone was"],
    [14,"DNA stands for"],
    [15,"The largest ocean on Earth is"],
    [16,"The planet closest to the Sun is"],
    [17,"Gravity was discovered by"],
    [18,"The Amazon rainforest is primarily located in"],
    [19,"The freezing point of water is"],
    [20,"The most abundant gas in Earth's atmosphere is"],
    [21,"The Mona Lisa was painted by"],
    [22,"The longest river in the world is"],
    [23,"Photosynthesis converts carbon dioxide and water into"],
    [24,"The study of earthquakes is called"],
    [25,"The first person to walk on the moon was"]
]

# %%
# =============================================================================
# Run Gradient-Based Swap Attack on All Sample Texts
# =============================================================================

# Configuration
EPSILON = 0.1  # Maximum perturbation per neuron
OUTPUT_FILE = "gradient_swap_attack_results_other.csv"

# Store results
all_results = {}

# Loop through each prompt
for i, prompt in enumerate(sample_texts):
   # print(f"\n>>>> Starting Gradient Swap Attack for Prompt {i+1}/{len(sample_texts)} <<<<")
    
    result = run_swap_attack_workflow(
        model=model,
        tokenizer=tokenizer,
        string_input=prompt,
        epsilon=EPSILON,
        filename=OUTPUT_FILE,
    )
    
    if result is not None:
        all_results[prompt[0]] = result

print("\n\n<<<< ALL SWAP ATTACKS COMPLETE >>>>")
print(f"Results saved to '{OUTPUT_FILE}'")

# Overall summary
print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)
successes = sum(1 for r in all_results.values() if r['success'])
total = len(all_results)
print(f"  Success rate: {successes}/{total} ({successes/total*100:.1f}%)")

if all_results:
    avg_neurons = sum(r['num_neurons'] for r in all_results.values()) / len(all_results)
    avg_ratio = sum(r['num_neurons'] / r['record']['total_neurons'] for r in all_results.values()) / len(all_results)
    print(f"  Average neurons perturbed: {avg_neurons:.1f}")
    print(f"  Average soundness ratio: {avg_ratio*100:.2f}%")

# %%
# =============================================================================
# Quick Test: Run on a single input to verify output
# =============================================================================

# TEST_PROMPT = [1, "The capital of France is"]
# TEST_EPSILON = 1.0

# print("Running quick test (gradient-based swap attack)...")
# result = run_swap_attack_workflow(
#     model=model,
#     tokenizer=tokenizer,
#     string_input=TEST_PROMPT,
#     epsilon=TEST_EPSILON,
#     filename="test_swap_attack.csv",
# )
# print("\nTest complete! Check for output.")


# %%



