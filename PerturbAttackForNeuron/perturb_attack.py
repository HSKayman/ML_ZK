# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import gc
from typing import List, Dict, Any, Optional, Tuple

# %%

# %%
# from huggingface_hub import login
# login()  

# %%
# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

# %%
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

# %%
def rank_neurons_by_alignment(
    gradient: torch.Tensor,
    W: torch.Tensor,
    exclude_indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Rank neurons by alignment: gradient sensitivity weighted by W column norms
    # gradient is [hidden_size], W is [vocab_size, hidden_size]
    # W[:,i] column norm tells us how much neuron i affects outputs
    
    w_col_norms = torch.norm(W, dim=0)  # [hidden_size]
    projections = gradient * w_col_norms  # [hidden_size]
    
    # Scores are absolute values
    scores = torch.abs(projections)
    
    # If excluding certain neurons, set their scores to -inf so they're ranked last
    if exclude_indices is not None and len(exclude_indices) > 0:
        for idx in exclude_indices:
            scores[idx] = -float('inf')
    
    # Sort by score descending
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    # Get signs for perturbation direction
    signs = torch.sign(projections)
    
    return sorted_indices, sorted_scores, signs

# %%
def identify_special_node(
    gradient: torch.Tensor,
    W: torch.Tensor
) -> Tuple[int, float]:
    #special_node_idx: Index of the special neuron
    # special_node_score: Its impact score

    w_col_norms = torch.norm(W, dim=0)  # [hidden_size]
    projections = gradient * w_col_norms  # [hidden_size]
    scores = torch.abs(projections)
    
    special_node_idx = torch.argmax(scores).item()
    special_node_score = scores[special_node_idx].item()
    
    return special_node_idx, special_node_score

# %%
def find_min_neurons_for_swap(
    z: torch.Tensor,
    W: torch.Tensor,
    epsilon: float,
    norm_layer: nn.Module,
    bias: Optional[torch.Tensor] = None,
    max_neurons: Optional[int] = None,
    exclude_neurons: Optional[List[int]] = None,
    special_node_idx: Optional[int] = None
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
    
    # Rank neurons by alignment with gradient (excluding specified neurons)
    sorted_indices, sorted_scores, signs = rank_neurons_by_alignment(gradient, W, exclude_neurons)
    
    # Greedy selection - add neurons one by one
    z_mod = z.clone()
    perturbed_neurons = []
    perturbations = {}
    special_node_used = False
    special_node_rank = None
    
    # Track where special node appears in ranking
    if special_node_idx is not None:
        for rank, idx in enumerate(sorted_indices):
            if idx.item() == special_node_idx:
                special_node_rank = rank
                break
    
    for i in range(min(max_neurons, hidden_size)):
        neuron_idx = sorted_indices[i].item()
        
        # Skip if this neuron has -inf score (was excluded)
        if sorted_scores[i] == -float('inf'):
            continue
        
        # Check if we're using the special node
        if special_node_idx is not None and neuron_idx == special_node_idx:
            special_node_used = True
        
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
        if new_top1_idx != top1_idx:
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
                'special_node_used': special_node_used,
                'special_node_rank': special_node_rank,
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
        'special_node_used': special_node_used,
        'special_node_rank': special_node_rank,
    }


# %%

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

# %%
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

# %%
def find_min_neurons_with_adaptive_epsilon(
    z: torch.Tensor,
    W: torch.Tensor,
    norm_layer: nn.Module,
    bias: Optional[torch.Tensor] = None,
    max_neurons: Optional[int] = None,
    exclude_neurons: Optional[List[int]] = None,
    special_node_idx: Optional[int] = None,
    epsilon_values: Optional[List[float]] = None
) -> Dict[str, Any]:

    # Find minimum neurons for swap with adaptive epsilon.
    # Try increasing epsilon values until output changes are achieved.
   
    if epsilon_values is None:
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0,20.0,50.0,100.0]
    
    hidden_size = z.shape[-1]
    if max_neurons is None:
        max_neurons = hidden_size
    
    best_result = None
    
    for epsilon in epsilon_values:
        result = find_min_neurons_for_swap(
            z=z,
            W=W,
            epsilon=epsilon,
            norm_layer=norm_layer,
            bias=bias,
            max_neurons=max_neurons,
            exclude_neurons=exclude_neurons,
            special_node_idx=special_node_idx
        )
        
        result['epsilon_used'] = epsilon
        
        # Compute total perturbation magnitude
        if result['perturbations']:
            perturbation_values = list(result['perturbations'].values())
            result['total_perturbation_magnitude'] = sum(abs(d) for d in perturbation_values)
            result['max_single_perturbation'] = max(abs(d) for d in perturbation_values)
        else:
            result['total_perturbation_magnitude'] = 0.0
            result['max_single_perturbation'] = 0.0
        
        if result['success']:
            return result
        
        # Keep track of best attempt (most neurons perturbed)
        if best_result is None or result['num_neurons'] > best_result['num_neurons']:
            best_result = result
    
    # If no epsilon achieved swap, return the best attempt
    return best_result if best_result is not None else result

# %%
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

def run_gradient_swap_attack_with_special_node(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    pre_norm_activations: torch.Tensor,
    input_id: int = 0,
    filename: str = "gradient_swap_attack_special_node_results.csv",
    use_adaptive_epsilon: bool = True,
    max_neurons: Optional[int] = None,
    epsilon_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    # Run gradient swap attack with special node monitoring.
    # Compares baseline (can use any neuron) vs constrained (avoiding special node).

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
    p_top2 = original_probs[top2_idx].item() # INeeficient part suca removed by using topk
    
    # Get original top-3 predictions for logging
    original_top3 = get_top_k_predictions(original_logits, tokenizer, k=3)
    
    # Compute analytical estimate
    estimated_alpha = estimate_required_alpha(z, W, p_top1, p_top2)
    
    # Compute gradient for special node identification
    gradient = compute_swap_gradient(z, W, top1_idx, top2_idx, norm_layer, bias)
    
    # Identify the special node
    special_node_idx, special_node_score = identify_special_node(gradient, W)
    
    print(f"  Original top-1: '{tokenizer.decode([top1_idx])}' (p={p_top1:.4f})")
    print(f"  Original top-2: '{tokenizer.decode([top2_idx])}' (p={p_top2:.4f})")
    print(f"  Special node: idx={special_node_idx}, score={special_node_score:.4f}")
    print(f"  Estimated Alpha_total: {estimated_alpha:.2f}")
    
    # Run BASELINE attack (can use any neuron, including special node)
    print(f"\n  Running BASELINE attack (no constraints)...")
    if use_adaptive_epsilon:
        baseline_result = find_min_neurons_with_adaptive_epsilon(
            z=z, W=W, norm_layer=norm_layer, bias=bias,
            max_neurons=max_neurons, exclude_neurons=None,
            special_node_idx=special_node_idx, epsilon_values=epsilon_values
        )
    else:
        baseline_result = find_min_neurons_for_swap(
            z=z, W=W, epsilon=epsilon_values[0] if epsilon_values else 0.1,
            norm_layer=norm_layer, bias=bias, max_neurons=max_neurons,
            exclude_neurons=None, special_node_idx=special_node_idx
        )
        baseline_result['epsilon_used'] = epsilon_values[0] if epsilon_values else 0.1
    
    # Compute distances for baseline
    baseline_distances = compute_all_distances(
        baseline_result['original_logits'],
        baseline_result['final_logits']
    )
    
    # Get final top-3 for baseline
    baseline_top3 = get_top_k_predictions(baseline_result['final_logits'], tokenizer, k=3)
    
    print(f"  BASELINE: success={baseline_result['success']}, neurons={baseline_result['num_neurons']}, "
          f"epsilon={baseline_result.get('epsilon_used', 'N/A')}, special_used={baseline_result.get('special_node_used', False)}")
    
    # Run CONSTRAINED attack (must avoid special node)
    print(f"\n  Running CONSTRAINED attack (avoiding special node {special_node_idx})...")
    if use_adaptive_epsilon:
        constrained_result = find_min_neurons_with_adaptive_epsilon(
            z=z, W=W, norm_layer=norm_layer, bias=bias,
            max_neurons=max_neurons, exclude_neurons=[special_node_idx],
            special_node_idx=special_node_idx, epsilon_values=epsilon_values
        )
    else:
        constrained_result = find_min_neurons_for_swap(
            z=z, W=W, epsilon=epsilon_values[0] if epsilon_values else 0.1,
            norm_layer=norm_layer, bias=bias, max_neurons=max_neurons,
            exclude_neurons=[special_node_idx], special_node_idx=special_node_idx
        )
        constrained_result['epsilon_used'] = epsilon_values[0] if epsilon_values else 0.1
    
    # Compute distances for constrained
    constrained_distances = compute_all_distances(
        constrained_result['original_logits'],
        constrained_result['final_logits']
    )
    
    # Get final top-3 for constrained
    constrained_top3 = get_top_k_predictions(constrained_result['final_logits'], tokenizer, k=3)
    
    print(f"  CONSTRAINED: success={constrained_result['success']}, neurons={constrained_result['num_neurons']}, "
          f"epsilon={constrained_result.get('epsilon_used', 'N/A')}, special_avoided={not constrained_result.get('special_node_used', True)}")
    
    # Build comparison record
    record = {
        'input_id': input_id,
        'allowed_neurons': max_neurons,
        #'total_neurons': hidden_size,
        #'estimated_alpha': estimated_alpha,
        
        # Special node info
        'special_node_idx': special_node_idx,
        'special_node_score': special_node_score,
        #'special_node_rank_baseline': baseline_result.get('special_node_rank', -1),
        #'special_node_rank_constrained': constrained_result.get('special_node_rank', -1),
        
        # Baseline attack results
        'baseline_success': baseline_result['success'],
        'baseline_num_neurons': baseline_result['num_neurons'],
        'baseline_epsilon': baseline_result.get('epsilon_used', 0.0),
        'baseline_special_used': baseline_result.get('special_node_used', False),
        'baseline_total_magnitude': baseline_result.get('total_perturbation_magnitude', 0.0),
        'baseline_max_perturbation': baseline_result.get('max_single_perturbation', 0.0),
        **{f'baseline_{k}': v for k, v in baseline_distances.items()},
        **{f'baseline_final_{k}': v for k, v in baseline_top3.items()},
        
        # Constrained attack results
        'constrained_success': constrained_result['success'],
        'constrained_num_neurons': constrained_result['num_neurons'],
        'constrained_epsilon': constrained_result.get('epsilon_used', 0.0),
        'constrained_special_avoided': not constrained_result.get('special_node_used', True),
        'constrained_total_magnitude': constrained_result.get('total_perturbation_magnitude', 0.0),
        'constrained_max_perturbation': constrained_result.get('max_single_perturbation', 0.0),
        **{f'constrained_{k}': v for k, v in constrained_distances.items()},
        **{f'constrained_final_{k}': v for k, v in constrained_top3.items()},
        
        # Original predictions (same for both)
        **{f'orig_{k}': v for k, v in original_top3.items()},
        
        # Comparison metrics
        'neurons_diff': constrained_result['num_neurons'] - baseline_result['num_neurons'],
        'epsilon_diff': constrained_result.get('epsilon_used', 0.0) - baseline_result.get('epsilon_used', 0.0),
        'magnitude_diff': constrained_result.get('total_perturbation_magnitude', 0.0) - baseline_result.get('total_perturbation_magnitude', 0.0),
    }
    
    # Save to CSV
    df = pd.DataFrame([record])
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    return {
        'baseline': baseline_result,
        'constrained': constrained_result,
        'special_node_idx': special_node_idx,
        'special_node_score': special_node_score,
        'record': record,
    }

# %%

def run_swap_attack_workflow_with_special_node(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    string_input: List,  # [input_id, input_text]
    filename: str = "gradient_swap_attack_special_node_results.csv",
    use_adaptive_epsilon: bool = True,
    epsilon_values: Optional[List[float]] = None,
    max_neurons_list: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run gradient-based swap attack with special node monitoring.
    Compares baseline vs constrained attacks.
    """
    input_id, input_text = string_input
    sample_input = tokenizer(input_text, return_tensors="pt")
    inputs_on_device = {k: v.to(model.device) for k, v in sample_input.items()}
    
    print(f"\n{'='*80}")
    print(f"Input ID: {input_id}")
    print(f"Input: '{tokenizer.decode(inputs_on_device['input_ids'][0])}'")
    print(f"Mode: {'Adaptive Epsilon' if use_adaptive_epsilon else 'Fixed Epsilon'}")
    if epsilon_values:
        print(f"Epsilon values: {epsilon_values}")
    print(f"{'='*80}")
    
    # Capture Pre-RMSNorm Activations
    original_activations = run_model_and_capture_activations(model, inputs=inputs_on_device)
    
    # Get pre-norm activations (input to final RMSNorm)
    try:
        pre_norm_activations = original_activations['final_norm']['input']
        if pre_norm_activations is None:
            print("WARNING: Using final_norm output as approximation for pre-norm activations")
            pre_norm_activations = original_activations['final_norm']['output']
        pre_norm_activations = pre_norm_activations.to(model.device).float()
    except KeyError:
        print("ERROR: Could not find 'final_norm' in activations.")
        return None
    
    hidden_size = pre_norm_activations.shape[2]
    for max_neurons in max_neurons_list:
        # Run the Special Node Attack
        result = run_gradient_swap_attack_with_special_node(
            model=model,
            tokenizer=tokenizer,
            pre_norm_activations=pre_norm_activations,
            input_id=input_id,
            filename=filename,
            use_adaptive_epsilon=use_adaptive_epsilon,
            epsilon_values=epsilon_values,
            max_neurons=max_neurons,
        )
    
    # Clean up
    del original_activations, pre_norm_activations
    clear_activations()
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"  Special Node: idx={result['special_node_idx']}, score={result['special_node_score']:.4f}")
    print(f"\n  BASELINE (unrestricted):")
    print(f"    Success: {result['baseline']['success']}")
    print(f"    Neurons: {result['baseline']['num_neurons']} / {hidden_size}")
    print(f"    Epsilon: {result['baseline'].get('epsilon_used', 'N/A')}")
    print(f"    Special node used: {result['baseline'].get('special_node_used', False)}")
    print(f"\n  CONSTRAINED (avoiding special node):")
    print(f"    Success: {result['constrained']['success']}")
    print(f"    Neurons: {result['constrained']['num_neurons']} / {hidden_size}")
    print(f"    Epsilon: {result['constrained'].get('epsilon_used', 'N/A')}")
    print(f"    Special node avoided: {not result['constrained'].get('special_node_used', True)}")
    print(f"\n  DIFFERENCE (constrained - baseline):")
    print(f"    Neurons: {result['record']['neurons_diff']:+d}")
    print(f"    Epsilon: {result['record']['epsilon_diff']:+.2f}")
    print(f"    Total magnitude: {result['record']['magnitude_diff']:+.4f}")
    print(f"{'='*80}")
    
    return result

# %%
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
# %%
# =============================================================================
# Run Special Node Monitoring Experiments
# =============================================================================
# get date
from datetime import datetime
date_of_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Experiment date: {date_of_run}")

# Configuration for Special Node Experiments
USE_ADAPTIVE_EPSILON = True  # Increase epsilon until output changes
EPSILON_VALUES = [i*0.1 for i in range(1,1000,1)]  # Try these epsilon values
OUTPUT_FILE_SPECIAL = f"./gradient_swap_attack_special_node_results_{date_of_run}.csv"
max_neurons_list = [ i for i in range(1,4096,1)]
# Store results
all_special_results = {}

print("\n" + "="*80)
print("SPECIAL NODE MONITORING EXPERIMENTS")
print("="*80)
print(f"Adaptive epsilon: {USE_ADAPTIVE_EPSILON}")
print(f"Epsilon progression: {EPSILON_VALUES}")
print(f"Output file: {OUTPUT_FILE_SPECIAL}")
print("="*80)

# Loop through each prompt
for i, prompt in enumerate(sample_texts):
    print(f"\n>>>> Prompt {i+1}/{len(sample_texts)} <<<<")
    
    result = run_swap_attack_workflow_with_special_node(
        model=model,
        tokenizer=tokenizer,
        string_input=prompt,
        filename=OUTPUT_FILE_SPECIAL,
        use_adaptive_epsilon=USE_ADAPTIVE_EPSILON,
        epsilon_values=EPSILON_VALUES,
        max_neurons_list=max_neurons_list
    )
    
    if result is not None:
        all_special_results[prompt[0]] = result

# %%

print("\n\n" + "="*80)
print("<<<< ALL SPECIAL NODE EXPERIMENTS COMPLETE >>>>")
print("="*80)
print(f"Results saved to '{OUTPUT_FILE_SPECIAL}'")


# Overall summary
print("\n" + "="*80)
print("OVERALL SUMMARY - SPECIAL NODE EXPERIMENTS")
print("="*80)

if all_special_results:
    # Baseline stats
    baseline_successes = sum(1 for r in all_special_results.values() if r['baseline']['success'])
    constrained_successes = sum(1 for r in all_special_results.values() if r['constrained']['success'])
    total = len(all_special_results)
    
    print(f"\nSuccess Rates:")
    print(f"  Baseline (unrestricted): {baseline_successes}/{total} ({baseline_successes/total*100:.1f}%)")
    print(f"  Constrained (avoiding special node): {constrained_successes}/{total} ({constrained_successes/total*100:.1f}%)")
    
    # Average neurons
    avg_baseline_neurons = sum(r['baseline']['num_neurons'] for r in all_special_results.values()) / total
    avg_constrained_neurons = sum(r['constrained']['num_neurons'] for r in all_special_results.values()) / total
    avg_diff_neurons = avg_constrained_neurons - avg_baseline_neurons
    
    print(f"\nAverage Neurons Perturbed:")
    print(f"  Baseline: {avg_baseline_neurons:.1f}")
    print(f"  Constrained: {avg_constrained_neurons:.1f}")
    print(f"  Difference: {avg_diff_neurons:+.1f} ({(avg_diff_neurons/avg_baseline_neurons)*100:+.1f}%)")
    
    # Average epsilon
    avg_baseline_epsilon = sum(r['baseline'].get('epsilon_used', 0) for r in all_special_results.values()) / total
    avg_constrained_epsilon = sum(r['constrained'].get('epsilon_used', 0) for r in all_special_results.values()) / total
    avg_diff_epsilon = avg_constrained_epsilon - avg_baseline_epsilon
    
    print(f"\nAverage Epsilon Used:")
    print(f"  Baseline: {avg_baseline_epsilon:.2f}")
    print(f"  Constrained: {avg_constrained_epsilon:.2f}")
    print(f"  Difference: {avg_diff_epsilon:+.2f}")
    
    # Average total magnitude
    avg_baseline_mag = sum(r['baseline'].get('total_perturbation_magnitude', 0) for r in all_special_results.values()) / total
    avg_constrained_mag = sum(r['constrained'].get('total_perturbation_magnitude', 0) for r in all_special_results.values()) / total
    avg_diff_mag = avg_constrained_mag - avg_baseline_mag
    
    print(f"\nAverage Total Perturbation Magnitude:")
    print(f"  Baseline: {avg_baseline_mag:.4f}")
    print(f"  Constrained: {avg_constrained_mag:.4f}")
    print(f"  Difference: {avg_diff_mag:+.4f} ({(avg_diff_mag/avg_baseline_mag)*100:+.1f}%)")
    
    # Special node usage stats
    special_used_count = sum(1 for r in all_special_results.values() if r['baseline'].get('special_node_used', False))
    print(f"\nSpecial Node Usage in Baseline:")
    print(f"  Used: {special_used_count}/{total} ({special_used_count/total*100:.1f}%)")
    print(f"  Not used: {total - special_used_count}/{total} ({(total - special_used_count)/total*100:.1f}%)")

print("="*80)

# %%






