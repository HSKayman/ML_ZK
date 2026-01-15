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
    print(f"Registered {len(current_hooks)} hooks.")

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

def find_boundary_delta_binary_search(
    final_norm_activations: torch.Tensor,
    original_last_logits: torch.Tensor,
    neuron_indices: List[int],
    last_token_pos: int,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    distance_threshold: float,
    distance_metric: str,
    tokenizer,  # Added to compute top3 predictions
    delta_min: float = 0.0,
    delta_max: float = 100.0,
    tolerance: float = 0.001,
) -> Tuple[float, Dict[str, float], torch.Tensor, List[Dict]]:
    """
    Binary search to find the minimum delta where distance >= threshold.
    Returns (boundary_delta, distances_at_boundary, perturbed_logits_at_boundary, search_history)
    """
    
    low, high = delta_min, delta_max
    best_delta = delta_max
    best_distances = None
    best_logits = None
    search_history = []  # Track all (delta, distance) pairs
    step = 0
    
    while high - low > tolerance:
        mid = (low + high) / 2
        step += 1
        
        # Apply perturbation with delta=mid
        perturbed_activations = final_norm_activations.clone()
        for idx in neuron_indices:
            perturbed_activations[0, last_token_pos, idx] += mid
        
        # Compute logits
        perturbed_last = perturbed_activations[0, last_token_pos, :].to(lm_head_weight.device)
        perturbed_logits = F.linear(perturbed_last, lm_head_weight, lm_head_bias).float()
        
        # Compute distance
        distances = compute_all_distances(original_last_logits.to(perturbed_logits.device), perturbed_logits)
        current_distance = distances[distance_metric]
        
        # Get top-3 predictions for perturbed logits
        perturbed_top3 = get_top_k_predictions(perturbed_logits, tokenizer, k=3)
        perturbed_top3_prefixed = {f'pert_{key}': val for key, val in perturbed_top3.items()}
        
        # Record this step in search history
        search_history.append({
            'search_step': step,
            'delta_tried': mid,
            'exceeded_threshold': current_distance >= distance_threshold,
            **distances,
            **perturbed_top3_prefixed,
        })
        
        if current_distance >= distance_threshold:
            # Found a valid delta, try to find smaller
            best_delta = mid
            best_distances = distances
            best_logits = perturbed_logits
            high = mid
        else:
            # Need larger delta
            low = mid
    
    return best_delta, best_distances, best_logits, search_history

def analyze_neuron_perturbation_sensitivity(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    final_norm_activations: torch.Tensor,
    original_logits: torch.Tensor,
    k_values: List[int],
    distance_threshold: float = 0.1,
    distance_metric: str = "js_divergence",
    delta_max: float = 100.0,
    tolerance: float = 0.001,  # Binary search precision
    num_trials_per_k: int = 10,
    save_every: int = 1000,
    input_text: str = "",
    input_id: int = 0,
    filename: str = "neuron_perturbation_analysis_other.csv",
) -> Dict[int, List[float]]:
    # For each K and each trial, find minimum delta where distance >= threshold using binary search.
    # Returns: {K: [boundary_delta_trial_0, boundary_delta_trial_1, ...]}
    
    results = []
    total_saved = 0
    boundary_deltas = {}
    
    # Get dimensions
    seq_len = final_norm_activations.shape[1]
    hidden_size = final_norm_activations.shape[2] 
    last_token_pos = seq_len - 1
    
    # Get original logits for the last token
    original_last_logits = original_logits[0, last_token_pos, :].float()
    
    # Get top-3 predictions for original logits
    original_top3 = get_top_k_predictions(original_last_logits, tokenizer, k=3)
    original_top3_prefixed = {f'orig_{key}': val for key, val in original_top3.items()}
    
    # Get lm_head weights
    lm_head_weight = model.lm_head.weight.detach()
    lm_head_bias = model.lm_head.bias.detach() if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None else None
    
    # Helper to save batch
    def save_batch(batch):
        nonlocal total_saved
        if not batch:
            return
        df = pd.DataFrame(batch)
        df.insert(0, 'input_id', input_id)
        file_exists = os.path.exists(filename)
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
        total_saved += len(batch)
        print(f"[Checkpoint] Saved {len(batch)} rows (total: {total_saved})")
    
    # For each K value
    for k in k_values:
        if k > hidden_size:
            print(f"Warning: K={k} > hidden_size={hidden_size}, skipping.")
            continue
        
        boundary_deltas[k] = []
        
        for trial in range(num_trials_per_k):
            # Randomly select K neurons (fixed for this trial)
            neuron_indices = torch.randperm(hidden_size)[:k].tolist()
            
            # Binary search for boundary delta
            boundary_delta, distances, perturbed_logits, search_history = find_boundary_delta_binary_search(
                final_norm_activations=final_norm_activations,
                original_last_logits=original_last_logits,
                neuron_indices=neuron_indices,
                last_token_pos=last_token_pos,
                lm_head_weight=lm_head_weight,
                lm_head_bias=lm_head_bias,
                distance_threshold=distance_threshold,
                distance_metric=distance_metric,
                tokenizer=tokenizer,
                delta_min=delta_max*-0.1,
                delta_max=delta_max,
                tolerance=tolerance,
            )
            
            boundary_deltas[k].append(boundary_delta)
            
            # Record all search steps (each step is a row)
            for step_data in search_history:
                result = {
                    'num_neurons_changed': k,
                    'trial': trial,
                    'boundary_delta': boundary_delta,  # Final boundary
                    'neuron_indices': str(neuron_indices),
                    'total_neurons': hidden_size,
                    'soundness_ratio': k / hidden_size,
                    **step_data,  # search_step, delta_tried, exceeded_threshold, all distances
                    **original_top3_prefixed,
                }
                results.append(result)
            
            # Periodic save
            if len(results) >= save_every:
                save_batch(results)
                results = []
        
        # Progress
        avg_delta = sum(boundary_deltas[k]) / len(boundary_deltas[k])
        print(f"[K={k}] avg boundary_delta = {avg_delta:.4f}")
    
    # Save remaining
    if results:
        save_batch(results)
    
    print(f"--- Total saved: {total_saved} rows to {filename} ---")
    return boundary_deltas

# %%
# =============================================================================
# Main Workflow: Neuron Perturbation Sensitivity Analysis
# =============================================================================

def run_perturbation_analysis_workflow(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    string_input: List,  # [input_id, input_text]
    k_values: List[int],
    distance_threshold: float = 0.1,
    distance_metric: str = "js_divergence",
    delta_max: float = 100.0,
    tolerance: float = 0.001,
    num_trials_per_k: int = 10,
    save_every: int = 1000,
):

    input_id, input_text = string_input
    sample_input = tokenizer(input_text, return_tensors="pt")
    inputs_on_device = {k: v.to(model.device) for k, v in sample_input.items()}
    
    print(f"\n{'='*60}")
    print(f"Input ID: {input_id}")
    print(f"Input: '{tokenizer.decode(inputs_on_device['input_ids'][0])}'")
    print(f"K values: {k_values}")
    print(f"Threshold: {distance_threshold} ({distance_metric})")
    print(f"Delta search: max={delta_max}, tolerance={tolerance} (binary search)")
    print(f"Trials per K: {num_trials_per_k}")
    print(f"{'='*60}")
    
    # --- Step 1: Get Original Logits and Activations ---
    print("Running forward pass to capture original state...")
    
    with torch.no_grad():
        original_logits = model(**inputs_on_device).logits
    
    # Capture activations (we only need final_norm)
    original_activations = run_model_and_capture_activations(model, inputs=inputs_on_device)
    
    # Get final_norm output (layer n-1, input to lm_head)
    try:
        final_norm_output = original_activations['final_norm']['output'].to(model.device)
    except KeyError:
        print("ERROR: Could not find 'final_norm' in activations.")
        return
    
    hidden_size = final_norm_output.shape[2]
    print(f"Final norm output shape: {final_norm_output.shape}")
    print(f"Hidden size (N = total neurons): {hidden_size}")
    
    # Show original prediction
    last_token_logits = original_logits[0, -1, :]
    top_token_idx = torch.argmax(last_token_logits).item()
    print(f"Original top prediction: '{tokenizer.decode(top_token_idx)}' (ID: {top_token_idx})")
    
    # --- Step 2: Find boundary delta for each K (binary search) ---
    print(f"\nFinding boundary delta for each K (binary search)...")
    
    boundary_deltas = analyze_neuron_perturbation_sensitivity(
        model=model,
        tokenizer=tokenizer,
        final_norm_activations=final_norm_output,
        original_logits=original_logits,
        k_values=k_values,
        distance_threshold=distance_threshold,
        distance_metric=distance_metric,
        delta_max=delta_max,
        tolerance=tolerance,
        num_trials_per_k=num_trials_per_k,
        save_every=save_every,
        input_text=input_text,
        input_id=input_id,
    )
    
    # Clean up
    del original_activations, final_norm_output
    clear_activations()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Boundary Delta per K")
    for k, deltas in boundary_deltas.items():
        avg = sum(deltas) / len(deltas)
        print(f"  K={k:4d}: avg={avg:.4f}, min={min(deltas):.4f}, max={max(deltas):.4f}")
    print(f"{'='*60}")
    
    return boundary_deltas

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
# Run Perturbation Analysis on All Sample Texts
# =============================================================================

# Configuration
K_VALUES = [i for i in range(1, 1024)] # K values to test
DISTANCE_THRESHOLD = 0.15  # Stop when distance >= this
DISTANCE_METRIC = "js_divergence"  # Options: l2_distance, cosine_distance, kl_divergence, js_divergence
DELTA_MAX = 1000.0  # Maximum delta to search
TOLERANCE = 0.001  # Binary search precision
NUM_TRIALS_PER_K = 5000  # Trials per K
SAVE_EVERY = 1000  # Checkpoint frequency

# Store results
all_results = {}

# Loop through each prompt
for i, prompt in enumerate(sample_texts):
    print(f"\n>>>> Starting Perturbation Analysis for Prompt {i+1}/{len(sample_texts)} <<<<")
    
    boundary_deltas = run_perturbation_analysis_workflow(
        model=model,
        tokenizer=tokenizer,
        string_input=prompt,
        k_values=K_VALUES,
        distance_threshold=DISTANCE_THRESHOLD,
        distance_metric=DISTANCE_METRIC,
        delta_max=DELTA_MAX,
        tolerance=TOLERANCE,
        num_trials_per_k=NUM_TRIALS_PER_K,
        save_every=SAVE_EVERY,
    )
    
    all_results[prompt[0]] = boundary_deltas

print("\n\n<<<< ALL ANALYSES COMPLETE >>>>")
print(f"Results saved to 'neuron_perturbation_analysis.csv'")

# Overall summary
print("\n" + "="*60)
print("OVERALL: Average Boundary Delta per K (across all prompts)")
print("="*60)
for k in K_VALUES:
    all_deltas = []
    for deltas_dict in all_results.values():
        if k in deltas_dict:
            all_deltas.extend(deltas_dict[k])
    if all_deltas:
        print(f"  K={k:4d}: avg={sum(all_deltas)/len(all_deltas):.4f}")

# %%
# # =============================================================================
# # Quick Test: Run on a single input to verify output
# # =============================================================================

# # Test configuration
# TEST_PROMPT = [1, "The capital of France is"]
# TEST_K_VALUES = [1, 2]  
# TEST_THRESHOLD = 0.1  # Distance threshold
# TEST_METRIC = "js_divergence"
# TEST_DELTA_MAX = 1000.0
# TEST_TOLERANCE = 0.001
# TEST_TRIALS = 3  
# TEST_SAVE_EVERY = 1000

# print("Running quick test (binary search for boundary delta)...")
# boundary_deltas = run_perturbation_analysis_workflow(
#     model=model,
#     tokenizer=tokenizer,
#     string_input=TEST_PROMPT,
#     k_values=TEST_K_VALUES,
#     distance_threshold=TEST_THRESHOLD,
#     distance_metric=TEST_METRIC,
#     delta_max=TEST_DELTA_MAX,
#     tolerance=TEST_TOLERANCE,
#     num_trials_per_k=TEST_TRIALS,
#     save_every=TEST_SAVE_EVERY,
# )
# print("\nTest complete! Check for output.")


# %%



