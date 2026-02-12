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

            # Safely detach (keep on same device to avoid CPU↔GPU round-trips)
            def safe_detach(tensor):
                if tensor is None:
                    return None
                try:
                    # Check if tensor is on meta device
                    if hasattr(tensor, 'device') and str(tensor.device) == 'meta':
                        return None
                    return tensor.detach()
                except Exception as e:
                    hook_errors.append(f"Detach error in {name}: {str(e)}")
                    return None

            captured_activations[name] = {
                'output': safe_detach(activation),
                'input': safe_detach(input_tensor),
                'weight': safe_detach(module.weight) if hasattr(module, 'weight') and module.weight is not None else None,
                'bias': safe_detach(module.bias) if hasattr(module, 'bias') and module.bias is not None else None
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

    # total_layers = len(model.model.layers)

    # for i in range(total_layers):
    #     layer = model.model.layers[i]
    #     layer_prefix = f"layer_{i}"
    #     components = [
    #         (layer.self_attn.q_proj, f"{layer_prefix}_attention_q"), (layer.self_attn.k_proj, f"{layer_prefix}_attention_k"),
    #         (layer.self_attn.v_proj, f"{layer_prefix}_attention_v"), (layer.self_attn.o_proj, f"{layer_prefix}_attention_output"),
    #         (layer.mlp.gate_proj, f"{layer_prefix}_mlp_gate"), (layer.mlp.up_proj, f"{layer_prefix}_mlp_up"),
    #         (layer.mlp.down_proj, f"{layer_prefix}_mlp_down"), (layer.input_layernorm, f"{layer_prefix}_input_norm"),
    #         (layer.post_attention_layernorm, f"{layer_prefix}_post_attn_norm"),
    #     ]
    #     for module, name in components:
    #         current_hooks.append(module.register_forward_hook(get_activation_hook(name)))
    
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
# =============================================================================
# RMSNorm and Gradient-Based Perturbation Functions
# =============================================================================

def sample_circle_target(
    original_probs: torch.Tensor,
    top1_idx: int,
    top2_idx: int,
    radius: float = 0.1,
  ) -> torch.Tensor:
    # Sample random point circle centered (p2, p1) with given radius
    # Circle equation: (x - p2)^2 + (y - p1)^2 = r^2
    # Random point:    (p2 + r*cos(θ), p1 + r*sin(θ))

    p1 = original_probs[top1_idx].item()  # top-1 (higher)
    p2 = original_probs[top2_idx].item()  # top-2 (lower)

    # Sample random angle, but enforce target_p2 > target_p1 (flip the ranking).
    # hahaha no other way solution
    for _ in range(1000):
        theta = np.random.uniform(0, 2 * np.pi)
        candidate_p1 = np.clip(p1 + radius * np.sin(theta), 1e-8, 1.0)
        candidate_p2 = np.clip(p2 + radius * np.cos(theta), 1e-8, 1.0)
        if candidate_p2 > candidate_p1:
            break

    target_p1 = candidate_p1
    target_p2 = candidate_p2

    # Build full target distribution: rescale remaining mass proportionally
    target_probs = original_probs.clone().detach().float()
    remaining_original = 1.0 - p1 - p2
    remaining_target = 1.0 - target_p1 - target_p2

    if remaining_original > 1e-12:# gemini recommendation
        scale = remaining_target / remaining_original
        target_probs *= scale
    else:
        target_probs[:] = remaining_target / len(target_probs)

    target_probs[top1_idx] = target_p1
    target_probs[top2_idx] = target_p2

    return target_probs, theta, target_p1, target_p2


def compute_swap_gradient(
    z: torch.Tensor,
    W: torch.Tensor,
    top1_idx: int,
    top2_idx: int,
    norm_layer: nn.Module,
    bias: Optional[torch.Tensor] = None,
    target_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Compute gradient of loss w.r.t. pre-norm activations z.

    z = z.clone().detach().requires_grad_(True)
    
    z_norm = norm_layer(z)
    logits = F.linear(z_norm, W, bias)
    probs = F.softmax(logits, dim=-1)
    
    if target_probs is not None:
        # KL(target || current) — drive current probs toward the circle target
        target = target_probs.to(z.device).detach()
        loss = F.kl_div(probs.log(), target, reduction='sum')
    else:
        # Original swap loss
        loss = probs[top1_idx] - probs[top2_idx]
    
    loss.backward()
    
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
def get_top_k_predictions(logits: torch.Tensor, tokenizer, k: int = 3) -> Dict[str, Any]:
    probs = F.softmax(logits, dim=-1)
    top_logits, top_indices = torch.topk(logits, k)
    top_probs = probs[top_indices]
    result = {}
    for i in range(k):
        idx = top_indices[i].item()
        result[f'top{i+1}_word'] = tokenizer.decode([idx])
        result[f'top{i+1}_index'] = idx
        result[f'top{i+1}_logit'] = top_logits[i].item()
        result[f'top{i+1}_softmax'] = top_probs[i].item()
    return result


def run_optimized_swap_attack(
    model, tokenizer, string_input, filename, epsilon_values, max_neurons_list
):
    input_id, input_text = string_input
    sample_input = tokenizer(input_text, return_tensors="pt")
    inputs_on_device = {k: v.to(model.device) for k, v in sample_input.items()}

    print(f"\n{'='*80}")
    print(f"Input ID: {input_id}, Input: '{input_text}'")
    print(f"{'='*80}")

    # Capture activations ONCE
    original_activations = run_model_and_capture_activations(model, inputs=inputs_on_device)
    try:
        pre_norm_activations = original_activations['final_norm']['input']
        if pre_norm_activations is None:
            print('I am making mistake check me!!!!')
            pre_norm_activations = original_activations['final_norm']['output']
        pre_norm_activations = pre_norm_activations.to(model.device).float()
    except KeyError:
        print("ERROR: Could not find 'final_norm' in activations.")
        return None

    z = pre_norm_activations[0, -1, :].float()
    norm_layer = model.model.norm
    W = model.lm_head.weight.detach().float()
    bias = model.lm_head.bias.detach().float() if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None else None
    hidden_size = z.shape[-1]
    max_budget = max(max_neurons_list)

    # Original predictions ONCE
    z_norm = norm_layer(z)
    original_logits = F.linear(z_norm, W, bias)
    top2_indices = torch.topk(original_logits, 2).indices
    top1_idx = top2_indices[0].item()
    top2_idx = top2_indices[1].item()
    original_top3 = get_top_k_predictions(original_logits, tokenizer, k=3)

    print(f"  Original top-1: '{original_top3['top1_word']}' (p={original_top3['top1_softmax']:.4f})")
    print(f"  Original top-2: '{original_top3['top2_word']}' (p={original_top3['top2_softmax']:.4f})")

    # Sample a random circle target: (p2 + r*cos(θ), p1 + r*sin(θ))
    original_probs = F.softmax(original_logits, dim=-1)
    circle_radius = 0.1
    target_probs, theta, target_p1, target_p2 = sample_circle_target(
        original_probs, top1_idx, top2_idx, radius=circle_radius
    )
    print(f"  Circle target: θ={theta:.4f} rad ({np.degrees(theta):.1f}°), r={circle_radius}")
    print(f"    target top-1 prob: {target_p1:.4f}  (Δ={target_p1 - original_top3['top1_softmax']:+.4f})")
    print(f"    target top-2 prob: {target_p2:.4f}  (Δ={target_p2 - original_top3['top2_softmax']:+.4f})")

    # Gradient toward circle target, special node, rankings — all computed ONCE
    gradient = compute_swap_gradient(z, W, top1_idx, top2_idx, norm_layer, bias, target_probs=target_probs)
    special_node_idx, special_node_score = identify_special_node(gradient, W)
    print(f"  Special node: idx={special_node_idx}, score={special_node_score:.4f}")

    bl_indices, bl_scores, bl_signs = rank_neurons_by_alignment(gradient, W)
    cn_indices, cn_scores, cn_signs = rank_neurons_by_alignment(gradient, W, [special_node_idx])

    # --- Nested epsilon x neuron loop (the core optimization) ---
    # Use KL divergence against the circle target distribution as the success criterion.
    target_probs_device = target_probs.to(z.device)
    target_log_probs = target_probs_device.log()

    def find_swap_points(sorted_indices, sorted_scores, signs):
        # For each epsilon, greedily add neurons and check if perturbed probs
        # are close enough to the circle target (KL divergence < threshold).
        results = []
        for epsilon in epsilon_values:
            z_mod = z.clone()
            perturbed = []
            found = False
            for k in range(min(max_budget, hidden_size)):
                print("Input {}, Processing neuron {}, epsilon: {}\r".format(input_id, k, epsilon),end='')
                neuron_idx = sorted_indices[k].item()
                if sorted_scores[k] == -float('inf'):
                    continue
                z_mod[neuron_idx] += signs[neuron_idx].item() * epsilon
                perturbed.append(neuron_idx)
                z_mod_norm = norm_layer(z_mod)
                new_logits = F.linear(z_mod_norm, W, bias)
                # KL divergence between perturbed probs and circle target
                new_log_probs = F.log_softmax(new_logits, dim=-1)
                kl_to_target = F.kl_div(new_log_probs, target_probs_device, reduction='sum', log_target=False)
                # Success: perturbed distribution is close enough to the circle target
                if kl_to_target < 0.01:
                    results.append({
                        'epsilon': epsilon, 'success': True,
                        'num_neurons': len(perturbed),
                        'kl_to_target': kl_to_target.item(),
                        'distances': compute_all_distances(original_logits, new_logits),
                        'top3': get_top_k_predictions(new_logits, tokenizer, k=3),
                        'special_node_used': special_node_idx in perturbed,
                        'top_token_changed': torch.argmax(new_logits) != top1_idx,
                    })
                    found = True
                    break
    
            if not found:
                z_mod_norm = norm_layer(z_mod)
                final_logits = F.linear(z_mod_norm, W, bias)
                new_log_probs = F.log_softmax(final_logits, dim=-1)
                kl_to_target = F.kl_div(new_log_probs, target_probs_device, reduction='sum', log_target=False)
                results.append({
                    'epsilon': epsilon, 'success': False,
                    'num_neurons': len(perturbed),
                    'kl_to_target': kl_to_target.item(),
                    'distances': compute_all_distances(original_logits, final_logits),
                    'top3': get_top_k_predictions(final_logits, tokenizer, k=3),
                    'special_node_used': special_node_idx in perturbed,
                })
        print("")
        return results

    print("  Computing baseline swap points...")
    bl_swap = find_swap_points(bl_indices, bl_scores, bl_signs)
    print("  Computing constrained swap points...")
    cn_swap = find_swap_points(cn_indices, cn_scores, cn_signs)

    # Build CSV records for each neuron budget
    records = []
    for mn in max_neurons_list:
        bl = next((r for r in bl_swap if r['success'] and r['num_neurons'] <= mn), bl_swap[-1])
        cn = next((r for r in cn_swap if r['success'] and r['num_neurons'] <= mn), cn_swap[-1])

        bl_ok = bl['success'] and bl['num_neurons'] <= mn
        cn_ok = cn['success'] and cn['num_neurons'] <= mn

        record = {
            'input_id': input_id,
            'allowed_neurons': mn,
            'special_node_idx': special_node_idx,
            'special_node_score': special_node_score,

            'baseline_success': bl_ok,
            'baseline_num_neurons': bl['num_neurons'],
            'baseline_epsilon': bl['epsilon'],
            'baseline_special_used': bl['special_node_used'],
            'baseline_total_magnitude': bl['num_neurons'] * abs(bl['epsilon']),
            'baseline_max_perturbation': abs(bl['epsilon']),
            **{f'baseline_{k}': v for k, v in bl['distances'].items()},
            **{f'baseline_final_{k}': v for k, v in bl['top3'].items()},

            'constrained_success': cn_ok,
            'constrained_num_neurons': cn['num_neurons'],
            'constrained_epsilon': cn['epsilon'],
            'constrained_special_avoided': not cn['special_node_used'],
            'constrained_total_magnitude': cn['num_neurons'] * abs(cn['epsilon']),
            'constrained_max_perturbation': abs(cn['epsilon']),
            **{f'constrained_{k}': v for k, v in cn['distances'].items()},
            **{f'constrained_final_{k}': v for k, v in cn['top3'].items()},

            **{f'orig_{k}': v for k, v in original_top3.items()},

            'neurons_diff': cn['num_neurons'] - bl['num_neurons'],
            'epsilon_diff': cn['epsilon'] - bl['epsilon'],
            'magnitude_diff': (cn['num_neurons'] * abs(cn['epsilon'])) - (bl['num_neurons'] * abs(bl['epsilon'])),
        }
        records.append(record)

    # Save all records at once
    df = pd.DataFrame(records)
    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode='a', header=not file_exists, index=False)

    del original_activations, pre_norm_activations
    clear_activations()

    bl_successes = sum(1 for r in records if r['baseline_success'])
    cn_successes = sum(1 for r in records if r['constrained_success'])
    print(f"  Saved {len(records)} records — baseline successes: {bl_successes}, constrained successes: {cn_successes}")
    return records

# %%
from datetime import datetime
date_of_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    [25,"The first person to walk on the moon was"],
    [26,"The largest desert in the world is"],
    [27,"The gas used by plants for photosynthesis is"],
    [28,"The powerhouse of the cell is the"],
    [29,"The author of the 'Harry Potter' series is"],
    [30,"The country with the largest population is"],
    [31,"The element with the atomic number 1 is"],
    [32,"The distance around a circle is called the"],
    [33,"The primary language spoken in Brazil is"],
    [34,"The tallest building in the world is"],
    [35,"The first President of the United States was"],
    [36,"The process by which liquid turns into gas is"],
    [37,"The red planet in our solar system is"],
    [38,"The number of continents on Earth is"],
    [39,"The hard white substance that covers teeth is"],
    [40,"The largest organ in the human body is"],
    [41,"The painting 'The Starry Night' was created by"],
    [42,"The instrument used to measure atmospheric pressure is"],
    [43,"The capital city of Italy is"],
    [44,"The three states of matter are solid, liquid, and"],
    [45,"The Olympic Games are held every how many years"],
    [46,"The fictional detective created by Arthur Conan Doyle is"],
    [47,"The layer of gas that protects Earth from UV radiation is"],
    [48,"The square root of 144 is"],
    [49,"The animal group that can live both on land and in water is"],
    [50,"The creator of the theory of relativity was"]
]

EPSILON_VALUES = [i * 0.1 for i in range(1, 1000, 1)]
max_neurons_list = list(range(1, 4096, 1))
OUTPUT_FILE = f"./gradient_swap_attack_optimized_{date_of_run}.csv"

print(f"Experiment date: {date_of_run}")
print(f"Epsilon values: {len(EPSILON_VALUES)} from {EPSILON_VALUES[0]} to {EPSILON_VALUES[-1]}")
print(f"Neuron budgets: {len(max_neurons_list)} from {max_neurons_list[0]} to {max_neurons_list[-1]}")
print(f"Output: {OUTPUT_FILE}")

for i, prompt in enumerate(sample_texts):
    print(f"\n>>>> Prompt {i+1}/{len(sample_texts)} <<<<")
    run_optimized_swap_attack(
        model=model, tokenizer=tokenizer, string_input=prompt,
        filename=OUTPUT_FILE, epsilon_values=EPSILON_VALUES,
        max_neurons_list=max_neurons_list
    )

print(f"\nAll done! Results saved to '{OUTPUT_FILE}'")

# %%



