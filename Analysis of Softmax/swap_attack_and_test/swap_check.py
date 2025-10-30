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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

# %%
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# %%
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# %%
def create_malicious_output(tokenizer, original_logits):

    last_token_logits = original_logits[0, -1, :].clone() # only last layer
    correct_token_idx = torch.argmax(last_token_logits).item()
    incorrect_token_idx = torch.argmin(last_token_logits).item()

    print("--- Logit Swap Attack ---")
    print(f"Original top prediction: '{tokenizer.decode(correct_token_idx)}' (ID: {correct_token_idx})")
    print(f"Target swap token:     '{tokenizer.decode(incorrect_token_idx)}' (ID: {incorrect_token_idx})")

    malicious_target_logits = last_token_logits.clone()
    correct_value = malicious_target_logits[correct_token_idx]
    incorrect_value = malicious_target_logits[incorrect_token_idx]

    malicious_target_logits[correct_token_idx] = incorrect_value
    malicious_target_logits[incorrect_token_idx] = correct_value

    print(f"New top prediction after swap: '{tokenizer.decode(torch.argmax(malicious_target_logits))}'\n")
    return malicious_target_logits.detach()

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
            activation = output[0] if isinstance(output, tuple) else output
            input_tensor = input[0] if isinstance(input, tuple) and len(input) > 0 else None

            captured_activations[name] = {
                'output': activation.detach().cpu() if activation is not None else None,
                'input': input_tensor.detach().cpu() if input_tensor is not None else None,
                'weight': module.weight.detach().cpu() if hasattr(module, 'weight') and module.weight is not None else None,
                'bias': module.bias.detach().cpu() if hasattr(module, 'bias') and module.bias is not None else None
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
    
    # return a copy of the captured activations
    return captured_activations.copy()

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
            calc_2 = w1[neuron_idx].item() * token_input[neuron_idx].item()
            
            if b1 is not None and neuron_idx < b1.shape[0]:
                calc_1 += b1[neuron_idx].item()
            if b2 is not None and neuron_idx < b2.shape[0]:
                calc_2 += b2[neuron_idx].item()
                
        else:
            # Linear layer calculation: input @ weight.T + bias
            if neuron_idx >= w1.shape[0]:
                return {'error': 'Neuron index out of bounds'}
                
            calc_1 = torch.matmul(token_input, w1[neuron_idx, :]).item()
            calc_2 = torch.matmul(token_input, w1[neuron_idx, :]).item()
            
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

# %%
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
    
def calculate_layer_output(
    layer_name: str,
    token_input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], str]:

    if token_input is None or weight is None:
        return None, "Missing input or weight"

    try:
        # Case 1: normalization layer (LayerNorm/RMSNorm)
        if 'norm' in layer_name:
            # PyTorch's functional LayerNorm which handles the formula:
            # y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
            calculated_output = F.layer_norm(
                token_input,
                normalized_shape=[token_input.shape[0]],
                weight=weight,
                bias=bias,
                eps=1e-5 # standard epsilon for Llama models
            )
            return calculated_output, "Success"

        # Case 2: linear projection (Attention, MLP, etc.)
        else:
            # y = x @ W^T + b
            calculated_output = F.linear(token_input, weight, bias)

            # Apply the SiLU activation function for specific MLP layers
            if 'mlp_gate' in layer_name or 'mlp_up' in layer_name:
                calculated_output = F.silu(calculated_output)

            return calculated_output, "Success"

    except Exception as e:
        return None, f"Calculation failed: {str(e)}"

# %%
# def analyze_calculation_vs_real_outputs(
#     original_activations: Dict[str, Dict[str, torch.Tensor]],
#     reconstructed_activations: Dict[str, Dict[str, torch.Tensor]]
# ) -> List[Dict[str, Any]]:

#     all_results = []
    
#     # Iterate over all layers captured in the original (benign) run
#     for layer_name in original_activations.keys():
        
#         # Ensure layer exists in both activation sets
#         if layer_name not in reconstructed_activations:
#             continue
            
#         orig_data = original_activations[layer_name]
#         recon_data = reconstructed_activations[layer_name]
        
#         def analyze_set(
#             data: Dict[str, torch.Tensor],
#             run_type: str
#         ) -> Optional[Dict[str, Any]]:
            
#             # 1. Get all necessary data
#             if not all(k in data and data[k] is not None for k in ['input', 'weight', 'output']):
#                 return None # Skip if data is incomplete

#             last_token_pos = data['input'].shape[1] - 1
#             token_input = data['input'][0, last_token_pos, :]
#             real_output = data['output'][0, last_token_pos, :]
#             weight = data['weight']
#             bias = data.get('bias') # Bias can be None

#             # 2. Call the calculation function
#             calculated_output, status = calculate_layer_output(
#                 layer_name, token_input, weight, bias
#             )
            
#             if calculated_output is None:
#                 print(f"Skipping {layer_name} ({run_type}): {status}")
#                 return None
                
#             # 3. Take the difference (the calculation error)
#             # This is the difference between what we calculated and what the model *actually* did
#             error_vector = calculated_output - real_output
            
#             # 4. Find the minimum one (minimum *absolute* error)
#             min_error_val, min_error_idx = torch.min(error_vector.abs(), dim=0)
#             min_error_idx = min_error_idx.item()
            
#             # 5. Select a random index
#             rand_idx = torch.randint(0, len(error_vector), (1,)).item()
            
#             # 6. Gather results as requested
#             result_entry = {
#                 'layer_name': layer_name,
#                 'run_type': run_type,
                
#                 # Randomly selected neuron
#                 'random_index': rand_idx,
#                 'random_index_real_value': real_output[rand_idx].item(),
#                 'random_index_calc_value': calculated_output[rand_idx].item(),
#                 'random_index_error': error_vector[rand_idx].item(),
                
#                 # Minimum error neuron
#                 'min_error_index': min_error_idx,
#                 'min_error_real_value': real_output[min_error_idx].item(),
#                 'min_error_calc_value': calculated_output[min_error_idx].item(),
#                 'min_error_value': error_vector[min_error_idx].item()
#             }
#             return result_entry


#         # Run the analysis for both the original and reconstructed sets
#         orig_analysis = analyze_set(orig_data, 'original')
#         recon_analysis = analyze_set(recon_data, 'reconstructed')
        
#         if orig_analysis:
#             all_results.append(orig_analysis)
#         if recon_analysis:
#             all_results.append(recon_analysis)
            
#     return all_results

# %%
# def analyze_calculation_vs_real_outputs(
#     original_activations: Dict[str, Dict[str, torch.Tensor]],
#     reconstructed_activations: Dict[str, Dict[str, torch.Tensor]],
# ) -> List[Dict[str, Any]]:    
#     all_results = []
    
#     for layer_name in original_activations.keys():
        
#         if layer_name not in reconstructed_activations:
#             continue
            
#         orig_data = original_activations[layer_name]
#         recon_data = reconstructed_activations[layer_name]
        
#         if not all(k in orig_data and orig_data[k] is not None for k in ['input', 'weight', 'output']) or \
#            not all(k in recon_data and recon_data[k] is not None for k in ['weight', 'output']):
#             continue

#         # Get the single input vector from the ORIGINAL data
#         last_token_pos = orig_data['input'].shape[1] - 1
#         orig_token_input = orig_data['input'][0, last_token_pos, :]

#         # --- 1. Analyze the Original Run ---
#         calc_orig, status_orig = calculate_layer_output(
#             layer_name, orig_token_input, orig_data['weight'], orig_data.get('bias')
#         )
#         if calc_orig is None: 
#             print(f"Skipping {layer_name} (original): {status_orig}")
#             continue
        
#         real_orig = orig_data['output'][0, last_token_pos, :]
#         error_vector_orig = calc_orig - real_orig

#         # --- 2. Analyze the Reconstructed Run (using ORIGINAL input) ---
#         calc_recon, status_recon = calculate_layer_output(
#             layer_name, orig_token_input, recon_data['weight'], recon_data.get('bias')
#         )
#         if calc_recon is None: continue
        
#         real_recon = recon_data['output'][0, last_token_pos, :]
#         # This error shows how much the reconstructed model deviates from its
#         # own hooked output when given the original benign input.
#         error_vector_recon = calc_recon - real_recon
        
#         # --- 3. Process results for both runs ---
        
#         # Find min and random indices for the ORIGINAL run's error
#         min_err_idx_orig = torch.argmin(error_vector_orig.abs()).item()
#         rand_idx_orig = torch.randint(0, len(error_vector_orig), (1,)).item()
        
#         all_results.append({
#             'layer_name': layer_name, 'run_type': 'original',
#             'random_index': rand_idx_orig,
#             'random_index_real_value': real_orig[rand_idx_orig].item(),
#             'random_index_calc_value': calc_orig[rand_idx_orig].item(),
#             'min_error_index': min_err_idx_orig,
#             'min_error_real_value': real_orig[min_err_idx_orig].item(),
#             'min_error_calc_value': calc_orig[min_err_idx_orig].item(),
#         })

#         # Find min and random indices for the RECONSTRUCTED run's error
#         min_err_idx_recon = torch.argmin(error_vector_recon.abs()).item()
#         rand_idx_recon = torch.randint(0, len(error_vector_recon), (1,)).item()
        
#         all_results.append({
#             'layer_name': layer_name, 'run_type': 'reconstructed_with_orig_input',
#             'random_index': rand_idx_recon,
#             'random_index_real_value': real_recon[rand_idx_recon].item(),
#             'random_index_calc_value': calc_recon[rand_idx_recon].item(),
#             'min_error_index': min_err_idx_recon,
#             'min_error_real_value': real_recon[min_err_idx_recon].item(),
#             'min_error_calc_value': calc_recon[min_err_idx_recon].item(),
#         })

#     return all_results

# %%
def analyze_calculation_vs_real_outputs(
    original_activations: Dict[str, Dict[str, torch.Tensor]],
    reconstructed_activations: Dict[str, Dict[str, torch.Tensor]],
    mode:str,
    n_rounds: int
) -> List[Dict[str, Any]]:    
    all_results = []
    if mode == 'min':
        for layer_name in original_activations.keys():
            
            if layer_name not in reconstructed_activations:
                continue
                
            orig_data = original_activations[layer_name]
            recon_data = reconstructed_activations[layer_name]
            
            if not all(k in orig_data and orig_data[k] is not None for k in ['input', 'weight', 'output']) or \
            not all(k in recon_data and recon_data[k] is not None for k in ['weight', 'output']):
                continue

            # Get the single input vector from the ORIGINAL data
            token_pos = orig_data['input'].shape[1] - 1  # will need to update here dont forgetttttttt !HSK!
            orig_token_input = orig_data['input'][0, token_pos, :]

            # --- 1. Analyze the Original Run ---
            calc_orig, status_orig = calculate_layer_output(
                layer_name, orig_token_input, orig_data['weight'], orig_data.get('bias')
            )
            if calc_orig is None: 
                print(f"Skipping {layer_name} (original): {status_orig}")
                continue
            
            real_orig = orig_data['output'][0, token_pos, :]
            error_vector_orig = calc_orig - real_orig

            # --- 2. Analyze the Reconstructed Run (using ORIGINAL input) ---
            calc_recon, status_recon = calculate_layer_output(
                layer_name, orig_token_input, recon_data['weight'], recon_data.get('bias')
            )
            if calc_recon is None: continue
            
            real_recon = recon_data['output'][0, token_pos, :]
            # This error shows how much the reconstructed model deviates from its
            # own hooked output when given the original benign input.
            error_vector_recon = calc_recon - real_recon
            
            # --- 3. Process results for both runs ---
            
            # Find min and random indices for the ORIGINAL run's error
            min_err_idx_orig = torch.argmin(error_vector_orig.abs()).item()
            #rand_idx_orig = torch.randint(0, len(error_vector_orig), (1,)).item() REMOVEW
            
            all_results.append({
                'round': -1,
                'layer_name': layer_name, 'run_type': 'original',
                'error_index': min_err_idx_orig,
                'error_real_value': real_orig[min_err_idx_orig].item(),
                'error_calc_value': calc_orig[min_err_idx_orig].item(),
            })

            # Find min and random indices for the RECONSTRUCTED run's error
            min_err_idx_recon = torch.argmin(error_vector_recon.abs()).item()
            
            all_results.append({
                'round': -1,
                'layer_name': layer_name, 'run_type': 'reconstructed_with_orig_input',
                'error_index': min_err_idx_recon,
                'error_real_value': real_recon[min_err_idx_recon].item(),
                'error_calc_value': calc_recon[min_err_idx_recon].item(),
            })
    else:
        for round in range(n_rounds):
            print(f"Analysis round {round+1}/{n_rounds}...")
            for layer_name in original_activations.keys():
                if layer_name not in reconstructed_activations:
                    continue
                    
                orig_data = original_activations[layer_name]
                recon_data = reconstructed_activations[layer_name]
                
                if not all(k in orig_data and orig_data[k] is not None for k in ['input', 'weight', 'output']) or \
                not all(k in recon_data and recon_data[k] is not None for k in ['weight', 'output']):
                    continue

                token_pos = orig_data['input'].shape[1] - 1
                orig_token_input = orig_data['input'][0, token_pos, :]
                num_neurons = orig_data['output'].shape[2]
                rand_idx = torch.randint(0, num_neurons, (1,)).item()
                
                # --- Handle Norm layers separately, as they need the full input context ---
                if 'norm' in layer_name:
                    calc_orig, _ = calculate_layer_output(layer_name, orig_token_input, orig_data['weight'], orig_data.get('bias'))
                    calc_recon, _ = calculate_layer_output(layer_name, orig_token_input, recon_data['weight'], recon_data.get('bias'))

                    calc_orig = calc_orig[rand_idx].item() if calc_orig is not None else None
                    calc_recon = calc_recon[rand_idx].item() if calc_recon is not None else None
                
                else:
                    # Slice the weight and bias for the randomly selected neuron
                    single_row_weight_orig = orig_data['weight'][rand_idx, :].unsqueeze(0) # Shape: [1, in_features]
                    single_row_weight_recon = recon_data['weight'][rand_idx, :].unsqueeze(0)
                    
                    bias_orig = orig_data.get('bias')
                    single_value_bias_orig = bias_orig[rand_idx].unsqueeze(0) if bias_orig is not None else None # Shape: [1]
                    
                    bias_recon = recon_data.get('bias')
                    single_value_bias_recon = bias_recon[rand_idx].unsqueeze(0) if bias_recon is not None else None

                    # Calculate output for the single neuron by passing its sliced weights
                    calc_orig_tensor, _ = calculate_layer_output(layer_name, orig_token_input, single_row_weight_orig, single_value_bias_orig)
                    calc_recon_tensor, _ = calculate_layer_output(layer_name, orig_token_input, single_row_weight_recon, single_value_bias_recon)
                    
                    # The result is a tensor with one value, so we extract it
                    calc_orig = calc_orig_tensor.item() if calc_orig_tensor is not None else None
                    calc_recon = calc_recon_tensor.item() if calc_recon_tensor is not None else None

                # --- Append results for the single random neuron ---
                if calc_orig is not None:
                    real_orig = orig_data['output'][0, token_pos, rand_idx].item()
                    all_results.append({
                        'round': round,
                        'layer_name': layer_name, 'run_type': 'original',
                        'error_index': rand_idx,
                        'error_real_value': real_orig,
                        'error_calc_value': calc_orig,
                    })

                if calc_recon is not None:
                    real_recon = recon_data['output'][0, token_pos, rand_idx].item()
                    all_results.append({
                        'round': round,
                        'layer_name': layer_name, 'run_type': 'reconstructed_with_orig_input', 
                        'error_index': rand_idx,
                        'error_real_value': real_recon,
                        'error_calc_value': calc_recon,
                    })

    return all_results

# %%
def save_analysis_results(
    results_list: List[Dict[str, Any]],
    input: str,
    recon_idx: int,
    filename: str = "attack_calc_error_analysis.csv"
):
    if not results_list:
        return
        
    df = pd.DataFrame(results_list)
    df.insert(0, 'reconstruction_idx', recon_idx)
    df.insert(0, 'input', input)
    
    # Append to the file if it exists, otherwise create it
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
    
    print(f"--- Saved {len(df)} analysis rows to {filename} ---")

# %%
def run_attack_and_analysis_workflow(
    model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    string_input: str,
    n_reconstructions: int = 3,
    n_test_rounds: int = 1,
    optimization_steps: int = 1500,
    learning_rate: float = 0.01,
    reg_loss_factor: float = 0.001
):
    sample_input = tokenizer(string_input,return_tensors="pt")
    inputs_on_device = {k: v.to(model.device) for k, v in sample_input.items()}
    
    print(f"\n{'='*60}")
    print(f"Input: '{tokenizer.decode(inputs_on_device['input_ids'][0])}'")
    print(f"{'='*60}")
    
    # --- Step 1: Get Original State ---
    with torch.no_grad():
        original_logits = model(**inputs_on_device).logits
    original_activations = run_model_and_capture_activations(
        model, inputs=inputs_on_device
    )
    
    # --- Step 2: Create Malicious Target ---
    malicious_target_logits = create_malicious_output(tokenizer, original_logits)

    # --- Step 3: Loop Through Reconstructions ---
    for recon_idx in range(n_reconstructions):
        print(f"\n--- [Recon {recon_idx+1}/{n_reconstructions}] Starting reconstruction ---")
        
        # Initialize random embeddings to optimize
        seq_len = inputs_on_device['input_ids'].shape[1]
        embed_dim = model.config.hidden_size
        reconstructed_embeds = torch.randn(
            1, seq_len, embed_dim,
            device=model.device, dtype=torch.float32, requires_grad=True
        )
        optimizer = optim.Adam([reconstructed_embeds], lr=learning_rate)

        for _ in tqdm(range(optimization_steps), desc=f"Optimizing Recon {recon_idx+1}", leave=False):
            optimizer.zero_grad()
            output_logits = model(inputs_embeds=reconstructed_embeds.to(model.dtype)).logits
            
            # Loss calculation
            loss = F.mse_loss(output_logits[0, -1, :].float(), malicious_target_logits.float())
            reg_loss = reg_loss_factor * torch.mean(reconstructed_embeds ** 2)
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < 1e-4:
                break
        
        print(f"Reconstruction complete. Final Loss: {total_loss.item():.6f}")

        # --- Step 4: Get Reconstructed State ---
        print("Capturing reconstructed activations...")
        final_embeds = reconstructed_embeds.detach().to(model.dtype)
        reconstructed_activations = run_model_and_capture_activations(
            model, inputs_embeds=final_embeds
        )

        # --- Step 5: Run Deep Analysis (using original input) ---
        print("Running deep calculation analysis...")
        analysis_results = analyze_calculation_vs_real_outputs(
                original_activations,
                reconstructed_activations,
                mode='min',
                n_rounds=-1
            )
        
        
        analysis_results.extend(analyze_calculation_vs_real_outputs(
            original_activations,
            reconstructed_activations,
            mode='random',
            n_rounds=n_test_rounds
        ))
        # --- Step 6: Save Results ---
        save_analysis_results(analysis_results, string_input,recon_idx)
        
        # Clean up memory
        del reconstructed_activations, final_embeds, analysis_results
        if 'clear_activations' in globals():
            globals()['clear_activations']() # Call clear_activations if it exists

    print(f"\n{'='*60}")
    print("Workflow complete.")
    print(f"Results are saved in 'attack_calc_error_analysis.csv'")
    print(f"{'='*60}")

# %%
sample_texts = [
    "The capital of France is",
    "The largest mammal on Earth is",
    "The process of photosynthesis occurs in",
    "The speed of light in a vacuum is",
    "The chemical symbol for gold is",
    "The human body has how many bones",
    "The Great Wall of China was built to",
    "Water boils at what temperature",
    "The smallest unit of matter is",
    "Shakespeare wrote the play",
    "The currency of Japan is",
    "Mount Everest is located in",
    "The inventor of the telephone was",
    "DNA stands for",
    "The largest ocean on Earth is",
    "The planet closest to the Sun is",
    "Gravity was discovered by",
    "The Amazon rainforest is primarily located in",
    "The freezing point of water is",
    "The most abundant gas in Earth's atmosphere is",
    "The Mona Lisa was painted by",
    "The longest river in the world is",
    "Photosynthesis converts carbon dioxide and water into",
    "The study of earthquakes is called",
    "The first person to walk on the moon was"
]

# %%
# Loop through each prompt and run the full workflow
for i, prompt in enumerate(sample_texts):
    
    print(f"\n>>>> Starting Analysis for Prompt {i+1} (Mode: min) <<<<")
    run_attack_and_analysis_workflow(
        model=model,
        tokenizer=tokenizer,
        string_input=prompt,
        n_reconstructions=300,
        n_test_rounds=5000,
        optimization_steps=5000,
        learning_rate=0.01,
        reg_loss_factor=0.001
    )

print("\n\n<<<< ALL TESTS COMPLETE >>>>")


