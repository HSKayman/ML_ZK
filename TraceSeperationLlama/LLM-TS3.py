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
from typing import List, Dict, Any, Optional, Tuple
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
            if 'mlp_gate' in layer_name: #or 'mlp_up' in layer_name:
                calculated_output = F.silu(calculated_output)

            return calculated_output, "Success"

    except Exception as e:
        return None, f"Calculation failed: {str(e)}"

# %%
def analyze_calculation_vs_real_outputs(
    original_activations: Dict[str, Dict[str, torch.Tensor]],
    other_activations: Dict[str, Dict[str, torch.Tensor]],
    mode:str,
    n_rounds: int,
    token_pos: int
) -> List[Dict[str, Any]]:    
    all_results = []
    if mode == 'min':
        for layer_name in original_activations.keys():
            
            if layer_name not in other_activations:
                continue
                
            orig_data = original_activations[layer_name]
            recon_data = other_activations[layer_name]
            
            if not all(k in orig_data and orig_data[k] is not None for k in ['input', 'weight', 'output']) or \
            not all(k in recon_data and recon_data[k] is not None for k in ['weight', 'output']):
                continue

            # Get the single input vector from the ORIGINAL data
            # token_pos = orig_data['input'].shape[1] - 1  # will need to update here dont forgetttttttt !HSK!
            recon_token_input = recon_data['input'][0, token_pos, :]
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
                layer_name, recon_token_input, orig_data['weight'], orig_data.get('bias')
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
                'layer_name': layer_name, 'run_type': 'other',
                'error_index': min_err_idx_recon,
                'error_real_value': real_recon[min_err_idx_recon].item(),
                'error_calc_value': calc_recon[min_err_idx_recon].item(),
            })
    else:
        for round in range(n_rounds):
            print(f"Analysis round {round+1}/{n_rounds}..."+"\r",end="")
            for layer_name in original_activations.keys():
                if layer_name not in other_activations:
                    continue
                    
                orig_data = original_activations[layer_name]
                recon_data = other_activations[layer_name]
                
                if not all(k in orig_data and orig_data[k] is not None for k in ['input', 'weight', 'output']) or \
                not all(k in recon_data and recon_data[k] is not None for k in ['weight', 'output']):
                    continue

                #token_pos = orig_data['input'].shape[1] - 1
                recon_token_input = recon_data['input'][0, token_pos, :]
                orig_token_input = orig_data['input'][0, token_pos, :]
                num_neurons = orig_data['output'].shape[2]
                rand_idx = torch.randint(0, num_neurons, (1,)).item()
                
                # --- Handle Norm layers separately, as they need the full input context ---
                if 'norm' in layer_name:
                    calc_orig, _ = calculate_layer_output(layer_name, orig_token_input, orig_data['weight'], orig_data.get('bias'))
                    calc_recon, _ = calculate_layer_output(layer_name, recon_token_input, orig_data['weight'], orig_data.get('bias'))

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
                    calc_recon_tensor, _ = calculate_layer_output(layer_name, recon_token_input, single_row_weight_orig, single_value_bias_orig)
                    
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
                        'layer_name': layer_name, 'run_type': 'other', 
                        'error_index': rand_idx,
                        'error_real_value': real_recon,
                        'error_calc_value': calc_recon,
                    })
        print()    

    return all_results

# %%
# %%
def save_analysis_results(
    results_list: List[Dict[str, Any]],
    input: str,
    token_pos: int,
    filename: str = "formula_analysis.csv"
):
    if not results_list:
        return
        
    df = pd.DataFrame(results_list)
    df.insert(0, 'token_pos', token_pos)
    df.insert(0, 'input', input)
    
    # Append to the file if it exists, otherwise create it
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
    
    print(f"--- Saved {len(df)} analysis rows to {filename} ---")

# %%
def run_test_and_analysis_workflow(
    original_model: "LlamaForCausalLM",
    other_model: "LlamaForCausalLM",
    tokenizer: "LlamaTokenizer",
    string_input: str,
    n_test_rounds: int = 1,
):
    sample_input = tokenizer(string_input[1],return_tensors="pt")
    inputs_on_device = {k: v.to(original_model.device) for k, v in sample_input.items()}
    
    print(f"\n{'='*60}")
    print(f"Input: '{tokenizer.decode(inputs_on_device['input_ids'][0])}'")
    print(f"{'='*60}")
    
    # --- Step 1: Get Original State ---
    original_activations = run_model_and_capture_activations(
        original_model, inputs=inputs_on_device
    )

    other_activations = run_model_and_capture_activations(
        other_model, inputs=inputs_on_device
    )
   
    for token_pos in range(inputs_on_device["input_ids"].shape[1]):
        # --- Step 2: Run Deep Analysis (using original input) ---
        print("Running deep calculation analysis...")
        analysis_results = analyze_calculation_vs_real_outputs(
                original_activations,
                other_activations,
                mode='min',
                n_rounds=-1,
                token_pos=token_pos
            )
        
        analysis_results.extend(analyze_calculation_vs_real_outputs(
            original_activations,
            other_activations,
            mode='random',
            n_rounds=n_test_rounds,
            token_pos=token_pos
        ))
        # --- Step 6: Save Results ---
        save_analysis_results(analysis_results, string_input[0],token_pos)
    
        # Clean up memory
        del analysis_results
        if 'clear_activations' in globals():
            globals()['clear_activations']()

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
for i, prompt in enumerate(sample_texts):
    
    print(f"\n>>>> Starting Analysis for Prompt {i+1}<<<<")
    run_test_and_analysis_workflow(
        original_model=model_1,
        other_model=model_2,
        tokenizer=tokenizer,
        string_input=prompt,
        n_test_rounds=5000,
    )

print("\n\n<<<< ALL TESTS COMPLETE >>>>")


