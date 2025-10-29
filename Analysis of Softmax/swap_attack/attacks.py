# %%
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars
import os

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

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
def get_original_activations_and_logits(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    original_hidden_states = outputs.hidden_states
    
    return outputs.logits, original_hidden_states

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

def reconstruct_internal_state(model, original_hidden_states, malicious_target_logits, epoch=200, lr=0.01):
    # first hidden state is the input embedding
    input_embeddings = original_hidden_states[0].detach()
    
    reconstructed_states = []
    for i in range(1, len(original_hidden_states)):
        # only need to reconstruct states for last token position
        state = original_hidden_states[i][0, -1, :].clone().detach().to(DEVICE)
        reconstructed_states.append(state.requires_grad_(True))

    # setup the optimizer to update our list of trainable state tensors
    optimizer = torch.optim.Adam(reconstructed_states, lr=lr)
    loss_function = torch.nn.MSELoss()
    
    print(f"--- Reconstructing Internal State (Optimizing {len(reconstructed_states)} tensors) ---")
    
    for step in tqdm(range(epoch), desc="Optimization Progress"):
        optimizer.zero_grad()
        
        total_consistency_loss = 0.0
        
        # start with the fixed embedding of the second to last token
        current_hidden_state = original_hidden_states[0][0, -1, :].unsqueeze(0).unsqueeze(0).detach()
        
        # run the forward pass layer by layer
        for i, layer in enumerate(model.model.layers):
            # simplifying by using the previous layers output as input
            layer_output = layer(current_hidden_state)[0]
            
            # trainable guess for the input of layer `i+1`.
            consistency_loss = loss_function(layer_output.squeeze(), reconstructed_states[i])
            total_consistency_loss += consistency_loss
            
            # output of this layer becomes the input for the next
            current_hidden_state = layer_output

        # final hidden state after the last layer
        final_hidden_state = current_hidden_state
        
        final_hidden_state = model.model.norm(final_hidden_state)
        reconstructed_logits = model.lm_head(final_hidden_state).squeeze()
        
        # penalize the difference between our result and the malicious target
        target_loss = loss_function(reconstructed_logits, malicious_target_logits)
        
        total_loss = target_loss + total_consistency_loss
        
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}: Total Loss={total_loss.item():.4f}, Target Loss={target_loss.item():.4f}, Consistency Loss={total_consistency_loss.item():.4f}")

    return [state.detach() for state in reconstructed_states]

# %%
def analyze_reconstruction(original_hidden_states, reconstructed_states):
    analysis_results = []
    
    # Compare the last token's hidden state across all layers
    for i in range(len(reconstructed_states)):
        original = original_hidden_states[i+1][0, -1, :].to(DEVICE)
        reconstructed = reconstructed_states[i]
        
        # 1. Calculate overall MSE for the layer
        mse = F.mse_loss(original, reconstructed).item()
        
        # 2. Find the neuron with the minimum absolute difference
        abs_diff = torch.abs(original - reconstructed)
        min_diff_val, min_diff_idx = torch.min(abs_diff, dim=0)
        
        # 3. Store detailed results for this layer
        analysis_results.append({
            'layer_index': i,
            'mse_error': mse,
            'min_abs_difference': min_diff_val.item(),
            'min_diff_neuron_index': min_diff_idx.item()
        })
        
    print("\n--- Analysis of Reconstruction ---")
    mean_mse = np.mean([res['mse_error'] for res in analysis_results])
    print(f"Mean Squared Error across all layers: {mean_mse:.6f}")
    
    best_layer = min(analysis_results, key=lambda x: x['min_abs_difference'])
    print(f"Most plausible neuron found in Layer {best_layer['layer_index']}:")
    print(f"  - Neuron Index: {best_layer['min_diff_neuron_index']}")
    print(f"  - Absolute Difference: {best_layer['min_abs_difference']:.8f}")

    mse_values = [res['mse_error'] for res in analysis_results]
    plt.figure(figsize=(12, 6))
    plt.plot(mse_values, marker='o', linestyle='-')
    plt.title("Reconstruction Error (MSE) vs. Layer Depth")
    plt.xlabel("Decoder Layer Index")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    save_path = 'attack_success.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return analysis_results

# %%
PROMPT = "The capital of Turkey is"
PROMPT = "My favorite color is"
inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

#STEP 1
original_logits, original_hidden_states = get_original_activations_and_logits(model, inputs)
malicious_target = create_malicious_output(tokenizer, original_logits)

#STEP 2
reconstructed_hidden_states = reconstruct_internal_state(
    model,
    original_hidden_states,
    malicious_target,
    epoch=250,
    lr=0.05
)

#STEP 3
analysis_data = analyze_reconstruction(original_hidden_states, reconstructed_hidden_states)



# %%
#STEP 4: Save the analysis to a CSV file
filename=f"analysis_{PROMPT.replace(' ', '_')}.csv"
df = pd.DataFrame(analysis_data)
df.to_csv(filename, index=False)

# %%



