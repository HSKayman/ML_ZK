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
print(f"Using device: {DEVICE}")

# %%
MODEL_1_PATH = "meta-llama/Llama-2-7b-chat-hf" 

# %%
tokenizer = LlamaTokenizer.from_pretrained(MODEL_1_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    MODEL_1_PATH,
    torch_dtype=torch.float16,   
    device_map="auto"           
)

model.eval()

# %%
def analyze_prompt_logits(prompt: str, 
                          model: LlamaForCausalLM, 
                          tokenizer: LlamaTokenizer, 
                          top_k: int = 5):
    
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids[0] # Get the 1D tensor of token IDs
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0] 
        
    probabilities = F.softmax(logits, dim=-1)
    
    analysis_results = []

    # Iterate through the sequence, token by token
    for i in range(len(input_ids) - 1):
        current_token_id = input_ids[i].item()
        current_token_str = tokenizer.decode(current_token_id)
        
        actual_next_token_id = input_ids[i+1].item()
        actual_next_token_str = tokenizer.decode(actual_next_token_id)

        logits_for_next = logits[i]
        probs_for_next = probabilities[i]

        actual_token_logit = logits_for_next[actual_next_token_id].item()
        actual_token_prob = probs_for_next[actual_next_token_id].item()

        top_k_probs, top_k_ids = torch.topk(probs_for_next, top_k)
        
        top_k_predictions_data = []
        for j in range(top_k):
            pred_id = top_k_ids[j].item()
            pred_prob = top_k_probs[j].item()
            pred_logit = logits_for_next[pred_id].item() # Get corresponding logit
            pred_str = tokenizer.decode(pred_id)
            
            top_k_predictions_data.append({
                "word": pred_str,
                "token_id": pred_id,
                "probability": pred_prob,
                "logit": pred_logit
            })

        analysis_results.append({
            "step": i,
            "current_token": f"{current_token_str} (ID: {current_token_id})",
            "actual_next_token": f"{actual_next_token_str} (ID: {actual_next_token_id})",
            "actual_token_logit": actual_token_logit,
            "actual_token_prob": actual_token_prob,
            "top_k_predictions": top_k_predictions_data
        })

    last_logits = logits[-1]
    last_probs = probabilities[-1]
    
    top_k_probs, top_k_ids = torch.topk(last_probs, top_k)
    top_k_predictions_data = []
    
    for j in range(top_k):
        pred_id = top_k_ids[j].item()
        pred_prob = top_k_probs[j].item()
        pred_logit = last_logits[pred_id].item()
        pred_str = tokenizer.decode(pred_id)
        
        top_k_predictions_data.append({
            "word": pred_str,
            "token_id": pred_id,
            "probability": pred_prob,
            "logit": pred_logit
        })

    analysis_results.append({
        "step": len(input_ids) - 1,
        "current_token": f"{tokenizer.decode(input_ids[-1].item())} (ID: {input_ids[-1].item()})",
        "actual_next_token": "N/A (End of Prompt)",
        "actual_token_logit": "N/A",
        "actual_token_prob": "N/A",
        "top_k_predictions": top_k_predictions_data
    })
    
    return analysis_results

# %%
def print_analysis(analysis_results):
    for result in analysis_results:
        print(f"\n==========================================")
        print(f" Step {result['step']} | Input Token: {result['current_token']}")
        print(f"==========================================")
        
        if result['actual_next_token'] != "N/A (End of Prompt)":
            print(f"Actual Next Token: {result['actual_next_token']}")
            print(f"  > Model's Logit for this token: {result['actual_token_logit']:.4f}")
            print(f"  > Model's Prob for this token:  {result['actual_token_prob']:.4f}")
        else:
            print("--- End of Prompt ---")

        print("\nModel's Top-5 Predictions for *this* position:")
        print("----------------------------------------------")
        print(f"{'Rank':<5} | {'Word':<15} | {'Token ID':<8} | {'Logit':<10} | {'Probability':<10}")
        print(f"-------------------------------------------------------------------")
        
        for rank, pred in enumerate(result['top_k_predictions'], 1):
            word_str = f"'{pred['word']}'"
            print(f"{rank:<5} | {word_str:<15} | {pred['token_id']:<8} | {pred['logit']:<10.4f} | {pred['probability']:<10.4f}")

# %%
import sys
def save_analysis_to_file(analysis_results, output_filename: str):
    print(f"\n--- Redirecting detailed analysis to '{output_filename}' ---")
    
    # Save the current standard output (the console)
    original_stdout = sys.stdout 
    
    try:
        # Redirect standard output to the file
        with open(output_filename, 'w') as f:
            sys.stdout = f
            print_analysis(analysis_results)
    finally:
        # **Crucially, restore the original standard output here.**
        # This block runs after the 'try' block, guaranteeing restoration.
        sys.stdout = original_stdout

    # Now that stdout is back to the console, this will print correctly.
    print(f"✅ Successfully saved detailed analysis to '{output_filename}'")

# %%
def save_all_logits_for_last_token(prompt: str, 
                                   model: LlamaForCausalLM, 
                                   tokenizer: LlamaTokenizer, 
                                   output_filename: str = "last_token_logits.json"):
    print(f"Analyzing prompt: '{prompt}'")
    
    #Get model outputs
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # for the very last token prediction
    
    last_logits = outputs.logits[0, -1, :]
    
    # probabilities for the entire vocabulary
    last_probs = F.softmax(last_logits, dim=-1)
    
    # 4. Sort 
    sorted_probs, sorted_indices = torch.sort(last_probs, descending=True)
    
    # 5. all token data
    all_logits_data = []
    for i in range(len(sorted_indices)):
        token_id = sorted_indices[i].item()
        prob = sorted_probs[i].item()
        logit = last_logits[token_id].item() 
        token_str = tokenizer.decode(token_id)
        
        all_logits_data.append({
            'rank': i + 1,
            'token': token_str,
            'token_id': token_id,
            'probability': prob,
            'logit': logit
        })
        
    with open(output_filename, 'w') as f:
        json.dump(all_logits_data, f, indent=4)
        
    print(f"✅ Successfully saved all {len(all_logits_data)} logits to '{output_filename}'")

# %%
sample_texts = [
        "The capital of France is",
        "The largest mammal on Earth is the",
        "The process of photosynthesis occurs in the"
    ]


for i in range(len(sample_texts)):
    print(f"\n--- Sample Prompt {i+1} ---")
    print(sample_texts[i])
    target_prompt = sample_texts[i]
    detailed_analysis_file = sample_texts[i].replace(" ", "_").lower() + "_detailed_analysis.txt"
    all_logits_file = sample_texts[i].replace(" ", "_").lower() + "_all_logits.json"


    analysis = analyze_prompt_logits(target_prompt, model, tokenizer, top_k=50)
    save_analysis_to_file(analysis, detailed_analysis_file)
    save_all_logits_for_last_token(target_prompt, model, tokenizer, output_filename=all_logits_file)

    # --- Optional: Print a brief summary to the console ---
    print("\n--- Console Summary---")
    print_analysis([analysis[-1]])


# Clean up to free VRAM
print("\nCleaning up model and tokenizer from memory.")
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()


