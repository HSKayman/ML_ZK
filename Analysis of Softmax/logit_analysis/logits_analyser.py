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
DEVICE

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

model.eval() # Set model to evaluation mode

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
prompt_to_analyze = "The capital of France is"

# Get the analysis
analysis = analyze_prompt_logits(prompt_to_analyze, model, tokenizer, top_k=5)

# Print the results
print_analysis(analysis)

# Clean up to free VRAM
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

# %%


# %%


# %%


# %%



