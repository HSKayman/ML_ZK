# %%
import pandas as pd
import numpy as np
import os

# %%
# =============================================================================
# Extract and Reformat Attack Data - Columns by Input ID
# =============================================================================

def extract_by_input_columns(
    input_csv: str = "gradient_swap_attack_results_other.csv",
    output_csv: str = "extracted_by_input.csv",
    top_k: int = 20
):
    df = pd.read_csv(input_csv)
    
    # Create output structure - rows are ranks
    output_data = {'rank': list(range(1, top_k + 1))}
    
    for _, row in df.iterrows():
        input_id = int(row['input_id'])
        prefix = f'input_{input_id}'
        
        # Original Z values
        z_orig_vals = []
        z_orig_indices = []
        for i in range(1, top_k + 1):
            idx_col = f'orig_z_act_rank{i}_idx'
            val_col = f'orig_z_act_rank{i}_val'
            if idx_col in row and pd.notna(row[idx_col]):
                z_orig_indices.append(int(row[idx_col]))
                z_orig_vals.append(row[val_col])
            else:
                z_orig_indices.append(None)
                z_orig_vals.append(None)
        
        output_data[f'{prefix}_z_idx'] = z_orig_indices
        output_data[f'{prefix}_z'] = z_orig_vals
        
        # Perturbed Z values
        z_pert_vals = []
        z_pert_indices = []
        for i in range(1, top_k + 1):
            idx_col = f'final_z_act_rank{i}_idx'
            val_col = f'final_z_act_rank{i}_val'
            if idx_col in row and pd.notna(row[idx_col]):
                z_pert_indices.append(int(row[idx_col]))
                z_pert_vals.append(row[val_col])
            else:
                z_pert_indices.append(None)
                z_pert_vals.append(None)
        
        output_data[f'{prefix}_perturbed_z_idx'] = z_pert_indices
        output_data[f'{prefix}_perturbed_z'] = z_pert_vals
        
        # Original softmax probabilities
        prob_orig_vals = []
        prob_orig_indices = []
        for i in range(1, top_k + 1):
            idx_col = f'orig_prob_rank{i}_idx'
            val_col = f'orig_prob_rank{i}'
            if idx_col in row and pd.notna(row[idx_col]):
                prob_orig_indices.append(int(row[idx_col]))
                prob_orig_vals.append(row[val_col])
            else:
                prob_orig_indices.append(None)
                prob_orig_vals.append(None)
        
        output_data[f'{prefix}_orig_prob_idx'] = prob_orig_indices
        output_data[f'{prefix}_orig_prob'] = prob_orig_vals
        
        # Perturbed softmax probabilities
        prob_pert_vals = []
        prob_pert_indices = []
        for i in range(1, top_k + 1):
            idx_col = f'final_prob_rank{i}_idx'
            val_col = f'final_prob_rank{i}'
            if idx_col in row and pd.notna(row[idx_col]):
                prob_pert_indices.append(int(row[idx_col]))
                prob_pert_vals.append(row[val_col])
            else:
                prob_pert_indices.append(None)
                prob_pert_vals.append(None)
        
        output_data[f'{prefix}_perturbed_prob_idx'] = prob_pert_indices
        output_data[f'{prefix}_perturbed_prob'] = prob_pert_vals
    
    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    print(f"Shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)[:10]}...")
    
    return output_df


def extract_z_only(
    input_csv: str = "gradient_swap_attack_results_other.csv",
    output_csv: str = "z_values_by_input.csv",
    top_k: int = 20
):

    df = pd.read_csv(input_csv)
    output_data = {'rank': list(range(1, top_k + 1))}
    
    for _, row in df.iterrows():
        input_id = int(row['input_id'])
        prefix = f'input_{input_id}'
        
        # Original Z
        z_orig = []
        for i in range(1, top_k + 1):
            val_col = f'orig_z_act_rank{i}_val'
            z_orig.append(row[val_col] if val_col in row else None)
        output_data[f'{prefix}_z'] = z_orig
        
        # Perturbed Z
        z_pert = []
        for i in range(1, top_k + 1):
            val_col = f'final_z_act_rank{i}_val'
            z_pert.append(row[val_col] if val_col in row else None)
        output_data[f'{prefix}_perturbed_z'] = z_pert
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return output_df


def extract_softmax_only(
    input_csv: str = "gradient_swap_attack_results_other.csv",
    output_csv: str = "softmax_values_by_input.csv",
    top_k: int = 20
):

    
    df = pd.read_csv(input_csv)
    output_data = {'rank': list(range(1, top_k + 1))}
    
    for _, row in df.iterrows():
        input_id = int(row['input_id'])
        prefix = f'input_{input_id}'
        
        # Original prob
        prob_orig = []
        for i in range(1, top_k + 1):
            val_col = f'orig_prob_rank{i}'
            prob_orig.append(row[val_col] if val_col in row else None)
        output_data[f'{prefix}_orig_prob'] = prob_orig
        
        # Perturbed prob
        prob_pert = []
        for i in range(1, top_k + 1):
            val_col = f'final_prob_rank{i}'
            prob_pert.append(row[val_col] if val_col in row else None)
        output_data[f'{prefix}_perturbed_prob'] = prob_pert
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return output_df


def extract_with_indices(
    input_csv: str = "gradient_swap_attack_results_other.csv",
    output_z_csv: str = "z_with_indices.csv",
    output_prob_csv: str = "prob_with_indices.csv",
    top_k: int = 20
):
    
    df = pd.read_csv(input_csv)
    
    # Z values with indices
    z_data = {'rank': list(range(1, top_k + 1))}
    prob_data = {'rank': list(range(1, top_k + 1))}
    
    for _, row in df.iterrows():
        input_id = int(row['input_id'])
        prefix = f'input_{input_id}'
        
        # Original Z with index
        z_orig_str = []
        for i in range(1, top_k + 1):
            idx_col = f'orig_z_act_rank{i}_idx'
            val_col = f'orig_z_act_rank{i}_val'
            if idx_col in row and pd.notna(row[idx_col]):
                z_orig_str.append(f"{int(row[idx_col])}:{row[val_col]:.4f}")
            else:
                z_orig_str.append(None)
        z_data[f'{prefix}_z'] = z_orig_str
        
        # Perturbed Z with index
        z_pert_str = []
        for i in range(1, top_k + 1):
            idx_col = f'final_z_act_rank{i}_idx'
            val_col = f'final_z_act_rank{i}_val'
            if idx_col in row and pd.notna(row[idx_col]):
                z_pert_str.append(f"{int(row[idx_col])}:{row[val_col]:.4f}")
            else:
                z_pert_str.append(None)
        z_data[f'{prefix}_perturbed_z'] = z_pert_str
        
        # Original prob with token index
        prob_orig_str = []
        for i in range(1, top_k + 1):
            idx_col = f'orig_prob_rank{i}_idx'
            val_col = f'orig_prob_rank{i}'
            if idx_col in row and pd.notna(row[idx_col]):
                prob_orig_str.append(f"{int(row[idx_col])}:{row[val_col]:.6f}")
            else:
                prob_orig_str.append(None)
        prob_data[f'{prefix}_orig_prob'] = prob_orig_str
        
        # Perturbed prob with token index
        prob_pert_str = []
        for i in range(1, top_k + 1):
            idx_col = f'final_prob_rank{i}_idx'
            val_col = f'final_prob_rank{i}'
            if idx_col in row and pd.notna(row[idx_col]):
                prob_pert_str.append(f"{int(row[idx_col])}:{row[val_col]:.6f}")
            else:
                prob_pert_str.append(None)
        prob_data[f'{prefix}_perturbed_prob'] = prob_pert_str
    
    z_df = pd.DataFrame(z_data)
    z_df.to_csv(output_z_csv, index=False)
    print(f"Saved: {output_z_csv}")
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(output_prob_csv, index=False)
    print(f"Saved: {output_prob_csv}")
    
    return z_df, prob_df


# %%
# =============================================================================
# Run Extraction
# =============================================================================

if __name__ == "__main__":
    INPUT_FILE = "gradient_swap_attack_results_other.csv"
    
    print("=" * 60)
    print("Extracting attack data with input_X format...")
    print("=" * 60)
    
    # Full extraction with all values
    print("\n1. Full extraction (z and softmax with indices)...")
    extract_by_input_columns(INPUT_FILE, "extracted_full.csv", top_k=20)
    
    # Z values only
    print("\n2. Z values only...")
    extract_z_only(INPUT_FILE, "z_by_input.csv", top_k=20)
    
    # Softmax values only
    print("\n3. Softmax values only...")
    extract_softmax_only(INPUT_FILE, "softmax_by_input.csv", top_k=20)
    
    # With indices as formatted strings
    print("\n4. Values with indices (idx:value format)...")
    extract_with_indices(INPUT_FILE, "z_with_idx.csv", "prob_with_idx.csv", top_k=20)
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
