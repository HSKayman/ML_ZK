import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# --- Configuration ---
INPUT_FILENAME = "attack_calc_error_analysis.csv"
OUTPUT_FILENAME = "hack_summary.csv"
MAP_FILENAME = "input_key_map.csv"

CHUNK_SIZE = 2900580 
TOLERANCE = 1e-6       # Small value to handle floating-point inaccuracies
GROUP_COLS = ['input', 'reconstruction_idx', 'round']
LAYER_GROUP_COLS = ['input', 'reconstruction_idx', 'round', 'layer_name']

# This dictionary will store the aggregated stats for each group
# Key: (input, recon_idx, round)
# Value: {'max_error': float, 'layer_at_max': str, 'has_nan': bool}
group_stats = {}

# Map for input_id
input_key_map = {}
input_counter = 0

print(f"--- Starting analysis of {INPUT_FILENAME} (One-Pass Method) ---")
if not os.path.exists(INPUT_FILENAME):
    print(f"Error: Could not find the file {INPUT_FILENAME}")
    sys.exit()

print(f"Processing in chunks of {CHUNK_SIZE:,} rows...")

try:
    with pd.read_csv(INPUT_FILENAME, chunksize=CHUNK_SIZE) as reader:
        for chunk in tqdm(reader, desc="Processing Chunks"):
            
            # 1. Filter to keep only the two run types we need
            chunk = chunk[
                (chunk['run_type'] == 'original') | 
                (chunk['run_type'] == 'reconstructed_with_orig_input')
            ].copy() # .copy() prevents the SettingWithCopyWarning

            if chunk.empty:
                continue

            # 2. Pivot the chunk to "pair up" the rows
            # This puts 'original' and 'reconstructed' data on the same line
            try:
                paired_df = chunk.pivot_table(
                    index=LAYER_GROUP_COLS,
                    columns='run_type',
                    values=['error_real_value', 'error_calc_value']
                )
            except Exception as e:
                print(f"Warning: Pivoting failed for a chunk, likely due to duplicates. Skipping. Error: {e}")
                continue # Skip this chunk if it's malformed

            # 3. Perform your calculation: (reconstructed_calc) - (original_real)
            # Flatten the multi-index columns for easier access
            paired_df.columns = ['_'.join(col).strip() for col in paired_df.columns.values]
            
            # Select the two columns you care about
            recon_calc_col = 'error_calc_value_reconstructed_with_orig_input'
            original_real_col = 'error_real_value_original'
            
            # Check if columns exist (in case a chunk is missing one type)
            if not (recon_calc_col in paired_df.columns and original_real_col in paired_df.columns):
                print("Warning: Chunk missing 'original' or 'reconstructed' data. Skipping.")
                continue

            paired_df['original_real'] = paired_df[original_real_col]
            paired_df['recon_calc'] = paired_df[recon_calc_col]
            
            # 4. Calculate error and check for NaNs
            paired_df['calc_error'] = (paired_df['recon_calc'] - paired_df['original_real']).abs()
            paired_df['has_nan'] = paired_df['original_real'].isna() | paired_df['recon_calc'].isna()
            
            # 5. Aggregate at the layer level
            # Reset index to get layer_name, etc. as columns
            layer_results = paired_df[['calc_error', 'has_nan']].reset_index()

            # 6. Aggregate at the group level (input, recon_idx, round)
            # Find the max error and if *any* layer had a NaN
            group_nan_status = layer_results.groupby(GROUP_COLS)['has_nan'].any()
            
            # Find index of max error
            layer_results.loc[layer_results['has_nan'], 'calc_error'] = -1 # Ignore errors from NaN rows
            group_max_indices = layer_results.groupby(GROUP_COLS)['calc_error'].idxmax()
            group_max_rows = layer_results.loc[group_max_indices]

            # 7. Update our global group_stats
            for _, row in group_max_rows.iterrows():
                group_key = (row['input'], row['reconstruction_idx'], row['round'])
                
                # Update input_id map
                if row['input'] not in input_key_map:
                    input_key_map[row['input']] = input_counter
                    input_counter += 1

                # We can just set it, since the chunk is already grouped
                group_stats[group_key] = {
                    'max_error': row['calc_error'],
                    'layer_at_max': row['layer_name'],
                    'has_nan': group_nan_status.get(group_key, True) # Default to True if key missing
                }
                
except Exception as e:
    print(f"\nAn error occurred while processing chunks: {e}")
    sys.exit()

if not group_stats:
    print("Analysis complete, but no valid data was found.")
    sys.exit()

print("\n--- Processing Complete. Generating final summary CSV... ---")

# --- Post-Processing ---
results_df = pd.DataFrame.from_dict(group_stats, orient='index')
results_df.index.names = GROUP_COLS
results_df = results_df.reset_index()

# Save the input map
map_df = pd.DataFrame(input_key_map.items(), columns=['input_string', 'input_id'])
map_df.to_csv(MAP_FILENAME, index=False)
print(f"Saved input string-to-ID map to {MAP_FILENAME}")

# Apply the input_id map
results_df['input_id'] = results_df['input'].map(input_key_map)

# Determine 'passed_or_not' using the 3-way logic
conditions = [
    (results_df['has_nan'] == True),
    (results_df['max_error'] > TOLERANCE),
]
choices = [
    'INCOMPLETE_TEST',  # If has_nan is True
    'FAILED',           # If max_error is > tolerance
]
results_df['passed_or_not'] = np.select(conditions, choices, default='PASSED')

# Finalize 'vulnerable_layer' column
results_df['vulnerable_layer'] = np.where(
    results_df['passed_or_not'] == 'FAILED', 
    results_df['layer_at_max'], # Show layer only if it FAILED
    'N/A'                     # N/A for PASSED or INCOMPLETE
)

# Select and rename final columns
results_df = results_df.rename(columns={
    'reconstruction_idx': 'construction_id',
    'round': 'round_id'
})
final_output_df = results_df[[
    'input_id',
    'construction_id',
    'round_id',
    'vulnerable_layer',
    'max_error',
    'passed_or_not'
]]

# Save the final CSV
final_output_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"Successfully generated summary file: {OUTPUT_FILENAME}")
print("\n--- Example output (first 5 rows): ---")
print(final_output_df.head().to_string())

# Print a summary of the results
print("\n--- Final Summary ---")
print(final_output_df['passed_or_not'].value_counts(dropna=False))