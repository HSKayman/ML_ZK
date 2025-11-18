import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# --- Configuration ---
INPUT_FILENAME = "formula_analysis.csv"
OUTPUT_FILENAME = "analysis_summary.csv"


CHUNK_SIZE = 2900580 
TOLERANCE = 1e-6       # Small value to handle floating-point inaccuracies
GROUP_COLS = ['input', 'token_pos', 'round']
LAYER_GROUP_COLS = ['input', 'token_pos', 'round', 'layer_name']

# Map for input_id
input_key_map = {}
input_counter = 0

all_chunk_results = []

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
                (chunk['run_type'] == 'other')
            ].copy() # .copy() prevents the SettingWithCopyWarning

            if chunk.empty:
                continue

            # 2. Pivot the chunk to "pair up" the rows
            try:
                paired_df = chunk.pivot_table(
                    index=LAYER_GROUP_COLS,
                    columns='run_type',
                    values=['error_real_value', 'error_calc_value']
                )
            except Exception as e:
                print(f"Warning: Pivoting failed for a chunk. Skipping. Error: {e}")
                continue

            # 3. Perform your calculation: (other_calc) - (original_real)
            paired_df.columns = ['_'.join(col).strip() for col in paired_df.columns.values]
            
            recon_calc_col = 'error_calc_value_other'
            original_real_col = 'error_real_value_original'
            
            if not (recon_calc_col in paired_df.columns and original_real_col in paired_df.columns):
                print("Warning: Chunk missing 'original' or 'other' data. Skipping.")
                continue

            paired_df['original_real'] = paired_df[original_real_col]
            paired_df['recon_calc'] = paired_df[recon_calc_col]
            
            # 4. Calculate error and check for NaNs
            paired_df['calc_error'] = (paired_df['recon_calc'] - paired_df['original_real']).abs()
            paired_df['has_nan'] = paired_df['original_real'].isna() | paired_df['recon_calc'].isna()
            
            # 5. Aggregate at the layer level
            layer_results = paired_df[['calc_error', 'has_nan']].reset_index()

            # 6. Aggregate at the group level (input, recon_idx, round)
            
            # 6a. Find if *any* layer had a NaN
            group_nan_status = layer_results.groupby(GROUP_COLS)['has_nan'].any()

            # 6b. Get Max Error (per group)
            valid_results = layer_results[layer_results['has_nan'] == False]
            group_max_indices = valid_results.groupby(GROUP_COLS)['calc_error'].idxmax()
            group_max_rows = valid_results.loc[group_max_indices].set_index(GROUP_COLS)
            group_max_rows = group_max_rows.rename(columns={'calc_error': 'max_error', 'layer_name': 'layer_at_max'})

            # 6c. Get Min Error (per group)
            group_min_indices = valid_results.groupby(GROUP_COLS)['calc_error'].idxmin()
            group_min_rows = valid_results.loc[group_min_indices].set_index(GROUP_COLS)
            group_min_rows = group_min_rows.rename(columns={'calc_error': 'min_error', 'layer_name': 'layer_at_min'})
            
            # 7. Merge aggregations
            combined_df = pd.merge(
                group_max_rows[['layer_at_max', 'max_error']],
                group_min_rows[['layer_at_min', 'min_error']],
                left_index=True,
                right_index=True,
                how='outer'
            )
            
            final_agg_df = pd.merge(
                combined_df,
                group_nan_status.rename('has_nan'),
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # If a group was 100% NaN, its 'has_nan' will be NaN. Fix it.
            final_agg_df['has_nan'] = final_agg_df['has_nan'].fillna(True)
            
            all_chunk_results.append(final_agg_df)
            
except Exception as e:
    print(f"\nAn error occurred while processing chunks: {e}")
    sys.exit()

if not all_chunk_results:
    print("Analysis complete, but no valid data was found.")
    sys.exit()

print("\n--- Processing Complete. Generating final summary CSV... ---")

# --- Post-Processing ---

# Combine all chunk results into one final DataFrame
results_df = pd.concat(all_chunk_results).reset_index()


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

# Your 'strongest_layer' logic:
results_df['strongest_layer'] = np.where(
    results_df['passed_or_not'] == 'FAILED', 
    results_df['layer_at_max'], # Show layer with max error
    'N/A'
)
# Vulnerable layer logic:
results_df['vulnerable_layer'] = np.where(
    results_df['passed_or_not'] == 'FAILED',
    results_df['layer_at_min'], # Show layer with min error
    'N/A'
)

# Select and rename final columns
results_df = results_df.rename(columns={
    'token_pos': 'token_position',
    'round': 'round_id',
    'input': 'input_id'   

})

final_output_df = results_df[[
    'input_id',
    'token_position',
    'round_id',
    'strongest_layer',    
    'max_error',
    'vulnerable_layer', 
    'min_error',
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
