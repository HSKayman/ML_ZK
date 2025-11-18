
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from typing import Dict, List, Tuple, Optional
import re
from tqdm import tqdm
import sys

# ===============================================================================
# Configuration
# ===============================================================================

INPUT_FILENAME = "attack_calc_error_analysis.csv"
OUTPUT_DIR = "."
CHUNK_SIZE = 2900580  

print("=" * 80)
print("Attack Data Analyzer for Input Reconstruction Attack")
print("Memory-Efficient Processing")
print("=" * 80)

# Check if input file exists
if not os.path.exists(INPUT_FILENAME):
    print(f"\nERROR: {INPUT_FILENAME} not found!")
    print("Please make sure attack_calc_error_analysis.csv is in the current directory.")
    sys.exit(1)

# Get file size
file_size_bytes = os.path.getsize(INPUT_FILENAME)
file_size_gb = file_size_bytes / (1024 ** 3)
print(f"\nInput file: {INPUT_FILENAME}")
print(f"File size: {file_size_gb:.2f} GB")
print(f"Chunk size: {CHUNK_SIZE:,} rows")

# ===============================================================================
# Helper Functions
# ===============================================================================

def extract_block_number(layer_name):
    """Extract transformer block number from layer name"""
    patterns = [r'layer_(\d+)_']
    for pattern in patterns:
        match = re.search(pattern, layer_name)
        if match:
            return int(match.group(1))
    return -1

def extract_component_type(layer_name):
    """Extract component type from layer name"""
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

# ===============================================================================
# Stream and Aggregate Data
# ===============================================================================

print("\n" + "=" * 80)
print("Reading and Processing Data in Chunks")
print("=" * 80)

# Initialize accumulators
reconstruction_accumulator = defaultdict(list)  # (input, reconstruction_idx) -> list of differences
layer_accumulator = defaultdict(lambda: {'count': 0, 'sum': 0, 'sum_sq': 0, 'min': float('inf'), 'max': float('-inf')})
component_accumulator = defaultdict(lambda: {'count': 0, 'sum': 0, 'sum_sq': 0})

total_rows_processed = 0
chunk_count = 0
skipped_chunks = 0

print(f"\nProcessing {INPUT_FILENAME} in chunks...")

try:
    with pd.read_csv(INPUT_FILENAME, chunksize=CHUNK_SIZE) as reader:
        for chunk in tqdm(reader, desc="Processing chunks"):
            chunk_count += 1
            
            # Filter to keep only original and reconstructed
            filtered = chunk[
                (chunk['run_type'] == 'original') | 
                (chunk['run_type'] == 'reconstructed')
            ].copy()
            
            if filtered.empty:
                skipped_chunks += 1
                continue
            
            # Pivot to pair up original and reconstructed
            try:
                paired = filtered.pivot_table(
                    index=['input', 'reconstruction_idx', 'round', 'layer_name'],
                    columns='run_type',
                    values=['error_real_value', 'error_calc_value']
                ).reset_index()
            except Exception as e:
                print(f"\n  Warning: Pivoting failed for chunk {chunk_count}: {e}")
                skipped_chunks += 1
                continue
            
            # Flatten column names
            paired.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in paired.columns.values]
            
            # Check if we have the required columns
            if 'error_calc_value_reconstructed' not in paired.columns or 'error_real_value_original' not in paired.columns:
                skipped_chunks += 1
                continue
            
            # Calculate the attack-induced difference (reconstructed calc vs original real)
            paired['abs_attack_difference'] = (
                paired['error_calc_value_reconstructed'] - paired['error_real_value_original']
            ).abs()
            
            # Add metadata
            paired['block_number'] = paired['layer_name'].apply(extract_block_number)
            paired['component_type'] = paired['layer_name'].apply(extract_component_type)
            
            total_rows_processed += len(paired)
            
            # Accumulate reconstruction statistics (input, reconstruction_idx)
            reconstruction_groups = paired.groupby(['input', 'reconstruction_idx'])['abs_attack_difference'].mean()
            for key, value in reconstruction_groups.items():
                reconstruction_accumulator[key].append(value)
            
            # Accumulate layer statistics (for valid blocks only)
            valid_layers = paired[paired['block_number'] >= 0]
            if not valid_layers.empty:
                layer_groups = valid_layers.groupby('block_number')['abs_attack_difference']
                
                for layer_num, group in layer_groups:
                    values = group.values
                    layer_accumulator[layer_num]['count'] += len(values)
                    layer_accumulator[layer_num]['sum'] += np.sum(values)
                    layer_accumulator[layer_num]['sum_sq'] += np.sum(values ** 2)
                    layer_accumulator[layer_num]['min'] = min(layer_accumulator[layer_num]['min'], np.min(values))
                    layer_accumulator[layer_num]['max'] = max(layer_accumulator[layer_num]['max'], np.max(values))
            
            # Accumulate component statistics
            if not valid_layers.empty:
                component_groups = valid_layers.groupby('component_type')['abs_attack_difference']
                
                for comp_type, group in component_groups:
                    values = group.values
                    component_accumulator[comp_type]['count'] += len(values)
                    component_accumulator[comp_type]['sum'] += np.sum(values)
                    component_accumulator[comp_type]['sum_sq'] += np.sum(values ** 2)

except KeyboardInterrupt:
    print("\n\nProcess interrupted by user!")
    sys.exit(1)
except Exception as e:
    print(f"\n\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n✓ Data processing complete!")
print(f"  Total chunks processed: {chunk_count}")
print(f"  Skipped chunks: {skipped_chunks}")
print(f"  Total rows processed: {total_rows_processed:,}")

# ===============================================================================
# Finalize Aggregated Statistics
# ===============================================================================

print("\n" + "=" * 80)
print("Finalizing Aggregated Statistics")
print("=" * 80)

# Calculate reconstruction averages
reconstruction_data = []
for (input_id, recon_idx), values in reconstruction_accumulator.items():
    reconstruction_data.append({
        'input': input_id,
        'reconstruction_idx': recon_idx,
        'mean_attack_difference': np.mean(values)
    })

reconstruction_df = pd.DataFrame(reconstruction_data)

# Calculate layer statistics
layer_data = []
for layer_num, stats in layer_accumulator.items():
    count = stats['count']
    if count > 0:
        mean = stats['sum'] / count
        # Calculate standard deviation: sqrt(E[X^2] - E[X]^2)
        variance = (stats['sum_sq'] / count) - (mean ** 2)
        std = np.sqrt(max(0, variance))  # max to handle floating point errors
        
        layer_data.append({
            'layer': layer_num,
            'count': count,
            'mean': mean,
            'std': std,
            'min': stats['min'],
            'max': stats['max']
        })

layer_stats_df = pd.DataFrame(layer_data).sort_values('layer')

# Calculate component statistics
component_data = []
for comp_type, stats in component_accumulator.items():
    count = stats['count']
    if count > 0:
        mean = stats['sum'] / count
        variance = (stats['sum_sq'] / count) - (mean ** 2)
        std = np.sqrt(max(0, variance))
        
        component_data.append({
            'component': comp_type,
            'count': count,
            'mean': mean,
            'std': std
        })

component_stats_df = pd.DataFrame(component_data)

print(f"\nAggregated statistics:")
print(f"  Reconstructions: {len(reconstruction_df):,}")
print(f"  Layers: {len(layer_stats_df)}")
print(f"  Components: {len(component_stats_df)}")

# Save aggregated data
reconstruction_df.to_csv(os.path.join(OUTPUT_DIR, "aggregated_attack_reconstructions.csv"), index=False)
layer_stats_df.to_csv(os.path.join(OUTPUT_DIR, "aggregated_attack_layer_stats.csv"), index=False)
component_stats_df.to_csv(os.path.join(OUTPUT_DIR, "aggregated_attack_component_stats.csv"), index=False)

print(f"\n✓ Saved aggregated statistics to CSV files")

# ===============================================================================
# Statistical Analysis
# ===============================================================================

print("\n" + "=" * 80)
print("🔍 INPUT RECONSTRUCTION ATTACK ANALYSIS")
print("=" * 80)

sorted_data = reconstruction_df['mean_attack_difference'].sort_values()

print(f"\n📊 Dataset Overview:")
print(f"   Reconstructions analyzed: {len(reconstruction_df):,}")
print(f"   Total rows processed: {total_rows_processed:,}")
print(f"   Unique inputs: {reconstruction_df['input'].nunique()}")
print(f"   Reconstructions per input: {reconstruction_df.groupby('input').size().mean():.1f} (avg)")

print(f"\n📈 Attack Effectiveness Summary:")
stats = {
    'Minimum': sorted_data.min(),
    'Maximum': sorted_data.max(),
    'Mean': sorted_data.mean(),
    'Median': sorted_data.median(),
    'Std Dev': sorted_data.std(),
    'Range': sorted_data.max() - sorted_data.min()
}

for name, value in stats.items():
    print(f"   {name:<10}: {value:.6f}")

print(f"\n📊 Percentile Distribution:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for pct in percentiles:
    value = sorted_data.quantile(pct/100)
    print(f"   {pct:2d}th percentile: {value:.6f}")

print(f"\n🎯 Attack Impact Thresholds:")
thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
for threshold in thresholds:
    pct_below = (sorted_data < threshold).sum() / len(sorted_data) * 100
    print(f"   < {threshold:5.3f}: {pct_below:5.1f}% of reconstructions")

# Extreme Cases
print(f"\n🔥 Highest Attack Impact (Top 5):")
top_5 = reconstruction_df.nlargest(5, 'mean_attack_difference')
for i, (_, row) in enumerate(top_5.iterrows(), 1):
    print(f"   {i}. {row['mean_attack_difference']:.6f} → Input {row['input']}, Recon #{row['reconstruction_idx']}")

print(f"\n❄️  Lowest Attack Impact (Top 5):")
bottom_5 = reconstruction_df.nsmallest(5, 'mean_attack_difference')
for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
    print(f"   {i}. {row['mean_attack_difference']:.6f} → Input {row['input']}, Recon #{row['reconstruction_idx']}")

# Layer-Level Analysis
if len(layer_stats_df) > 0:
    print(f"\n" + "=" * 60)
    print("🏗️  LAYER-BY-LAYER VULNERABILITY ANALYSIS")
    print("=" * 60)
    
    print(f"\n📋 Per-Layer Statistics (top 10 by mean):")
    print(f"{'Layer':<6} {'Count':<10} {'Mean':<12} {'Std':<12}")
    print("-" * 50)
    
    top_layers = layer_stats_df.nlargest(10, 'mean')
    for _, row in top_layers.iterrows():
        print(f"{int(row['layer']):<6} {int(row['count']):<10} {row['mean']:<12.6f} {row['std']:<12.6f}")
    
    # Component Analysis
    print(f"\n🧩 Component Type Vulnerability:")
    print(f"{'Component':<20} {'Count':<10} {'Mean':<12} {'Std':<12}")
    print("-" * 55)
    
    for _, row in component_stats_df.iterrows():
        print(f"{row['component']:<20} {int(row['count']):<10} {row['mean']:<12.6f} {row['std']:<12.6f}")

# Summary Insights
print(f"\n" + "=" * 60)
print("💡 KEY INSIGHTS")
print("=" * 60)

high_impact_pct = (sorted_data > 0.01).sum() / len(sorted_data) * 100
low_impact_pct = (sorted_data < 0.001).sum() / len(sorted_data) * 100

print(f"   • {high_impact_pct:.1f}% of reconstructions show high attack impact (>0.01)")
print(f"   • {low_impact_pct:.1f}% of reconstructions show low attack impact (<0.001)")
print(f"   • Average attack-induced difference: {sorted_data.mean():.6f}")

if len(layer_stats_df) > 0:
    most_vulnerable_layer = layer_stats_df.loc[layer_stats_df['mean'].idxmax()]
    least_vulnerable_layer = layer_stats_df.loc[layer_stats_df['mean'].idxmin()]
    
    print(f"   • Most vulnerable layer: {int(most_vulnerable_layer['layer'])} (avg: {most_vulnerable_layer['mean']:.6f})")
    print(f"   • Least vulnerable layer: {int(least_vulnerable_layer['layer'])} (avg: {least_vulnerable_layer['mean']:.6f})")

# Per-input analysis
print(f"\n📊 Per-Input Attack Success:")
input_stats = reconstruction_df.groupby('input')['mean_attack_difference'].agg(['mean', 'std', 'min', 'max', 'count'])
input_stats = input_stats.sort_values('mean', ascending=False)
print(f"\nTop 5 most affected inputs:")
for i, (input_id, row) in enumerate(input_stats.head(5).iterrows(), 1):
    print(f"   {i}. Input {input_id}: mean={row['mean']:.6f}, std={row['std']:.6f}, n={int(row['count'])}")

print("\n" + "=" * 80)
print("✅ ATTACK DATA ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • aggregated_attack_reconstructions.csv - Reconstruction statistics")
print("  • aggregated_attack_layer_stats.csv - Layer-level vulnerability")
print("  • aggregated_attack_component_stats.csv - Component-level vulnerability")
print("\nRun attack_visualizer.py to generate plots from these aggregated statistics.")
print("=" * 80)

