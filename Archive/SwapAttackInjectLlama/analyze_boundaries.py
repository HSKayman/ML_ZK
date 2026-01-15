#!/usr/bin/env python3
"""
Analyze Neuron Perturbation Boundaries
For each K, shows the average boundary delta needed to exceed the distance threshold.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import sys

# Configuration
INPUT_FILE = "neuron_perturbation_analysis_other.csv"
CHUNK_SIZE = 500000

print("=" * 70)
print("Neuron Perturbation Boundary Analysis")
print("=" * 70)

# Accumulators for each K
k_boundary_data = defaultdict(list)
k_stats = {}

print(f"\nReading {INPUT_FILE} in chunks...")

total_rows = 0
try:
    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        
        # Get unique (K, trial) boundary deltas
        # Each (K, trial) has multiple search steps, but one boundary_delta
        unique_boundaries = chunk.groupby(['num_neurons_changed', 'trial', 'input_id']).agg({
            'boundary_delta': 'first',
            'soundness_ratio': 'first'
        }).reset_index()
        
        for _, row in unique_boundaries.iterrows():
            k = row['num_neurons_changed']
            k_boundary_data[k].append(row['boundary_delta'])
        
        print(f"  Processed {total_rows:,} rows...", end='\r')

except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)

print(f"\n\nTotal rows processed: {total_rows:,}")
print(f"Unique K values: {len(k_boundary_data)}")

# Calculate statistics for each K
print("\n" + "=" * 70)
print("BOUNDARY DELTA STATISTICS BY K")
print("=" * 70)
print(f"\n{'K':>6} | {'Trials':>8} | {'Mean Δ':>12} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'Ratio':>10}")
print("-" * 85)

results = []
for k in sorted(k_boundary_data.keys()):
    deltas = k_boundary_data[k]
    n = len(deltas)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    min_delta = np.min(deltas)
    max_delta = np.max(deltas)
    soundness_ratio = k / 4096
    
    results.append({
        'K': k,
        'num_trials': n,
        'mean_boundary_delta': mean_delta,
        'std_boundary_delta': std_delta,
        'min_boundary_delta': min_delta,
        'max_boundary_delta': max_delta,
        'soundness_ratio': soundness_ratio
    })
    
    print(f"{k:>6} | {n:>8} | {mean_delta:>12.4f} | {std_delta:>10.4f} | {min_delta:>10.4f} | {max_delta:>10.4f} | {soundness_ratio:>10.6f}")

# Save to CSV
results_df = pd.DataFrame(results)
output_file = "boundary_analysis_summary.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ Summary saved to {output_file}")

# Key insights
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

if results:
    # Sort by mean boundary delta
    sorted_results = sorted(results, key=lambda x: x['mean_boundary_delta'])
    
    print(f"\n🔹 Easiest to break (lowest delta needed):")
    for r in sorted_results[-5:]:
        print(f"   K={r['K']:>4}: mean_delta={r['mean_boundary_delta']:.4f}, soundness={r['soundness_ratio']:.6f}")
    
    print(f"\n🔹 Hardest to break (highest delta needed):")
    for r in sorted_results[:5]:
        print(f"   K={r['K']:>4}: mean_delta={r['mean_boundary_delta']:.4f}, soundness={r['soundness_ratio']:.6f}")
    
    # Find K where mean delta is reasonable (e.g., < 50)
    reasonable_k = [r for r in results if r['mean_boundary_delta'] < 50]
    if reasonable_k:
        min_k = min(reasonable_k, key=lambda x: x['K'])
        print(f"\n🔹 Smallest K with boundary_delta < 50: K={min_k['K']}")
        print(f"   This gives soundness ratio = {min_k['soundness_ratio']:.6f} ({min_k['K']}/4096)")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
