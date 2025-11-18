
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ===============================================================================
# Configuration
# ===============================================================================

OUTPUT_DIR = "."

print("=" * 80)
print("Visualizer for LLM-TS3 Analysis")
print("Generating Plots from Aggregated Statistics")
print("=" * 80)

# ===============================================================================
# Load Aggregated Data
# ===============================================================================

print("\n" + "=" * 80)
print("Loading Aggregated Statistics")
print("=" * 80)

# Check if aggregated files exist
required_files = [
    "aggregated_test_cases.csv",
    "aggregated_layer_stats.csv",
    "aggregated_component_stats.csv"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"\nERROR: Missing required files:")
    for f in missing_files:
        print(f"  • {f}")
    print("\nPlease run data_analyzer.py first to generate these files.")
    sys.exit(1)

# Load the data
print("\nLoading aggregated statistics...")
test_case_df = pd.read_csv("aggregated_test_cases.csv")
layer_stats_df = pd.read_csv("aggregated_layer_stats.csv")
component_stats_df = pd.read_csv("aggregated_component_stats.csv")

print(f"✓ Loaded test case statistics: {len(test_case_df):,} test cases")
print(f"✓ Loaded layer statistics: {len(layer_stats_df)} layers")
print(f"✓ Loaded component statistics: {len(component_stats_df)} components")

# ===============================================================================
# PLOT 1: Threshold Analysis - CDF Plot
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 1: Threshold Analysis")
print("=" * 80)

sorted_data = test_case_df['mean_cross_model_difference'].sort_values()

plt.figure(figsize=(10, 8))

y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

plt.step(sorted_data, y_values, where='post', color='steelblue', linewidth=2.5)
plt.fill_between(sorted_data, y_values, alpha=0.15, color='steelblue', step='post')

median_value = sorted_data.median()

key_percentiles = [0, 1, 10, 25, 50, 75, 90]
for pct in key_percentiles:
    percentile_value = sorted_data.quantile(pct/100)
    plt.axhline(y=pct, color='lightgray', linestyle=':', alpha=0.5)
    plt.plot(percentile_value, pct, marker='o', color='darkred', markersize=6)
    plt.text(percentile_value, pct + 2, f"{percentile_value:.3f}", 
            color='darkred', ha='center', va='bottom', fontweight='bold', fontsize=20)

plt.grid(True, linestyle='-', alpha=0.3)
plt.title(f'Threshold Analysis: Distribution of Random Activation Differences', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Cross-Model Activation Difference Value', fontsize=20)
plt.ylabel('Threshold Coverage (%)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(sorted_data.min() * 0.95, sorted_data.max() * 1.05)
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 20))

stats_text = (f"Min: {sorted_data.min():.3f}\n"
              f"Median: {median_value:.3f}\n" 
              f"Max: {sorted_data.max():.3f}\n"
              f"Std: {sorted_data.std():.3f}")

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=20,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'DCA-TS3.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# PLOT 2: Layer-by-Layer Bar Chart
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 2: Cross-Model Activation Differences Across Transformer Blocks")
print("=" * 80)

plt.figure(figsize=(10, 8))

layers = layer_stats_df['layer'].tolist()
avg_values = layer_stats_df['mean'].values

bars = plt.bar(layers, avg_values, color='steelblue', alpha=0.7, width=0.6)

# Highlight top 3 layers
top_3_indices = layer_stats_df.nlargest(3, 'mean').index
top_3_layers = layer_stats_df.loc[top_3_indices, 'layer'].tolist()

for layer_num in top_3_layers:
    if layer_num in layers:
        idx = layers.index(layer_num)
        bars[idx].set_color('darkred')
        bars[idx].set_alpha(0.9)
        value = avg_values[idx]
        plt.text(layer_num, value + max(avg_values) * 0.01, 
                f'{value:.3f}', 
                ha='center', va='bottom', 
                fontsize=20, fontweight='bold')

plt.title('Cross-Model Activation Differences Across Transformer Blocks', 
          fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Transformer Block Number', fontsize=20)
plt.ylabel('Cross-Model Activation Difference', fontsize=20)
plt.grid(True, alpha=0.3, axis='y')

plt.ylim(0, max(avg_values) * 1.15)

even_layers = [int(layer) for layer in layers if int(layer) % 2 == 0]
plt.xticks(even_layers, rotation=0, fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'CAD-TS3.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# PLOT 3: Cumulative Difference Plot
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 3: Cumulative Activation Differences")
print("=" * 80)

layer_means = layer_stats_df.set_index('layer')['mean'].sort_index()
cumulative_diff = layer_means.cumsum()
layers = layer_means.index.tolist()
layer_values = layer_means.values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

ax1.plot(layers, cumulative_diff.values, 'b-', linewidth=3, 
         label='Cumulative Difference', marker='o', markersize=4)
ax1.fill_between(layers, cumulative_diff.values, alpha=0.3, color='lightblue')

top_layers_series = layer_means.nlargest(3)
for layer_num in top_layers_series.index:
    ax1.axvline(x=layer_num, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(layer_num, cumulative_diff[layer_num] + cumulative_diff.max() * 0.02, 
            f'L{int(layer_num)}', ha='center', va='bottom', color='red', 
            fontweight='bold', fontsize=10)

early_phase_end = len(layers) // 3
late_phase_start = 2 * len(layers) // 3

if len(layers) > 6:
    ax1.annotate('Early Layers\n(Slow Growth)', 
                xy=(layers[early_phase_end], cumulative_diff.iloc[early_phase_end]), 
                xytext=(layers[early_phase_end] + 2, cumulative_diff.max() * 0.3),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    ax1.annotate('Late Layers\n(Rapid Accumulation)', 
                xy=(layers[late_phase_start], cumulative_diff.iloc[late_phase_start]), 
                xytext=(layers[late_phase_start] - 2, cumulative_diff.max() * 0.7),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

ax1.set_title('Cumulative Activation Differences Through Network Depth', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Cumulative Activation Difference', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)

colors = ['orange' if layer_num in top_layers_series.index else 'lightcoral' 
          for layer_num in layers]
bars = ax2.bar(layers, layer_values, alpha=0.7, color=colors, width=0.8)

for i, (layer_num, bar) in enumerate(zip(layers, bars)):
    if layer_num in top_layers_series.index:
        bar.set_color('darkred')
        bar.set_alpha(0.8)

ax2.set_xlabel('Transformer Layer (Block Number)', fontsize=12)
ax2.set_ylabel('Individual Layer\nDifference', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(ax1.get_xlim())

total_cumulative = cumulative_diff.iloc[-1]
early_contrib = cumulative_diff.iloc[early_phase_end] if len(layers) > 6 else 0
late_contrib = cumulative_diff.iloc[-1] - cumulative_diff.iloc[late_phase_start] if len(layers) > 6 else 0

stats_text = f'Total Cumulative: {total_cumulative:.6f}\n'
if len(layers) > 6:
    stats_text += f'Early Layers Contribution: {early_contrib/total_cumulative*100:.1f}%\n'
    stats_text += f'Late Layers Contribution: {late_contrib/total_cumulative*100:.1f}%'

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Cumulative-TS3.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# Final Summary
# ===============================================================================

print("\n" + "=" * 80)
print("🎉 ALL VISUALIZATIONS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • DCA-TS3.pdf - Threshold Analysis (CDF Plot)")
print("  • CAD-TS3.pdf - Cross-Model Activation Differences (Bar Chart)")
print("  • Cumulative-TS3.pdf - Cumulative Differences Through Network Depth")
print("=" * 80)

