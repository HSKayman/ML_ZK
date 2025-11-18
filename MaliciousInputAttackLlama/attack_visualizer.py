
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
print("Attack Visualizer for Input Reconstruction Attack")
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
    "aggregated_attack_reconstructions.csv",
    "aggregated_attack_layer_stats.csv",
    "aggregated_attack_component_stats.csv"
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"\nERROR: Missing required files:")
    for f in missing_files:
        print(f"  • {f}")
    print("\nPlease run attack_analyzer.py first to generate these files.")
    sys.exit(1)

# Load the data
print("\nLoading aggregated statistics...")
reconstruction_df = pd.read_csv("aggregated_attack_reconstructions.csv")
layer_stats_df = pd.read_csv("aggregated_attack_layer_stats.csv")
component_stats_df = pd.read_csv("aggregated_attack_component_stats.csv")

print(f"✓ Loaded reconstruction statistics: {len(reconstruction_df):,} reconstructions")
print(f"✓ Loaded layer statistics: {len(layer_stats_df)} layers")
print(f"✓ Loaded component statistics: {len(component_stats_df)} components")

# ===============================================================================
# PLOT 1: Attack Impact Distribution - CDF Plot
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 1: Attack Impact Distribution")
print("=" * 80)

sorted_data = reconstruction_df['mean_attack_difference'].sort_values()

plt.figure(figsize=(10, 8))

y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

plt.step(sorted_data, y_values, where='post', color='crimson', linewidth=2.5)
plt.fill_between(sorted_data, y_values, alpha=0.15, color='crimson', step='post')

median_value = sorted_data.median()

key_percentiles = [0, 1, 10, 25, 50, 75, 90]
for pct in key_percentiles:
    percentile_value = sorted_data.quantile(pct/100)
    plt.axhline(y=pct, color='lightgray', linestyle=':', alpha=0.5)
    plt.plot(percentile_value, pct, marker='o', color='darkred', markersize=6)
    plt.text(percentile_value, pct + 2, f"{percentile_value:.3f}", 
            color='darkred', ha='center', va='bottom', fontweight='bold', fontsize=20)

plt.grid(True, linestyle='-', alpha=0.3)
plt.title(f'Attack Impact Distribution: Input Reconstruction Analysis', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Attack-Induced Activation Difference', fontsize=20)
plt.ylabel('Cumulative Percentage (%)', fontsize=20)
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

save_path = os.path.join(OUTPUT_DIR, 'DCA-Attack.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# PLOT 2: Layer Vulnerability Bar Chart
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 2: Layer Vulnerability Across Transformer Blocks")
print("=" * 80)

plt.figure(figsize=(10, 8))

layers = layer_stats_df['layer'].tolist()
avg_values = layer_stats_df['mean'].values

bars = plt.bar(layers, avg_values, color='crimson', alpha=0.7, width=0.6)

# Highlight top 3 most vulnerable layers
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

plt.title('Layer Vulnerability to Input Reconstruction Attack', 
          fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Transformer Block Number', fontsize=20)
plt.ylabel('Attack-Induced Activation Difference', fontsize=20)
plt.grid(True, alpha=0.3, axis='y')

plt.ylim(0, max(avg_values) * 1.15)

even_layers = [int(layer) for layer in layers if int(layer) % 2 == 0]
plt.xticks(even_layers, rotation=0, fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'CAD-Attack.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# PLOT 3: Cumulative Attack Impact Plot
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 3: Cumulative Attack Impact Through Network")
print("=" * 80)

layer_means = layer_stats_df.set_index('layer')['mean'].sort_index()
cumulative_impact = layer_means.cumsum()
layers = layer_means.index.tolist()
layer_values = layer_means.values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

ax1.plot(layers, cumulative_impact.values, color='crimson', linewidth=3, 
         label='Cumulative Attack Impact', marker='o', markersize=4)
ax1.fill_between(layers, cumulative_impact.values, alpha=0.2, color='crimson')

top_layers_series = layer_means.nlargest(3)
for layer_num in top_layers_series.index:
    ax1.axvline(x=layer_num, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(layer_num, cumulative_impact[layer_num] + cumulative_impact.max() * 0.02, 
            f'L{int(layer_num)}', ha='center', va='bottom', color='red', 
            fontweight='bold', fontsize=10)

early_phase_end = len(layers) // 3
late_phase_start = 2 * len(layers) // 3

if len(layers) > 6:
    ax1.annotate('Early Layers\n(Lower Vulnerability)', 
                xy=(layers[early_phase_end], cumulative_impact.iloc[early_phase_end]), 
                xytext=(layers[early_phase_end] + 2, cumulative_impact.max() * 0.3),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    ax1.annotate('Late Layers\n(Higher Vulnerability)', 
                xy=(layers[late_phase_start], cumulative_impact.iloc[late_phase_start]), 
                xytext=(layers[late_phase_start] - 2, cumulative_impact.max() * 0.7),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

ax1.set_title('Cumulative Attack Impact Through Network Depth', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Cumulative Attack Impact', fontsize=12)
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
ax2.set_ylabel('Individual Layer\nVulnerability', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(ax1.get_xlim())

total_cumulative = cumulative_impact.iloc[-1]
early_contrib = cumulative_impact.iloc[early_phase_end] if len(layers) > 6 else 0
late_contrib = cumulative_impact.iloc[-1] - cumulative_impact.iloc[late_phase_start] if len(layers) > 6 else 0

stats_text = f'Total Cumulative: {total_cumulative:.6f}\n'
if len(layers) > 6:
    stats_text += f'Early Layers Contribution: {early_contrib/total_cumulative*100:.1f}%\n'
    stats_text += f'Late Layers Contribution: {late_contrib/total_cumulative*100:.1f}%'

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Cumulative-Attack.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# PLOT 4: Per-Input Attack Success Rate
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 4: Per-Input Attack Effectiveness")
print("=" * 80)

# Calculate per-input statistics
input_stats = reconstruction_df.groupby('input')['mean_attack_difference'].agg(['mean', 'std', 'count'])
input_stats = input_stats.sort_values('mean', ascending=False)

plt.figure(figsize=(12, 8))

x_pos = np.arange(len(input_stats))
means = input_stats['mean'].values
stds = input_stats['std'].values

bars = plt.bar(x_pos, means, yerr=stds, alpha=0.7, color='crimson', 
               capsize=5, error_kw={'linewidth': 2, 'alpha': 0.5})

# Highlight top 3 inputs
for i in range(min(3, len(bars))):
    bars[i].set_color('darkred')
    bars[i].set_alpha(0.9)
    plt.text(x_pos[i], means[i] + stds[i] + max(means) * 0.02, 
            f'{means[i]:.3f}', 
            ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

plt.title('Attack Effectiveness Per Input Prompt', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Input ID', fontsize=14)
plt.ylabel('Mean Attack-Induced Difference', fontsize=14)
plt.xticks(x_pos, input_stats.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'PerInput-Attack.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# PLOT 5: Component Vulnerability Comparison
# ===============================================================================

print("\n" + "=" * 80)
print("Creating Plot 5: Component Type Vulnerability")
print("=" * 80)

plt.figure(figsize=(10, 8))

components = component_stats_df['component'].tolist()
means = component_stats_df['mean'].values
stds = component_stats_df['std'].values

# Sort by mean for better visualization
sorted_indices = np.argsort(means)[::-1]
components_sorted = [components[i] for i in sorted_indices]
means_sorted = means[sorted_indices]
stds_sorted = stds[sorted_indices]

x_pos = np.arange(len(components_sorted))

bars = plt.bar(x_pos, means_sorted, yerr=stds_sorted, alpha=0.7, color='crimson',
               capsize=5, error_kw={'linewidth': 2, 'alpha': 0.5})

# Highlight most vulnerable component
bars[0].set_color('darkred')
bars[0].set_alpha(0.9)
plt.text(x_pos[0], means_sorted[0] + stds_sorted[0] + max(means_sorted) * 0.02, 
        f'{means_sorted[0]:.3f}', 
        ha='center', va='bottom', 
        fontsize=14, fontweight='bold')

plt.title('Component Type Vulnerability to Attack', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Component Type', fontsize=14)
plt.ylabel('Mean Attack-Induced Difference', fontsize=14)
plt.xticks(x_pos, components_sorted, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Component-Attack.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved plot to {save_path}")

# ===============================================================================
# Final Summary
# ===============================================================================

print("\n" + "=" * 80)
print("🎉 ALL ATTACK VISUALIZATIONS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  • DCA-Attack.pdf - Attack Impact Distribution (CDF Plot)")
print("  • CAD-Attack.pdf - Layer Vulnerability (Bar Chart)")
print("  • Cumulative-Attack.pdf - Cumulative Attack Impact Through Network")
print("  • PerInput-Attack.pdf - Per-Input Attack Effectiveness")
print("  • Component-Attack.pdf - Component Type Vulnerability")
print("=" * 80)

