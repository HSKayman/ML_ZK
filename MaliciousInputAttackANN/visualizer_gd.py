# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('activation_diff_results_formula.csv')

# %%
# Set global font size
plt.rcParams.update({'font.size': 20})

# Calculate reasonable thresholds based on data distribution
layers = ['fc1', 'fc2', 'fc3']
layer_stats = {}
for layer in layers:
    column = f'{layer}_mean_abs_diff'  # Using mean (median-like central tendency)
    values = df[column].values
    layer_stats[layer] = {
        'min': np.min(values),
        'p10': np.percentile(values, 10),
        'p25': np.percentile(values, 25),
        'p50': np.percentile(values, 50),
        'p75': np.percentile(values, 75),
        'p90': np.percentile(values, 90),
        'p95': np.percentile(values, 95),
        'p99': np.percentile(values, 99),
        'max': np.max(values)
    }

# Find thresholds that provide good separation between layers
# Use percentiles that show clear differences between layers
all_percentiles = []
for layer in layers:
    all_percentiles.extend([
        layer_stats[layer]['p10'],
        layer_stats[layer]['p25'],
        layer_stats[layer]['p50'],
        layer_stats[layer]['p75'],
        layer_stats[layer]['p90'],
        layer_stats[layer]['p95']
    ])

# Select thresholds that span the range and provide good separation
min_val = min([layer_stats[l]['min'] for l in layers])
max_val = max([layer_stats[l]['max'] for l in layers])

# Calculate reasonable key thresholds based on data distribution
# Use percentiles that show separation: typically 25th, 50th, 75th, 90th, 95th
key_thresholds_candidates = []
for p in [25, 50, 75, 90, 95]:
    thresholds_at_p = [layer_stats[l][f'p{p}'] for l in layers]
    # Use the median of the three layers at this percentile
    key_thresholds_candidates.append(np.median(thresholds_at_p))

# Also include some fixed small values for reference
key_thresholds_candidates.extend([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

# Sort and remove duplicates, keep reasonable range
key_thresholds_candidates = sorted(set(key_thresholds_candidates))
key_thresholds = [t for t in key_thresholds_candidates if min_val <= t <= max_val]

# If we don't have enough thresholds, add some log-spaced ones
if len(key_thresholds) < 4:
    log_min = np.log10(max(min_val, 1e-6))
    log_max = np.log10(min(max_val, 1))
    key_thresholds = np.logspace(log_min, log_max, 4).tolist()

# Use the calculated key thresholds for reference lines
calculated_reference_threshold = key_thresholds[2] if len(key_thresholds) > 2 else 0.0001

# Print calculated thresholds for reference
print("\nCalculated thresholds based on data distribution:")
print("=" * 70)
print(f"Reference threshold (vertical line): {calculated_reference_threshold:.6e}")
print(f"Key thresholds for statistics: {[f'{t:.6e}' for t in key_thresholds[:5]]}")
print("\nLayer value ranges (for visibility check):")
for layer in layers:
    col = f'{layer}_mean_abs_diff'  # Using mean (median-like central tendency)
    vals = df[col].values
    print(f"  {layer.upper()}: min={np.min(vals):.6e}, median={np.median(vals):.6e}, max={np.max(vals):.6e}")
print("=" * 70)

# Define threshold values to test (for smooth curves)
thresholds = np.logspace(-6, 0, 100)  # From 1e-6 to 1 (0.000001 to 1)

# Create single plot figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

layer_labels = {
    'fc1': r'$L_1$',
    'fc2': r'$L_2$', 
    'fc3': r'$L_3$'
}


# Define colors for each layer (more distinct and vibrant, darker blue for FC1)
layer_colors = {
    'fc1': '#0066CC',  # Darker, more vibrant blue for better visibility
    'fc2': '#ff7f0e',  # Orange
    'fc3': '#2ca02c'   # Green
}

# Define line styles for better distinction
layer_linestyles = {
    'fc1': '-',   # Solid
    'fc2': '--',  # Dashed
    'fc3': '-.'   # Dash-dot
}

# Define layer alphas for visual distinction (higher opacity for better visibility)
layer_alphas = {'fc1': 1.0, 'fc2': 0.9, 'fc3': 0.9}  # Full opacity for FC1

# Define line widths (thicker for FC1 to make it more visible)
layer_linewidths = {'fc1': 4.0, 'fc2': 3.5, 'fc3': 3.5}

# Plot median/mean values for each layer
# Plot FC1 last so it appears on top (since it has lower values, it might be obscured)
plot_order = ['fc2', 'fc3', 'fc1']  # Plot FC1 last to ensure visibility
for layer in plot_order:
    column = f'{layer}_mean_abs_diff'  # Using mean (median-like central tendency)
    values = df[column].values
    
    # Calculate percentage passing each threshold
    percentages = []
    for threshold in thresholds:
        passing = np.sum(values <= threshold) / len(values) * 100
        percentages.append(passing)
    
    # Diagnostic: Print FC1 percentages at key thresholds
    if layer == 'fc1':
        test_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
        print(f"\nFC1 visibility check - Percentage passing at key thresholds:")
        for t in test_thresholds:
            pct = np.sum(values <= t) / len(values) * 100
            print(f"  Threshold {t:.3f}: {pct:.1f}%")
    
    # Plot cumulative distribution with distinct styles
    ax.semilogx(thresholds, percentages, 
               label=layer_labels[layer],
               color=layer_colors[layer],
               linestyle=layer_linestyles[layer],
               linewidth=layer_linewidths[layer], 
               alpha=layer_alphas[layer],
               marker='',  # No markers for cleaner look
               markersize=0,
               zorder=10 if layer == 'fc1' else 5)  # Higher z-order for FC1 to ensure it's on top

# Plot the all-layers line (worst case - ALL layers must pass)
percentages_all = []
for threshold in thresholds:
    passing_count = 0
    for idx in df.index:
        row = df.loc[idx]
        # Check if all layers pass for this sample
        all_pass = True
        for layer in layers:
            if row[f'{layer}_mean_abs_diff'] > threshold:  # Using mean (median-like central tendency)
                all_pass = False
                break
        if all_pass:
            passing_count += 1
    
    percentage = (passing_count / len(df)) * 100
    percentages_all.append(percentage)

ax.semilogx(thresholds, percentages_all, 
           label='All Layers',
           color='red',  # Changed to red for better distinction
           linestyle=':',  # Dotted line style to distinguish from FC1
           linewidth=4.5,  # Slightly thicker
           alpha=1.0,
           zorder=15)  # Highest z-order to ensure it's visible

# Add reference lines using calculated thresholds
ax.axvline(x=calculated_reference_threshold, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=2)
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=2)
ax.axhline(y=99, color='gray', linestyle=':', alpha=0.5, linewidth=2)

# Add text labels
ref_label = f'{calculated_reference_threshold:.2e}' if calculated_reference_threshold < 0.01 else f'{calculated_reference_threshold:.4f}'
ax.text(calculated_reference_threshold * 0.7, 50, f'{ref_label} threshold', rotation=90, fontsize=20, alpha=0.7)
ax.text(1e-6, 10, '10%', fontsize=20, alpha=0.7)
ax.text(1e-6, 50, '50%', fontsize=20, alpha=0.7)
ax.text(1e-6, 100, '99%', fontsize=20, alpha=0.7)

ax.set_xlabel('Cross-Model Activation Difference Value', fontsize=20)
ax.set_ylabel('Threshold Coverage (%)', fontsize=20)
ax.set_title('Threshold Analysis: Distribution of Mean Activation Differences', fontsize=20, 
             fontweight='bold')

# Create legend
ax.legend(fontsize=20, 
          loc='lower right',
          framealpha=0.9)

ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
ax.set_xlim(1e-6, 1)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig('DCA-SQ1.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics at key thresholds
print("\nPercentage passing at key thresholds:")
print("=" * 70)
# Use calculated key thresholds (limit to 4-5 most relevant ones)
if len(key_thresholds) > 5:
    key_thresholds = key_thresholds[:5]

for threshold in key_thresholds:
    print(f"\nThreshold = {threshold}:")
    print("-" * 50)
    
    for layer in layers:
        column = f'{layer}_mean_abs_diff'  # Using mean (median-like central tendency)
        values = df[column].values
        percentage = np.sum(values <= threshold) / len(values) * 100
        print(f"  {layer.upper()}: {percentage:.1f}%")
    
    # All layers simultaneous
    passing_count = 0
    for idx in df.index:
        row = df.loc[idx]
        all_pass = all(row[f'{layer}_mean_abs_diff'] <= threshold for layer in layers)  # Using mean
        if all_pass:
            passing_count += 1
    percentage = (passing_count / len(df)) * 100
    print(f"  All Layers: {percentage:.1f}%")


# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the results
results = pd.read_csv('activation_diff_results_formula.csv')

# Define threshold values to test
thresholds = np.logspace(-6, 0, 100)  # From 1e-6 to 1

# Create figure with 1x3 subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 16))
fig.suptitle('Threshold Pass Rate Analysis by Metric Type: Non-Random Neuron Selection Strategy', 
             fontsize=16, fontweight='bold')

layers = ['fc1', 'fc2', 'fc3']
metrics = ['min_abs_diff', 'mean_abs_diff', 'max_abs_diff']
metric_titles = ['Minimum Activation Differences', 'Mean Activation Differences', 'Maximum Activation Differences']

# Calculate reasonable thresholds for heatmap based on data distribution
# Use percentiles that show good separation across all metrics
all_metric_percentiles = []
for metric in metrics:
    for layer in layers:
        column = f'{layer}_{metric}'
        values = results[column].values
        all_metric_percentiles.extend([
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            np.percentile(values, 90)
        ])

# Select 4 thresholds that span the range well
min_thresh = max(1e-6, np.min(all_metric_percentiles))
max_thresh = min(1.0, np.max(all_metric_percentiles))
log_min = np.log10(min_thresh)
log_max = np.log10(max_thresh)

# Use log-spaced thresholds that cover the data range
threshold_values = np.logspace(log_min, log_max, 4).tolist()
# Round to reasonable precision
threshold_values = [round(t, 6) if t < 0.001 else round(t, 4) for t in threshold_values]

layer_labels = {
    'fc1': r'$L_1$',
    'fc2': r'$L_2$', 
    'fc3': r'$L_3$'
}
def format_percentile_not_bold(val):
    if val < 0.001:
        exponent = int(np.floor(np.log10(abs(val))))
        mantissa = val / (10 ** exponent)
        return f"$ 10^{{{exponent}}}$"#{mantissa:.2f} \\times
    elif val < 0.1:
        return f"{val:.3f}"
    else:
        return f"{val:.3f}"
# Function to create heatmap data
def create_heatmap_data(metric_name):
    heatmap_data = []
    for layer in layers:
        column = f'{layer}_{metric_name}'
        values = results[column].values
        
        pass_rates = []
        for threshold in threshold_values:
            passing = np.sum(values <= threshold) / len(values) * 100
            pass_rates.append(passing)
        
        heatmap_data.append(pass_rates)
    
    return np.array(heatmap_data)

# Create heatmaps for each metric
for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes[idx]
    
    # Create heatmap data
    heatmap_data = create_heatmap_data(metric)
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=[f'{format_percentile_not_bold(t)}' for t in threshold_values],
                yticklabels=[f'{layer_labels[layer]}' for layer in layers],
                annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Pass Rate (%)'}, 
                vmin=0, vmax=100, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold Value', fontsize=12)
    
    # Only add y-label to the first subplot
    if idx == 0:
        #ax.set_ylabel('Layer', fontsize=12)
        pass
    else:
        ax.set_ylabel('')

plt.tight_layout()
save_path = 'TPR-SQ1.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()



