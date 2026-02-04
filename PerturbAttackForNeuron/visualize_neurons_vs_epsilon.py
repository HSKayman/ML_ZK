# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

# Color palette
COLORS = {
    'baseline': '#E07A5F',      # Terracotta
    'constrained': '#3D5A80',   # Dark blue
    'accent': '#81B29A',        # Sage green
    'highlight': '#F2CC8F',     # Gold
    'background': '#F4F1DE'     # Cream
}

# %%
# =============================================================================
# Load Data
# =============================================================================

df = pd.read_csv("./PerturbAttackForNeuron/gradient_swap_attack_special_node_results_2026-01-29_15-10-39.csv")
print(f"Loaded {len(df)} rows")
print(f"Unique inputs: {df['input_id'].nunique()}")

# %%
# =============================================================================
# FIGURE 1: Neurons vs Epsilon - SEPARATE Baseline and Constrained
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor(COLORS['background'])
fig.suptitle('Neurons Perturbed vs Epsilon: Core Analysis', fontsize=18, fontweight='bold', y=0.98)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# --- Plot 1: BASELINE Scatter ---
ax1 = axes[0, 0]
ax1.set_facecolor('#FFFFFF')

scatter1 = ax1.scatter(df['baseline_num_neurons'], df['baseline_epsilon'], 
                       c=COLORS['baseline'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

# Add trend line
z = np.polyfit(df['baseline_num_neurons'], df['baseline_epsilon'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df['baseline_num_neurons'].min(), df['baseline_num_neurons'].max(), 100)
ax1.plot(x_trend, p(x_trend), color='darkred', linewidth=2.5, linestyle='--', alpha=0.8)

ax1.set_xlabel('Number of Neurons Perturbed', fontsize=12, fontweight='bold')
ax1.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
ax1.set_title('BASELINE: Neurons vs Epsilon', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 310)
ax1.set_ylim(0, 22)

# Add stats text
r_base, _ = stats.pearsonr(df['baseline_num_neurons'], df['baseline_epsilon'])
ax1.text(0.95, 0.95, f'r = {r_base:.3f}\nn = {len(df)}', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Plot 2: CONSTRAINED Scatter ---
ax2 = axes[0, 1]
ax2.set_facecolor('#FFFFFF')

scatter2 = ax2.scatter(df['constrained_num_neurons'], df['constrained_epsilon'],
                       c=COLORS['constrained'], alpha=0.6, s=50, marker='s', edgecolors='white', linewidth=0.5)

# Add trend line
z2 = np.polyfit(df['constrained_num_neurons'], df['constrained_epsilon'], 2)
p2 = np.poly1d(z2)
x_trend2 = np.linspace(df['constrained_num_neurons'].min(), df['constrained_num_neurons'].max(), 100)
ax2.plot(x_trend2, p2(x_trend2), color='darkblue', linewidth=2.5, linestyle='--', alpha=0.8)

ax2.set_xlabel('Number of Neurons Perturbed', fontsize=12, fontweight='bold')
ax2.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
ax2.set_title('CONSTRAINED: Neurons vs Epsilon', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 310)
ax2.set_ylim(0, 22)

# Add stats text
r_cons, _ = stats.pearsonr(df['constrained_num_neurons'], df['constrained_epsilon'])
ax2.text(0.95, 0.95, f'r = {r_cons:.3f}\nn = {len(df)}', 
         transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Plot 3: BASELINE Log-Log ---
ax3 = axes[1, 0]
ax3.set_facecolor('#FFFFFF')

ax3.scatter(df['baseline_num_neurons'], df['baseline_epsilon'],
            c=COLORS['baseline'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Neurons (log scale)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Epsilon (log scale)', fontsize=12, fontweight='bold')
ax3.set_title('BASELINE: Log-Log Scale', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax3.grid(True, alpha=0.3, which='both')

# Power law fit
log_n = np.log(df['baseline_num_neurons'])
log_eps = np.log(df['baseline_epsilon'])
slope, intercept, r_val, _, _ = stats.linregress(log_n, log_eps)
x_fit = np.linspace(df['baseline_num_neurons'].min(), df['baseline_num_neurons'].max(), 100)
y_fit = np.exp(intercept) * x_fit ** slope
ax3.plot(x_fit, y_fit, 'r-', linewidth=2.5, label=f'ε = {np.exp(intercept):.2f} × n^{slope:.2f}')
ax3.legend(loc='upper right', fontsize=10)

# --- Plot 4: CONSTRAINED Log-Log ---
ax4 = axes[1, 1]
ax4.set_facecolor('#FFFFFF')

ax4.scatter(df['constrained_num_neurons'], df['constrained_epsilon'],
            c=COLORS['constrained'], alpha=0.6, s=50, marker='s', edgecolors='white', linewidth=0.5)

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Neurons (log scale)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Epsilon (log scale)', fontsize=12, fontweight='bold')
ax4.set_title('CONSTRAINED: Log-Log Scale', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax4.grid(True, alpha=0.3, which='both')

# Power law fit
log_n_c = np.log(df['constrained_num_neurons'])
log_eps_c = np.log(df['constrained_epsilon'])
slope_c, intercept_c, r_val_c, _, _ = stats.linregress(log_n_c, log_eps_c)
x_fit_c = np.linspace(df['constrained_num_neurons'].min(), df['constrained_num_neurons'].max(), 100)
y_fit_c = np.exp(intercept_c) * x_fit_c ** slope_c
ax4.plot(x_fit_c, y_fit_c, 'b-', linewidth=2.5, label=f'ε = {np.exp(intercept_c):.2f} × n^{slope_c:.2f}')
ax4.legend(loc='upper right', fontsize=10)

plt.savefig('fig_neurons_vs_epsilon_main.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
plt.show()

# %%
# =============================================================================
# FIGURE 2: Trade-off Analysis - SEPARATE graphs
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor(COLORS['background'])
fig.suptitle('Neurons-Epsilon Trade-off Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Hyperbolic model
def hyperbolic(n, k):
    return k / n

# Fit models
popt_base, _ = curve_fit(hyperbolic, df['baseline_num_neurons'], df['baseline_epsilon'], p0=[100])
popt_cons, _ = curve_fit(hyperbolic, df['constrained_num_neurons'], df['constrained_epsilon'], p0=[100])
n_range = np.linspace(10, 290, 100)

# --- Plot 1: BASELINE hyperbolic fit ---
ax1 = axes[0, 0]
ax1.set_facecolor('#FFFFFF')

ax1.scatter(df['baseline_num_neurons'], df['baseline_epsilon'],
            c=COLORS['baseline'], alpha=0.5, s=40)
ax1.plot(n_range, hyperbolic(n_range, *popt_base), 
         color='darkred', linewidth=3, linestyle='--',
         label=f'Fit: ε = {popt_base[0]:.1f}/n')

ax1.set_xlabel('Number of Neurons (n)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
ax1.set_title('BASELINE: Hyperbolic Trade-off', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 310)
ax1.set_ylim(0, 22)

# --- Plot 2: CONSTRAINED hyperbolic fit ---
ax2 = axes[0, 1]
ax2.set_facecolor('#FFFFFF')

ax2.scatter(df['constrained_num_neurons'], df['constrained_epsilon'],
            c=COLORS['constrained'], alpha=0.5, s=40, marker='s')
ax2.plot(n_range, hyperbolic(n_range, *popt_cons),
         color='darkblue', linewidth=3, linestyle='--',
         label=f'Fit: ε = {popt_cons[0]:.1f}/n')

ax2.set_xlabel('Number of Neurons (n)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
ax2.set_title('CONSTRAINED: Hyperbolic Trade-off', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 310)
ax2.set_ylim(0, 22)

# --- Plot 3: BASELINE - Mean epsilon by allowed neurons ---
ax3 = axes[1, 0]
ax3.set_facecolor('#FFFFFF')

grouped = df.groupby('allowed_neurons').agg({
    'baseline_num_neurons': 'mean',
    'baseline_epsilon': ['mean', 'std']
}).reset_index()
grouped.columns = ['allowed', 'neurons', 'eps_mean', 'eps_std']

ax3.errorbar(grouped['allowed'], grouped['eps_mean'], yerr=grouped['eps_std'],
             color=COLORS['baseline'], linewidth=2.5, marker='o', markersize=5,
             capsize=3, alpha=0.9, ecolor='lightcoral')
ax3.fill_between(grouped['allowed'], 
                 grouped['eps_mean'] - grouped['eps_std'],
                 grouped['eps_mean'] + grouped['eps_std'],
                 color=COLORS['baseline'], alpha=0.15)

ax3.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean Epsilon ± Std', fontsize=12, fontweight='bold')
ax3.set_title('BASELINE: Epsilon vs Allowed Neurons', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax3.grid(True, alpha=0.3)

# --- Plot 4: CONSTRAINED - Mean epsilon by allowed neurons ---
ax4 = axes[1, 1]
ax4.set_facecolor('#FFFFFF')

grouped_c = df.groupby('allowed_neurons').agg({
    'constrained_num_neurons': 'mean',
    'constrained_epsilon': ['mean', 'std']
}).reset_index()
grouped_c.columns = ['allowed', 'neurons', 'eps_mean', 'eps_std']

ax4.errorbar(grouped_c['allowed'], grouped_c['eps_mean'], yerr=grouped_c['eps_std'],
             color=COLORS['constrained'], linewidth=2.5, marker='s', markersize=5,
             capsize=3, alpha=0.9, ecolor='lightsteelblue')
ax4.fill_between(grouped_c['allowed'],
                 grouped_c['eps_mean'] - grouped_c['eps_std'],
                 grouped_c['eps_mean'] + grouped_c['eps_std'],
                 color=COLORS['constrained'], alpha=0.15)

ax4.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
ax4.set_ylabel('Mean Epsilon ± Std', fontsize=12, fontweight='bold')
ax4.set_title('CONSTRAINED: Epsilon vs Allowed Neurons', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax4.grid(True, alpha=0.3)

plt.savefig('fig_neurons_epsilon_tradeoff.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
plt.show()

# %%
# =============================================================================
# FIGURE 3: Statistical Analysis - SEPARATE
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor(COLORS['background'])
fig.suptitle('Neurons vs Epsilon: Statistical Distributions', fontsize=18, fontweight='bold', y=0.98)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# --- Plot 1: BASELINE 2D Histogram ---
ax1 = axes[0, 0]
ax1.set_facecolor('#FFFFFF')

h1 = ax1.hist2d(df['baseline_num_neurons'], df['baseline_epsilon'], 
                bins=25, cmap='OrRd', cmin=1)
plt.colorbar(h1[3], ax=ax1, label='Count')

slope_b, intercept_b, r_b, _, _ = stats.linregress(df['baseline_num_neurons'], df['baseline_epsilon'])
x_line = np.array([df['baseline_num_neurons'].min(), df['baseline_num_neurons'].max()])
ax1.plot(x_line, slope_b * x_line + intercept_b, 'b-', linewidth=2.5)

ax1.set_xlabel('Neurons Perturbed', fontsize=12, fontweight='bold')
ax1.set_ylabel('Epsilon', fontsize=12, fontweight='bold')
ax1.set_title(f'BASELINE: Joint Distribution (r={r_b:.3f})', fontsize=14, fontweight='bold', color=COLORS['baseline'])

# --- Plot 2: CONSTRAINED 2D Histogram ---
ax2 = axes[0, 1]
ax2.set_facecolor('#FFFFFF')

h2 = ax2.hist2d(df['constrained_num_neurons'], df['constrained_epsilon'],
                bins=25, cmap='Blues', cmin=1)
plt.colorbar(h2[3], ax=ax2, label='Count')

slope_c, intercept_c, r_c, _, _ = stats.linregress(df['constrained_num_neurons'], df['constrained_epsilon'])
ax2.plot(x_line, slope_c * x_line + intercept_c, 'r-', linewidth=2.5)

ax2.set_xlabel('Neurons Perturbed', fontsize=12, fontweight='bold')
ax2.set_ylabel('Epsilon', fontsize=12, fontweight='bold')
ax2.set_title(f'CONSTRAINED: Joint Distribution (r={r_c:.3f})', fontsize=14, fontweight='bold', color=COLORS['constrained'])

# --- Plot 3: BASELINE n×ε product ---
ax3 = axes[1, 0]
ax3.set_facecolor('#FFFFFF')

df['baseline_product'] = df['baseline_num_neurons'] * df['baseline_epsilon']
ax3.hist(df['baseline_product'], bins=35, color=COLORS['baseline'], edgecolor='white', alpha=0.8)
ax3.axvline(df['baseline_product'].mean(), color='darkred', linestyle='--', linewidth=2.5,
            label=f'Mean = {df["baseline_product"].mean():.1f}')
ax3.axvline(df['baseline_product'].median(), color='black', linestyle=':', linewidth=2,
            label=f'Median = {df["baseline_product"].median():.1f}')

ax3.set_xlabel('n × ε (Total Budget)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('BASELINE: Distribution of n × ε', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Plot 4: CONSTRAINED n×ε product ---
ax4 = axes[1, 1]
ax4.set_facecolor('#FFFFFF')

df['constrained_product'] = df['constrained_num_neurons'] * df['constrained_epsilon']
ax4.hist(df['constrained_product'], bins=35, color=COLORS['constrained'], edgecolor='white', alpha=0.8)
ax4.axvline(df['constrained_product'].mean(), color='darkblue', linestyle='--', linewidth=2.5,
            label=f'Mean = {df["constrained_product"].mean():.1f}')
ax4.axvline(df['constrained_product'].median(), color='black', linestyle=':', linewidth=2,
            label=f'Median = {df["constrained_product"].median():.1f}')

ax4.set_xlabel('n × ε (Total Budget)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('CONSTRAINED: Distribution of n × ε', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.savefig('fig_neurons_epsilon_stats.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
plt.show()

# %%
# =============================================================================
# FIGURE 4: Per-Input Heatmaps - SEPARATE
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor(COLORS['background'])
fig.suptitle('Per-Input Neurons-Epsilon Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

sample_cols_limit = 150

# --- Plot 1: BASELINE Epsilon heatmap ---
ax1 = axes[0, 0]
pivot_eps_b = df.pivot_table(values='baseline_epsilon', index='input_id', 
                              columns='allowed_neurons', aggfunc='mean')
sample_cols = [c for c in pivot_eps_b.columns if c <= sample_cols_limit]

sns.heatmap(pivot_eps_b[sample_cols], cmap='YlOrRd', ax=ax1,
            cbar_kws={'label': 'Epsilon', 'shrink': 0.8})
ax1.set_title('BASELINE: Epsilon by Input × Allowed', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax1.set_xlabel('Allowed Neurons', fontsize=11)
ax1.set_ylabel('Input ID', fontsize=11)

# --- Plot 2: CONSTRAINED Epsilon heatmap ---
ax2 = axes[0, 1]
pivot_eps_c = df.pivot_table(values='constrained_epsilon', index='input_id',
                              columns='allowed_neurons', aggfunc='mean')

sns.heatmap(pivot_eps_c[sample_cols], cmap='YlGnBu', ax=ax2,
            cbar_kws={'label': 'Epsilon', 'shrink': 0.8})
ax2.set_title('CONSTRAINED: Epsilon by Input × Allowed', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax2.set_xlabel('Allowed Neurons', fontsize=11)
ax2.set_ylabel('Input ID', fontsize=11)

# --- Plot 3: BASELINE Neurons used heatmap ---
ax3 = axes[1, 0]
pivot_n_b = df.pivot_table(values='baseline_num_neurons', index='input_id',
                            columns='allowed_neurons', aggfunc='mean')

sns.heatmap(pivot_n_b[sample_cols], cmap='Oranges', ax=ax3,
            cbar_kws={'label': 'Neurons Used', 'shrink': 0.8})
ax3.set_title('BASELINE: Neurons Used by Input × Allowed', fontsize=14, fontweight='bold', color=COLORS['baseline'])
ax3.set_xlabel('Allowed Neurons', fontsize=11)
ax3.set_ylabel('Input ID', fontsize=11)

# --- Plot 4: CONSTRAINED Neurons used heatmap ---
ax4 = axes[1, 1]
pivot_n_c = df.pivot_table(values='constrained_num_neurons', index='input_id',
                            columns='allowed_neurons', aggfunc='mean')

sns.heatmap(pivot_n_c[sample_cols], cmap='Blues', ax=ax4,
            cbar_kws={'label': 'Neurons Used', 'shrink': 0.8})
ax4.set_title('CONSTRAINED: Neurons Used by Input × Allowed', fontsize=14, fontweight='bold', color=COLORS['constrained'])
ax4.set_xlabel('Allowed Neurons', fontsize=11)
ax4.set_ylabel('Input ID', fontsize=11)

plt.savefig('fig_neurons_epsilon_detail.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
plt.show()

# %%
# =============================================================================
# FIGURE 5: Box plots - Grouped in 3 subplots (ranges 1-2, 3-4, 5-6)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor(COLORS['background'])
fig.suptitle('Epsilon Distribution by Neuron Range: Baseline vs Constrained', 
             fontsize=18, fontweight='bold', y=1.02)
plt.subplots_adjust(wspace=0.3)

# Create neuron range bins
df['neuron_range_b'] = pd.cut(df['baseline_num_neurons'], 
                               bins=[0, 50, 100, 150, 200, 250, 300],
                               labels=['1-50', '51-100', '101-150', '151-200', '201-250', '251-300'])
df['neuron_range_c'] = pd.cut(df['constrained_num_neurons'],
                               bins=[0, 50, 100, 150, 200, 250, 300],
                               labels=['1-50', '51-100', '101-150', '151-200', '201-250', '251-300'])

# Prepare data for grouped boxplot
box_data = []
for _, row in df.iterrows():
    if pd.notna(row['neuron_range_b']):
        box_data.append({
            'Neuron Range': row['neuron_range_b'],
            'Epsilon': row['baseline_epsilon'],
            'Type': 'Baseline'
        })
    if pd.notna(row['neuron_range_c']):
        box_data.append({
            'Neuron Range': row['neuron_range_c'],
            'Epsilon': row['constrained_epsilon'],
            'Type': 'Constrained'
        })

box_df = pd.DataFrame(box_data)

# Define range groups
range_groups = [
    (['1-50', '51-100'], 'Ranges 1-2: Small (1-100 neurons)'),
    (['101-150', '151-200'], 'Ranges 3-4: Medium (101-200 neurons)'),
    (['201-250', '251-300'], 'Ranges 5-6: Large (201-300 neurons)')
]

for idx, (ranges, title) in enumerate(range_groups):
    ax = axes[idx]
    ax.set_facecolor('#FFFFFF')
    
    # Filter data for this group
    group_df = box_df[box_df['Neuron Range'].isin(ranges)]
    
    # Create boxplot
    sns.boxplot(data=group_df, x='Neuron Range', y='Epsilon', hue='Type',
                palette={'Baseline': COLORS['baseline'], 'Constrained': COLORS['constrained']},
                ax=ax, width=0.6, gap=0.1)
    
    ax.set_xlabel('Neuron Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Only show legend on first subplot
    if idx == 0:
        ax.legend(title='Attack Type', fontsize=10, title_fontsize=11, loc='upper right')
    else:
        ax.legend_.remove()
    
    # Add mean annotations
    for i, nr in enumerate(ranges):
        base_vals = group_df[(group_df['Neuron Range'] == nr) & (group_df['Type'] == 'Baseline')]['Epsilon']
        cons_vals = group_df[(group_df['Neuron Range'] == nr) & (group_df['Type'] == 'Constrained')]['Epsilon']
        
        if len(base_vals) > 0:
            base_mean = base_vals.mean()
            ax.text(i - 0.2, base_mean + 0.3, f'μ={base_mean:.2f}', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['baseline'])
        if len(cons_vals) > 0:
            cons_mean = cons_vals.mean()
            ax.text(i + 0.2, cons_mean + 0.3, f'μ={cons_mean:.2f}', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['constrained'])

plt.savefig('fig_neurons_epsilon_boxplots.png', dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
plt.show()

# %%
# =============================================================================
# TABLE: Epsilon Statistics by Number of Neurons Perturbed (baseline_num_neurons)
# =============================================================================

# Group by baseline_num_neurons (actual neurons used)
table_data = df.groupby('baseline_num_neurons').agg({
    'baseline_epsilon': ['min', 'mean', 'max'],
    'constrained_epsilon': ['min', 'mean', 'max']
}).round(4)

table_data.columns = ['baseline_min_epsilon', 'baseline_avg_epsilon', 'baseline_max_epsilon',
                      'constrained_min_epsilon', 'constrained_avg_epsilon', 'constrained_max_epsilon']

# Reset index to make baseline_num_neurons a column
table_data = table_data.reset_index()
table_data = table_data.rename(columns={'baseline_num_neurons': 'num_neurons_perturbed'})

# Save to CSV
table_data.to_csv('table_neurons_epsilon_stats.csv', index=False)

print("\n" + "="*80)
print(" TABLE: Epsilon Statistics by Number of Neurons Perturbed (baseline_num_neurons)")
print("="*80)
print(table_data.to_string(index=False))
print("\n✓ Table saved to: table_neurons_epsilon_stats.csv")

# %%
# =============================================================================
# Statistical Summary
# =============================================================================

print("\n" + "="*80)
print(" NEURONS vs EPSILON: KEY STATISTICS")
print("="*80)

# Correlation
r_base, p_base = stats.pearsonr(df['baseline_num_neurons'], df['baseline_epsilon'])
r_cons, p_cons = stats.pearsonr(df['constrained_num_neurons'], df['constrained_epsilon'])

print(f"\n📊 Pearson Correlation:")
print(f"   Baseline:    r = {r_base:.4f}, p = {p_base:.2e}")
print(f"   Constrained: r = {r_cons:.4f}, p = {p_cons:.2e}")

# Spearman
rho_base, p_rho_base = stats.spearmanr(df['baseline_num_neurons'], df['baseline_epsilon'])
rho_cons, p_rho_cons = stats.spearmanr(df['constrained_num_neurons'], df['constrained_epsilon'])

print(f"\n📊 Spearman Rank Correlation:")
print(f"   Baseline:    ρ = {rho_base:.4f}, p = {p_rho_base:.2e}")
print(f"   Constrained: ρ = {rho_cons:.4f}, p = {p_rho_cons:.2e}")

# Power law
log_n_base = np.log(df['baseline_num_neurons'])
log_eps_base = np.log(df['baseline_epsilon'])
slope_base, intercept_base, _, _, _ = stats.linregress(log_n_base, log_eps_base)

log_n_cons = np.log(df['constrained_num_neurons'])
log_eps_cons = np.log(df['constrained_epsilon'])
slope_cons, intercept_cons, _, _, _ = stats.linregress(log_n_cons, log_eps_cons)

print(f"\n📊 Power Law Fit (ε = c·n^α):")
print(f"   Baseline:    α = {slope_base:.4f}, c = {np.exp(intercept_base):.4f}")
print(f"   Constrained: α = {slope_cons:.4f}, c = {np.exp(intercept_cons):.4f}")

# Hyperbolic
print(f"\n📊 Hyperbolic Fit (ε = k/n):")
print(f"   Baseline:    k = {popt_base[0]:.2f}")
print(f"   Constrained: k = {popt_cons[0]:.2f}")

# Product
print(f"\n📊 Product n × ε Statistics:")
print(f"   Baseline:    mean = {df['baseline_product'].mean():.2f}, std = {df['baseline_product'].std():.2f}")
print(f"   Constrained: mean = {df['constrained_product'].mean():.2f}, std = {df['constrained_product'].std():.2f}")

print("\n" + "="*80)
