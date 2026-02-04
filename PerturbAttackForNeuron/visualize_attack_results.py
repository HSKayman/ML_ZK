# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['figure.dpi'] = 100

# Color palette - warm coral vs cool teal
COLORS = {
    'baseline': '#E07A5F',      # Terracotta
    'constrained': '#3D5A80',   # Dark blue
    'accent': '#81B29A',        # Sage green
    'background': '#F4F1DE'     # Cream
}

# %%
# =============================================================================
# Load Data
# =============================================================================

def load_data(csv_path="gradient_swap_attack_special_node_results.csv"):
    """Load and prepare the attack results data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Unique input_ids: {df['input_id'].nunique()}")
    print(f"Allowed neurons range: {df['allowed_neurons'].min()} - {df['allowed_neurons'].max()}")
    return df

# %%
# =============================================================================
# Statistical Summary
# =============================================================================

def print_statistical_summary(df):
    """Print statistical summary for baseline and constrained."""
    print("\n" + "="*70)
    print(" STATISTICAL SUMMARY: NUMBER OF NEURONS PERTURBED")
    print("="*70)
    
    baseline_neurons = df['baseline_num_neurons']
    constrained_neurons = df['constrained_num_neurons']
    
    stats_data = {
        'Metric': ['Mean', 'Std', 'Median', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)'],
        'Baseline': [
            baseline_neurons.mean(),
            baseline_neurons.std(),
            baseline_neurons.median(),
            baseline_neurons.min(),
            baseline_neurons.max(),
            baseline_neurons.quantile(0.25),
            baseline_neurons.quantile(0.75)
        ],
        'Constrained': [
            constrained_neurons.mean(),
            constrained_neurons.std(),
            constrained_neurons.median(),
            constrained_neurons.min(),
            constrained_neurons.max(),
            constrained_neurons.quantile(0.25),
            constrained_neurons.quantile(0.75)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(baseline_neurons, constrained_neurons)
    print(f"\nPaired t-test: t={t_stat:.4f}, p={p_val:.6f}")
    
    print("\n" + "="*70)
    print(" STATISTICAL SUMMARY: EPSILON VALUES")
    print("="*70)
    
    baseline_eps = df['baseline_epsilon']
    constrained_eps = df['constrained_epsilon']
    
    stats_eps = {
        'Metric': ['Mean', 'Std', 'Median', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)'],
        'Baseline': [
            baseline_eps.mean(),
            baseline_eps.std(),
            baseline_eps.median(),
            baseline_eps.min(),
            baseline_eps.max(),
            baseline_eps.quantile(0.25),
            baseline_eps.quantile(0.75)
        ],
        'Constrained': [
            constrained_eps.mean(),
            constrained_eps.std(),
            constrained_eps.median(),
            constrained_eps.min(),
            constrained_eps.max(),
            constrained_eps.quantile(0.25),
            constrained_eps.quantile(0.75)
        ]
    }
    
    stats_eps_df = pd.DataFrame(stats_eps)
    print(stats_eps_df.to_string(index=False))
    
    t_stat_eps, p_val_eps = stats.ttest_rel(baseline_eps, constrained_eps)
    print(f"\nPaired t-test: t={t_stat_eps:.4f}, p={p_val_eps:.6f}")
    
    return stats_df, stats_eps_df

# %%
# =============================================================================
# Visualization Functions
# =============================================================================

def plot_neurons_by_allowed(df, save_path=None):
    """Plot number of neurons perturbed vs allowed neurons."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Group by allowed_neurons and compute mean/std
    grouped = df.groupby('allowed_neurons').agg({
        'baseline_num_neurons': ['mean', 'std'],
        'constrained_num_neurons': ['mean', 'std']
    }).reset_index()
    
    grouped.columns = ['allowed_neurons', 'baseline_mean', 'baseline_std', 
                       'constrained_mean', 'constrained_std']
    
    # Left plot: Line plot with confidence bands
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    
    ax1.plot(grouped['allowed_neurons'], grouped['baseline_mean'], 
             color=COLORS['baseline'], linewidth=2.5, marker='o', 
             markersize=4, label='Baseline', alpha=0.9)
    ax1.fill_between(grouped['allowed_neurons'], 
                     grouped['baseline_mean'] - grouped['baseline_std'],
                     grouped['baseline_mean'] + grouped['baseline_std'],
                     color=COLORS['baseline'], alpha=0.2)
    
    ax1.plot(grouped['allowed_neurons'], grouped['constrained_mean'], 
             color=COLORS['constrained'], linewidth=2.5, marker='s', 
             markersize=4, label='Constrained', alpha=0.9)
    ax1.fill_between(grouped['allowed_neurons'], 
                     grouped['constrained_mean'] - grouped['constrained_std'],
                     grouped['constrained_mean'] + grouped['constrained_std'],
                     color=COLORS['constrained'], alpha=0.2)
    
    # Perfect matching line (allowed = used)
    ax1.plot(grouped['allowed_neurons'], grouped['allowed_neurons'], 
             'k--', alpha=0.4, label='Perfect Match (n=allowed)')
    
    ax1.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Neurons Perturbed', fontsize=12, fontweight='bold')
    ax1.set_title('Neurons Perturbed vs Allowed', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Box plot comparison
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    
    # Prepare data for boxplot
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({'allowed': row['allowed_neurons'], 
                         'neurons': row['baseline_num_neurons'], 
                         'type': 'Baseline'})
        plot_data.append({'allowed': row['allowed_neurons'], 
                         'neurons': row['constrained_num_neurons'], 
                         'type': 'Constrained'})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Sample some allowed_neurons values for cleaner visualization
    sample_allowed = [10, 50, 100, 150, 200, 250, 300]
    sample_df = plot_df[plot_df['allowed'].isin(sample_allowed)]
    
    sns.boxplot(data=sample_df, x='allowed', y='neurons', hue='type',
                palette=[COLORS['baseline'], COLORS['constrained']], ax=ax2)
    
    ax2.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Neurons Perturbed', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution by Allowed Neurons', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(title='Attack Type', loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

# %%
def plot_epsilon_comparison(df, save_path=None):
    """Plot epsilon values comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Group by allowed_neurons
    grouped = df.groupby('allowed_neurons').agg({
        'baseline_epsilon': ['mean', 'std'],
        'constrained_epsilon': ['mean', 'std']
    }).reset_index()
    
    grouped.columns = ['allowed_neurons', 'baseline_mean', 'baseline_std',
                       'constrained_mean', 'constrained_std']
    
    # Left: Line plot
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    
    ax1.plot(grouped['allowed_neurons'], grouped['baseline_mean'],
             color=COLORS['baseline'], linewidth=2.5, marker='o',
             markersize=4, label='Baseline ε', alpha=0.9)
    ax1.fill_between(grouped['allowed_neurons'],
                     grouped['baseline_mean'] - grouped['baseline_std'],
                     grouped['baseline_mean'] + grouped['baseline_std'],
                     color=COLORS['baseline'], alpha=0.2)
    
    ax1.plot(grouped['allowed_neurons'], grouped['constrained_mean'],
             color=COLORS['constrained'], linewidth=2.5, marker='s',
             markersize=4, label='Constrained ε', alpha=0.9)
    ax1.fill_between(grouped['allowed_neurons'],
                     grouped['constrained_mean'] - grouped['constrained_std'],
                     grouped['constrained_mean'] + grouped['constrained_std'],
                     color=COLORS['constrained'], alpha=0.2)
    
    ax1.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
    ax1.set_title('Epsilon vs Allowed Neurons', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Scatter plot of baseline vs constrained epsilon
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    
    scatter = ax2.scatter(df['baseline_epsilon'], df['constrained_epsilon'],
                          c=df['allowed_neurons'], cmap='viridis', alpha=0.6, 
                          s=30, edgecolors='white', linewidth=0.5)
    
    # Identity line
    max_eps = max(df['baseline_epsilon'].max(), df['constrained_epsilon'].max())
    ax2.plot([0, max_eps], [0, max_eps], 'r--', alpha=0.6, label='y=x (equal)')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Allowed Neurons', fontsize=11)
    
    ax2.set_xlabel('Baseline Epsilon', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Constrained Epsilon', fontsize=12, fontweight='bold')
    ax2.set_title('Baseline vs Constrained Epsilon', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

# %%
def plot_difference_analysis(df, save_path=None):
    """Analyze and plot differences between baseline and constrained."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Compute differences
    df['neuron_diff'] = df['constrained_num_neurons'] - df['baseline_num_neurons']
    df['epsilon_diff_computed'] = df['constrained_epsilon'] - df['baseline_epsilon']
    
    # Top-left: Histogram of neuron differences
    ax1 = axes[0, 0]
    ax1.set_facecolor('#FFFFFF')
    ax1.hist(df['neuron_diff'], bins=30, color=COLORS['accent'], 
             edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference')
    ax1.axvline(x=df['neuron_diff'].mean(), color=COLORS['baseline'], 
                linestyle='-', linewidth=2, label=f'Mean: {df["neuron_diff"].mean():.2f}')
    ax1.set_xlabel('Constrained - Baseline (Neurons)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Neuron Difference', fontsize=14, fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Histogram of epsilon differences
    ax2 = axes[0, 1]
    ax2.set_facecolor('#FFFFFF')
    ax2.hist(df['epsilon_diff_computed'], bins=30, color=COLORS['accent'],
             edgecolor='white', alpha=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference')
    ax2.axvline(x=df['epsilon_diff_computed'].mean(), color=COLORS['baseline'],
                linestyle='-', linewidth=2, label=f'Mean: {df["epsilon_diff_computed"].mean():.4f}')
    ax2.set_xlabel('Constrained - Baseline (Epsilon)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Epsilon Difference', fontsize=14, fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Neuron diff vs allowed neurons
    ax3 = axes[1, 0]
    ax3.set_facecolor('#FFFFFF')
    
    grouped_diff = df.groupby('allowed_neurons')['neuron_diff'].agg(['mean', 'std']).reset_index()
    ax3.plot(grouped_diff['allowed_neurons'], grouped_diff['mean'],
             color=COLORS['constrained'], linewidth=2.5, marker='o', markersize=4)
    ax3.fill_between(grouped_diff['allowed_neurons'],
                     grouped_diff['mean'] - grouped_diff['std'],
                     grouped_diff['mean'] + grouped_diff['std'],
                     color=COLORS['constrained'], alpha=0.2)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.6)
    ax3.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Neuron Difference (Constrained - Baseline)', fontsize=12, fontweight='bold')
    ax3.set_title('Neuron Difference by Allowed Neurons', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Epsilon diff vs allowed neurons
    ax4 = axes[1, 1]
    ax4.set_facecolor('#FFFFFF')
    
    grouped_eps_diff = df.groupby('allowed_neurons')['epsilon_diff_computed'].agg(['mean', 'std']).reset_index()
    ax4.plot(grouped_eps_diff['allowed_neurons'], grouped_eps_diff['mean'],
             color=COLORS['baseline'], linewidth=2.5, marker='s', markersize=4)
    ax4.fill_between(grouped_eps_diff['allowed_neurons'],
                     grouped_eps_diff['mean'] - grouped_eps_diff['std'],
                     grouped_eps_diff['mean'] + grouped_eps_diff['std'],
                     color=COLORS['baseline'], alpha=0.2)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.6)
    ax4.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Epsilon Difference (Constrained - Baseline)', fontsize=12, fontweight='bold')
    ax4.set_title('Epsilon Difference by Allowed Neurons', fontsize=14, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

# %%
def plot_heatmap_by_input(df, save_path=None):
    """Create heatmap showing metrics by input_id and allowed_neurons."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Pivot for baseline neurons
    pivot_baseline = df.pivot_table(
        values='baseline_num_neurons', 
        index='input_id', 
        columns='allowed_neurons',
        aggfunc='mean'
    )
    
    # Pivot for constrained neurons
    pivot_constrained = df.pivot_table(
        values='constrained_num_neurons',
        index='input_id',
        columns='allowed_neurons',
        aggfunc='mean'
    )
    
    # Select subset of columns for better visualization
    sample_cols = [c for c in pivot_baseline.columns if c <= 150]
    
    ax1 = axes[0]
    sns.heatmap(pivot_baseline[sample_cols], cmap='YlOrRd', ax=ax1, 
                cbar_kws={'label': 'Neurons'})
    ax1.set_title('Baseline: Neurons Perturbed', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Allowed Neurons', fontsize=12)
    ax1.set_ylabel('Input ID', fontsize=12)
    
    ax2 = axes[1]
    sns.heatmap(pivot_constrained[sample_cols], cmap='YlGnBu', ax=ax2,
                cbar_kws={'label': 'Neurons'})
    ax2.set_title('Constrained: Neurons Perturbed', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Allowed Neurons', fontsize=12)
    ax2.set_ylabel('Input ID', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

# %%
def plot_success_rate_comparison(df, save_path=None):
    """Compare success rates between baseline and constrained."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor('#FFFFFF')
    
    # Calculate success rates by allowed_neurons
    success_df = df.groupby('allowed_neurons').agg({
        'baseline_success': 'mean',
        'constrained_success': 'mean'
    }).reset_index()
    
    success_df['baseline_success'] *= 100
    success_df['constrained_success'] *= 100
    
    ax.plot(success_df['allowed_neurons'], success_df['baseline_success'],
            color=COLORS['baseline'], linewidth=2.5, marker='o',
            markersize=5, label='Baseline Success Rate')
    
    ax.plot(success_df['allowed_neurons'], success_df['constrained_success'],
            color=COLORS['constrained'], linewidth=2.5, marker='s',
            markersize=5, label='Constrained Success Rate')
    
    ax.set_xlabel('Allowed Neurons', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Success Rate vs Allowed Neurons', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

# %%
def create_summary_table(df):
    """Create a summary table grouped by allowed_neurons ranges."""
    # Create bins for allowed_neurons
    bins = [0, 50, 100, 150, 200, 250, 300, 350]
    labels = ['1-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-350']
    df['neuron_range'] = pd.cut(df['allowed_neurons'], bins=bins, labels=labels)
    
    summary = df.groupby('neuron_range', observed=True).agg({
        'baseline_num_neurons': ['mean', 'std'],
        'constrained_num_neurons': ['mean', 'std'],
        'baseline_epsilon': ['mean', 'std'],
        'constrained_epsilon': ['mean', 'std'],
        'baseline_success': 'mean',
        'constrained_success': 'mean'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    print("\n" + "="*80)
    print(" SUMMARY BY ALLOWED NEURON RANGE")
    print("="*80)
    print(summary.to_string())
    
    return summary

# %%
# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Load data
    df = load_data("./PerturbAttackForNeuron/gradient_swap_attack_special_node_results_2026-01-29_15-10-39.csv")
    
    # Print statistical summary
    print_statistical_summary(df)
    
    # Create summary table
    summary = create_summary_table(df)
    
    # Generate visualizations
    print("\n" + "="*70)
    print(" GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1. Neurons Perturbed Analysis...")
    plot_neurons_by_allowed(df, save_path="fig_neurons_comparison.png")
    
    print("\n2. Epsilon Comparison...")
    plot_epsilon_comparison(df, save_path="fig_epsilon_comparison.png")
    
    print("\n3. Difference Analysis...")
    plot_difference_analysis(df, save_path="fig_difference_analysis.png")
    
    print("\n4. Success Rate Comparison...")
    plot_success_rate_comparison(df, save_path="fig_success_rate.png")
    
    print("\n5. Heatmap by Input...")
    plot_heatmap_by_input(df, save_path="fig_heatmap_by_input.png")
    
    print("\n" + "="*70)
    print(" VISUALIZATION COMPLETE!")
    print("="*70)
