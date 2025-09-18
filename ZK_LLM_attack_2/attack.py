# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pickle

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
print(f"Using device: {DEVICE}")

# %%
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map='auto' if torch.cuda.is_available() else None
)
model.eval()


# %%
# Helper function to capture activations
def get_activation(name, activations_dict):
    def hook(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            activations_dict[name] = output[0].detach().clone()
        else:
            activations_dict[name] = output.detach().clone()
    return hook

# %%
# Generate sample inputs for analysis
def generate_sample_inputs(tokenizer, n_samples=100, seq_length=32):    
    sample_texts = [
        "The future of artificial intelligence",
        "Climate change affects global weather",
        "Machine learning algorithms can",
        "Deep neural networks are",
        "Natural language processing enables",
        "Computer vision systems detect",
        "Quantum computing will revolutionize",
        "Blockchain technology provides",
        "Renewable energy sources include",
        "Medical research has shown",
        "Space exploration reveals",
        "Economic policies influence",
        "Educational systems should",
        "Transportation networks connect",
        "Communication technologies enable"
    ]
    
    inputs = []
    for i in range(n_samples):
        # Cycle through sample texts and add variations
        base_text = sample_texts[i % len(sample_texts)]
        
        # Add some randomness
        if i > len(sample_texts):
            base_text = base_text + f" in {2020 + (i % 10)} with"
        
        # Tokenize
        tokenized = tokenizer(
            base_text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=seq_length
        )
        inputs.append(tokenized.input_ids.to(model.device))
    
    return inputs

# %%
# Generate activation differences for Llama-2 layers
def generate_activation_differences_llama(model, X_data, n_samples=50, n_reconstructions=3):
    results = []
    
    # Select specific layers to analyze (first few transformer layers)
    # layer_names = [
    #     'model.layers.0',  # First transformer layer
    #     'model.layers.1',  # Second transformer layer  
    #     'model.layers.2',  # Third transformer layer
    # ]
    layer_names = [f'model.layers.{i}' for i in range(len(model.model.layers))]
    for sample_idx in tqdm(range(min(n_samples, len(X_data))), desc="Processing samples"):
        original_input = X_data[sample_idx]
        
        # Get original activations
        original_activations = {}
        hooks = []
        
       # Register hooks for all layers
        for layer_name in layer_names:
            layer_module = model
            for attr in layer_name.split('.'):
                layer_module = getattr(layer_module, attr)
            hooks.append(layer_module.register_forward_hook(
                get_activation(layer_name, original_activations)
            ))
        
        # Get original output and activations
        with torch.no_grad():
            original_output = model(original_input).logits
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Multiple reconstruction attempts
        for recon_idx in range(n_reconstructions):
            # Initialize random input embeddings for reconstruction
            seq_length = original_input.shape[1]
            embedding_dim = model.config.hidden_size
            
            # Use embeddings instead of token IDs for gradient-based optimization
            reconstructed_embeddings = torch.randn(
                1, seq_length, embedding_dim,
                device=model.device,
                dtype=torch.float32,
                requires_grad=True
            )
            
            optimizer = optim.Adam([reconstructed_embeddings], lr=0.001)
            
            # Reconstruction optimization
            for iteration in range(10000):  # Reduced iterations for efficiency
                optimizer.zero_grad()
                
                # Forward pass with embeddings
                embeddings_model_dtype = reconstructed_embeddings.to(model.dtype)
                output = model(inputs_embeds=embeddings_model_dtype).logits
                
                # Loss: match original output
                loss = nn.functional.mse_loss(output.float(), original_output.float())
                
                # Add regularization
                reg_loss = 0.001 * torch.mean(reconstructed_embeddings ** 2)
                total_loss = loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                if total_loss.item() < 1e-4:
                    break
                if iteration % 100 == 0:
                    print(f"Sample {sample_idx}, Recon {recon_idx}, Iter {iteration}, Loss: {total_loss.item():.6f}")
            
            # Get reconstructed activations
            reconstructed_activations = {}
            hooks = []
            
            for layer_name in layer_names:
                layer_module = model
                for attr in layer_name.split('.'):
                    layer_module = getattr(layer_module, attr)
                hooks.append(layer_module.register_forward_hook(
                    get_activation(layer_name, reconstructed_activations)
                ))
            
            with torch.no_grad():
                embeddings_model_dtype = reconstructed_embeddings.to(model.dtype)
                _ = model(inputs_embeds=embeddings_model_dtype)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Calculate differences for each layer
            row = {'sample_idx': sample_idx, 'reconstruction_idx': recon_idx}
            
            # Store individual layer metrics
            all_layer_max_diffs = []
            
            for layer_name in layer_names:
                if layer_name in original_activations and layer_name in reconstructed_activations:
                    orig_act = original_activations[layer_name].flatten().float()
                    recon_act = reconstructed_activations[layer_name].flatten().float()
                    
                    abs_diff = torch.abs(orig_act - recon_act)
                    
                    layer_short = layer_name.split('.')[-1]  # Get layer number
                    row[f'layer_{layer_short}_min_abs_diff'] = abs_diff.min().item()
                    row[f'layer_{layer_short}_mean_abs_diff'] = abs_diff.mean().item()
                    row[f'layer_{layer_short}_max_abs_diff'] = abs_diff.max().item()
                    
                    all_layer_max_diffs.append(abs_diff.max().item())
            
            # Store the maximum difference across ALL layers
            if all_layer_max_diffs:
                row['all_layers_max_diff'] = max(all_layer_max_diffs)
                row['all_layers_min_of_max'] = min(all_layer_max_diffs)
            
            results.append(row)
    
    return pd.DataFrame(results)

# %%
# Generate sample data
print("Generating sample inputs...")
X_data = generate_sample_inputs(tokenizer, n_samples=20, seq_length=16) 
print(f"Generated {len(X_data)} samples")

# %%
# Generate results
print("Generating activation differences for Llama-2 layers...")
results = generate_activation_differences_llama(model, X_data, n_samples=10, n_reconstructions=3)
#406m 40.9s

# %%
# Save results
results.to_csv('llama2_activation_diff_results.csv', index=False)
print(f"Results saved. Shape: {results.shape}")
print("\nFirst few rows:")
print(results.head())

# %%
# Define threshold values to test
thresholds = np.logspace(-6, 0, 100)  # From 1e-6 to 1

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Llama-2 Activation Reconstruction Analysis', fontsize=16)

layers = ['0', '1', '2']  # Layer numbers
metrics = ['min_abs_diff', 'mean_abs_diff', 'max_abs_diff']
colors = {'min_abs_diff': 'blue', 'mean_abs_diff': 'green', 'max_abs_diff': 'red'}

# Plot for each layer
for idx, layer in enumerate(layers):
    if idx < 3:  # Only plot first 3 layers
        ax = axes[idx//2, idx%2]
        
        for metric in metrics:
            column = f'layer_{layer}_{metric}'
            if column in results.columns:
                values = results[column].values
                
                # Calculate percentage passing each threshold
                percentages = []
                for threshold in thresholds:
                    passing = np.sum(values <= threshold) / len(values) * 100
                    percentages.append(passing)
                
                # Plot cumulative distribution
                ax.semilogx(thresholds, percentages, 
                           label=f'Layer {layer} - {metric.replace("_abs_diff", "").capitalize()}',
                           color=colors[metric], linewidth=2, alpha=0.7)
        
        # Add reference lines
        ax.axvline(x=0.007, color='black', linestyle='--', alpha=0.5, label='0.007 threshold')
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=99, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Threshold Value', fontsize=12)
        ax.set_ylabel('Percentage Passing (%)', fontsize=12)
        ax.set_title(f'Transformer Layer {layer}', fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

# Plot ALL LAYERS simultaneous check
ax = axes[1, 1]

if 'all_layers_max_diff' in results.columns:
    values = results['all_layers_max_diff'].values
    percentages_all = []
    for threshold in thresholds:
        passing = np.sum(values <= threshold) / len(values) * 100
        percentages_all.append(passing)
    
    ax.semilogx(thresholds, percentages_all, 
               label='ALL Layers Must Pass (Worst Case)',
               color='darkred', linewidth=3)
    
    # Also plot individual layer maximums for comparison
    for layer in layers:
        column = f'layer_{layer}_max_abs_diff'
        if column in results.columns:
            values = results[column].values
            percentages = []
            for threshold in thresholds:
                passing = np.sum(values <= threshold) / len(values) * 100
                percentages.append(passing)
            ax.semilogx(thresholds, percentages, 
                       label=f'Layer {layer} only',
                       linewidth=1, alpha=0.5, linestyle='--')

ax.axvline(x=0.007, color='black', linestyle='--', alpha=0.5, label='0.007 threshold')
ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=99, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Threshold Value', fontsize=12)
ax.set_ylabel('Percentage Passing (%)', fontsize=12)
ax.set_title('ALL Layers Simultaneous Pass Rate', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('llama2_activation_reconstruction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create threshold analysis table
threshold_values = [0.001, 0.005, 0.007, 0.01, 0.05, 0.1]
analysis_results = []

# Individual layer analysis
for layer in layers:
    for threshold in threshold_values:
        row = {'Layer': f'Layer {layer}', 'Threshold': threshold}
        
        for metric in metrics:
            column = f'layer_{layer}_{metric}'
            if column in results.columns:
                values = results[column].values
                passing_percentage = np.sum(values <= threshold) / len(values) * 100
                row[f'{metric.replace("_abs_diff", "").capitalize()} Pass %'] = f'{passing_percentage:.1f}%'
        
        analysis_results.append(row)

# ALL LAYERS analysis
if 'all_layers_max_diff' in results.columns:
    for threshold in threshold_values:
        row = {'Layer': 'ALL LAYERS', 'Threshold': threshold}
        
        values = results['all_layers_max_diff'].values
        passing_percentage = np.sum(values <= threshold) / len(values) * 100
        row['Min Pass %'] = '-'
        row['Mean Pass %'] = '-'
        row['Max Pass %'] = f'{passing_percentage:.1f}%'
        
        analysis_results.append(row)

# Create DataFrame and display
threshold_df = pd.DataFrame(analysis_results)
print("\nLlama-2 Threshold Analysis Table:")
print("="*80)
print(threshold_df.to_string(index=False))

# Save to CSV
threshold_df.to_csv('llama2_threshold_analysis.csv', index=False)

# %%
# Summary statistics
print("\n" + "="*80)
print("LLAMA-2 ACTIVATION RECONSTRUCTION ANALYSIS SUMMARY")
print("="*80)

if 'all_layers_max_diff' in results.columns:
    values = results['all_layers_max_diff'].values
    
    print(f"\nDataset size: {len(results)} reconstruction attempts")
    print(f"Layers analyzed: {layers}")
    
    # Find threshold for different pass rates
    pass_rates = [90, 95, 99]
    print("\nThreshold needed for target pass rates (ALL LAYERS):")
    
    for rate in pass_rates:
        if len(values) > 0:
            threshold_for_rate = np.percentile(values, rate)
            print(f"  {rate}% pass rate: {threshold_for_rate:.6f}")
    
    # Statistics at specific thresholds
    print("\nPass rates at specific thresholds (ALL LAYERS):")
    for threshold in [0.001, 0.007, 0.01, 0.1]:
        pass_rate = np.sum(values <= threshold) / len(values) * 100
        print(f"  Threshold {threshold}: {pass_rate:.1f}% pass rate")

print("\nAnalysis completed! Check the generated plots and CSV files for detailed results.")

# %%
def generate_comprehensive_visualizations_all_layers(model, results):
    """Generate comprehensive visualizations for ALL layers of Llama 2"""
    
    # Get all layer numbers from the results columns
    layers = sorted(list(set([
        col.split('_')[1] for col in results.columns 
        if col.startswith('layer_') and col.split('_')[2] in ['min', 'mean', 'max']
    ])))
    
    metrics = ['min_abs_diff', 'mean_abs_diff', 'max_abs_diff']
    
    # 1. Layer Activation Distributions (Multiple Pages)
    layers_per_page = 4
    for page_start in range(0, len(layers), layers_per_page):
        page_layers = layers[page_start:page_start + layers_per_page]
        plt.figure(figsize=(20, 5))
        
        for i, layer in enumerate(page_layers, 1):
            plt.subplot(1, layers_per_page, i)
            for metric in metrics:
                column = f'layer_{layer}_{metric}'
                if column in results.columns:
                    sns.kdeplot(data=results[column], label=metric.replace('_abs_diff', ''))
            plt.title(f'Layer {layer} Activations')
            plt.xlabel('Absolute Difference')
            plt.ylabel('Density')
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'llama2_activations_page_{page_start//layers_per_page + 1}.png')
        plt.close()

    # 2. Success Rates Across All Layers
    threshold_values = [0.001, 0.003, 0.005, 0.007, 0.01, 0.05]
    success_rates = []
    
    for threshold in threshold_values:
        rates = {'threshold': threshold}
        for layer in layers:
            column = f'layer_{layer}_max_abs_diff'
            if column in results.columns:
                rate = (results[column] <= threshold).mean() * 100
                rates[f'Layer_{layer}'] = rate
        success_rates.append(rates)
    
    success_df = pd.DataFrame(success_rates)
    
    # Plot success rates (multiple lines)
    plt.figure(figsize=(15, 8))
    for layer in layers:
        if f'Layer_{layer}' in success_df.columns:
            plt.plot(success_df['threshold'], 
                    success_df[f'Layer_{layer}'], 
                    label=f'Layer {layer}',
                    alpha=0.7)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Threshold')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rates Across All Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('llama2_all_layers_success_rates.png')
    plt.close()

    # 3. Layer Performance Heatmap
    performance_data = pd.DataFrame(index=layers)
    for threshold in [0.001, 0.007, 0.01]:
        rates = []
        for layer in layers:
            column = f'layer_{layer}_max_abs_diff'
            if column in results.columns:
                rate = (results[column] <= threshold).mean() * 100
                rates.append(rate)
        performance_data[f'thresh_{threshold}'] = rates
    
    plt.figure(figsize=(10, len(layers)//2))
    sns.heatmap(performance_data, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Success Rate (%)'})
    plt.title('Layer Performance at Different Thresholds')
    plt.tight_layout()
    plt.savefig('llama2_layer_performance_heatmap.png')
    plt.close()

    # 4. Layer Difficulty Analysis
    difficulty_metrics = pd.DataFrame(index=layers)
    for layer in layers:
        column = f'layer_{layer}_max_abs_diff'
        if column in results.columns:
            difficulty_metrics.loc[layer, 'Median'] = results[column].median()
            difficulty_metrics.loc[layer, '90th Percentile'] = results[column].quantile(0.9)
            difficulty_metrics.loc[layer, '99th Percentile'] = results[column].quantile(0.99)
    
    plt.figure(figsize=(15, 8))
    difficulty_metrics.plot(marker='o')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Layer')
    plt.ylabel('Activation Difference')
    plt.title('Layer Difficulty Analysis')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('llama2_layer_difficulty.png')
    plt.close()

    # 5. Cross-Layer Correlation Analysis
    max_diff_columns = [f'layer_{l}_max_abs_diff' for l in layers 
                       if f'layer_{l}_max_abs_diff' in results.columns]
    correlation_matrix = results[max_diff_columns].corr()
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu', center=0)
    plt.title('Cross-Layer Correlation of Maximum Differences')
    plt.tight_layout()
    plt.savefig('llama2_cross_layer_correlation.png')
    plt.close()

    # 6. Layer-wise Success Distribution
    threshold = 0.007
    success_distribution = pd.DataFrame(index=['Pass Rate'])
    
    for layer in layers:
        column = f'layer_{layer}_max_abs_diff'
        if column in results.columns:
            success_distribution[f'Layer_{layer}'] = [(results[column] <= threshold).mean() * 100]
    
    plt.figure(figsize=(15, 6))
    success_distribution.T.plot(kind='bar')
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% target')
    plt.grid(True, alpha=0.3)
    plt.title(f'Layer-wise Success Distribution (threshold={threshold})')
    plt.xlabel('Layer')
    plt.ylabel('Pass Rate (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('llama2_layer_success_distribution.png')
    plt.close()

    # Save summary statistics
    summary_stats = pd.DataFrame(index=layers)
    for layer in layers:
        column = f'layer_{layer}_max_abs_diff'
        if column in results.columns:
            data = results[column]
            summary_stats.loc[layer, 'Mean'] = data.mean()
            summary_stats.loc[layer, 'Median'] = data.median()
            summary_stats.loc[layer, 'Std'] = data.std()
            summary_stats.loc[layer, 'Pass_Rate_0.007'] = (data <= 0.007).mean() * 100
    
    summary_stats.to_csv('llama2_layer_summary_stats.csv')

    return summary_stats

# %%


# %%



