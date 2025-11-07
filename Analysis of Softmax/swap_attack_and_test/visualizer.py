import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- Configuration ---
INPUT_FILE = 'hack_summary.csv'
TOP_N_LAYERS = 20 # How many layers to show in the bar chart

# --- 1. Load Data ---
print(f"--- Loading data from {INPUT_FILE} ---")
if not os.path.exists(INPUT_FILE):
    print(f"Error: Could not find the file {INPUT_FILE}")
    sys.exit()

try:
    df = pd.read_csv(INPUT_FILE)
except pd.errors.EmptyDataError:
    print(f"Error: The file {INPUT_FILE} is empty.")
    sys.exit()

# --- 2. Show Overall Summary ---
print("\n--- Overall Test Summary ---")
status_counts = df['passed_or_not'].value_counts()
print(status_counts)

# --- 3. Filter for FAILED tests ---
# We only care about tests that actually FAILED for this analysis
failed_tests_df = df[df['passed_or_not'] == 'FAILED'].copy()

if failed_tests_df.empty:
    print("\nAnalysis complete: No 'FAILED' tests were found. Nothing to plot.")
    sys.exit()

print(f"\nFound {len(failed_tests_df)} FAILED tests to analyze.")

# --- 4. Plot Distribution of Max Errors ---
print("\n--- Generating Max Error Distribution Plot ---")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Use a log scale for the x-axis, as errors are likely very small
# bins=50 creates 50 bars in the histogram
# kde=True adds a smooth line over the distribution
hist_plot = sns.histplot(
    data=failed_tests_df, 
    x='max_error', 
    bins=50, 
    log_scale=True,  # Log scale is crucial for seeing the spread
    kde=True
)

plt.title('Distribution of Max Difference for ADVERSAYRY FAILED', fontsize=16, fontweight='bold')
plt.xlabel('Max Calculation Difference (Log Scale)', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

# Save the figure
output_filename = 'max_error_distribution.png'
plt.tight_layout()
plt.savefig(output_filename)
plt.close()
print(f"Successfully saved plot to '{output_filename}'")
print("This plot shows the 'how bad' the errors were.")


# --- 5. Generate Vulnerable Layers Table & Plot ---
# Note: Based on your last script, 'strongest_layer' is the one with max_error
# (the one that caused the FAILED status). We will use that here.
print("\n--- Generating Vulnerable Layer Analysis ---")

# Get the count for each layer that appeared as the 'strongest_layer'
layer_counts = failed_tests_df['vulnerable_layer'].value_counts()

# Format as a table
layer_table = layer_counts.reset_index()
layer_table.columns = ['Layer', 'Failure Count']

# Print the table to the console
print(f"\n--- Top {TOP_N_LAYERS} Most Vulnerable Layers (by Failure Count) ---")
print(layer_table.head(TOP_N_LAYERS).to_string(index=False))

# --- Create a bar chart of the Top N layers ---
top_layers_df = layer_table.head(TOP_N_LAYERS)

# Create a tall figure for the horizontal bar chart
plt.figure(figsize=(12, 10)) 
bar_plot = sns.barplot(
    data=top_layers_df, 
    y='Layer',          # Layers on the y-axis
    x='Failure Count',  # Counts on the x-axis
    palette='viridis'   # A nice color palette
)
bar_plot.set(xscale="log")
plt.title(f'Top {TOP_N_LAYERS} Most Vulnerable Layers', fontsize=16, fontweight='bold')
plt.xlabel('Number of Times this Layer Had Min Difference', fontsize=12)
plt.ylabel('Layer Name', fontsize=12)

# Save the figure
output_filename = 'vulnerable_layers_barchart.png'
plt.tight_layout()
plt.savefig(output_filename)
plt.close()
print(f"\nSuccessfully saved bar chart to '{output_filename}'")
print("This plot shows 'which' layers are failing most often.")