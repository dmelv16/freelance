import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- 1. EDITED: The Final Cleaning Function with Stability Analysis ---
def clean_one_group(group_df):
    
    # --- Tunable Parameters ---
    # The number of data points to look at in our sliding window.
    STABILITY_WINDOW = 5 
    # The maximum standard deviation for the voltage to be considered "stable".
    STABILITY_THRESHOLD = 0.1 
    
    df = group_df.copy()
    df['voltage_28v_dc1_cal'] = pd.to_numeric(df['voltage_28v_dc1_cal'], errors='coerce')
    
    # --- First Pass: Calculate rolling stability ---
    df['voltage_std'] = df['voltage_28v_dc1_cal'].rolling(window=STABILITY_WINDOW).std()

    # --- Second Pass: Classify based on both voltage level and stability ---
    def classify_with_stability(row):
        voltage = row['voltage_28v_dc1_cal']
        stability = row['voltage_std']
        
        if pd.isna(voltage) or voltage < 2.2:
            return 'De-energized'
        
        # A point is 'Steady State' ONLY if the voltage is high AND it has stabilized.
        if voltage >= 22 and not pd.isna(stability) and stability < STABILITY_THRESHOLD:
            return 'Steady State'
        
        # If the voltage is high but not yet stable, it's still stabilizing.
        return 'Stabilizing'

    df['Cleaned_Status'] = df.apply(classify_with_stability, axis=1)

    # --- Third Pass: Clean up any brief dips within a steady state block ---
    steady_indices = df.index[df['Cleaned_Status'] == 'Steady State']
    if not steady_indices.empty:
        first_steady_index, last_steady_index = steady_indices[0], steady_indices[-1]
        is_stabilizing = df['Cleaned_Status'] == 'Stabilizing'
        is_between = (df.index > first_steady_index) & (df.index < last_steady_index)
        df.loc[is_stabilizing & is_between, 'Cleaned_Status'] = 'Steady State'

    df['dc1_status'] = df['Cleaned_Status']
    # Return the final DataFrame without the temporary columns
    return df.drop(columns=['Cleaned_Status', 'voltage_std'])

# --- 2. Setup ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 
output_directory = 'output_plots_cleaned'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")

# --- 3. Read Files and Gather Groups ---
groups_data = defaultdict(list)
print("Step 1: Reading files and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

# --- 4. Loop Through Each Group, Clean, and Create a Plot ---
print(f"\nStep 2: Cleaning data and generating a plot for each group...")

for group_key, list_of_chunks in tqdm(groups_data.items()):
    # Combine all the small pieces for this one group into a single DataFrame
    full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
    
    # Run the cleaning function on the group's data
    cleaned_group_df = clean_one_group(full_group_df)
    
    # Create a unique filename for the plot
    filename_parts = [f"{col}_{val}" for col, val in zip(group_cols, group_key)]
    plot_filename = "-".join(filename_parts) + ".png"
    output_path = os.path.join(output_directory, plot_filename)

    # Generate the Plot using the CLEANED data
    fig, ax = plt.subplots(figsize=(12, 7))
    
    color_map = {
        'De-energized': '#4A90E2', # Blue
        'Stabilizing': '#F5A623',  # Orange
        'Steady State': '#7ED321'  # Green
    }
    
    for status, color in color_map.items():
        subset = cleaned_group_df[cleaned_group_df['dc1_status'] == status]
        if not subset.empty:
            ax.scatter(subset['timestamp'], subset['voltage_28v_dc1_cal'], 
                       c=color, label=status, s=15, alpha=0.8)

    ax.axhline(22, color='gray', linestyle='--', linewidth=1, label='Original 22V Threshold')
    
    title_str = "\n".join(filename_parts)
    ax.set_title(f"Cleaned Voltage Profile\n{title_str}", fontsize=10)
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel("Voltage (voltage_28v_dc1_cal)", fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path, dpi=100)
    
    # **IMPORTANT**: Close the figure to free up memory
    plt.close(fig)

print(f"\nProcessing complete. All plots of cleaned data saved to the '{output_directory}' folder.")
