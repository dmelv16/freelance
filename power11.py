import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# --- 1. FINALIZED CLEANING FUNCTIONS ---

# This is the most robust version of the cleaning logic we developed.
def classify_voltage_channel(voltage_series: pd.Series):
    """
    Analyzes voltage data using a fast, vectorized scan followed by a
    robust "Lock-In" to correctly handle mid-session dips.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22
    STABILITY_WINDOW = 10
    STABILITY_THRESHOLD = 0.1

    # Use a DataFrame for easier handling of columns
    temp_df = pd.DataFrame({'voltage': pd.to_numeric(voltage_series, errors='coerce')})
    temp_df['status'] = 'De-energized' # Initialize all points

    # --- Pass 1: Fast, Vectorized Initial Classification ---
    energized_mask = temp_df['voltage'] >= STEADY_VOLTAGE_THRESHOLD
    rolling_std = temp_df['voltage'].rolling(window=STABILITY_WINDOW).std()
    stable_mask = (rolling_std < STABILITY_THRESHOLD) & energized_mask
    
    temp_df.loc[stable_mask, 'status'] = 'Steady State'
    stabilizing_mask = energized_mask & (~stable_mask)
    temp_df.loc[stabilizing_mask, 'status'] = 'Stabilizing'

    # --- Pass 2: Find Absolute Boundaries and "Lock-In" Steady State ---
    steady_indices = temp_df.index[temp_df['status'] == 'Steady State']
    
    if not steady_indices.empty:
        first_steady_index = steady_indices[0]
        last_steady_index = steady_indices[-1]
        temp_df.loc[first_steady_index:last_steady_index, 'status'] = 'Steady State'
        
    return temp_df['status']


def clean_dc_channels(group_df):
    """
    Orchestrates the cleaning of both DC1 and DC2 channels.
    """
    df = group_df.copy()
    df['dc1_status'] = classify_voltage_channel(df['voltage_28v_dc1_cal'])
    if 'voltage_28v_dc2_cal' in df.columns:
        df['dc2_status'] = classify_voltage_channel(df['voltage_28v_dc2_cal'])
    else:
        df['dc2_status'] = 'De-energized'
    return df

# --- 2. Setup ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
output_directory = 'output_plots_cleaned'
exclusion_file_path = 'groups_to_exclude.csv' # Path to your exclusion CSV

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")

# --- INSERTED: Robust Fingerprint Helper Function ---
def create_fingerprint(df: pd.DataFrame, cols: list) -> pd.Series:
    """
    Creates a robust, type-safe string fingerprint for each row of a DataFrame.
    """
    temp_df = df[cols].copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(temp_df[col]):
            temp_df[col] = temp_df[col].map('{:.6f}'.format).str.rstrip('0').str.rstrip('.')
        else:
            temp_df[col] = temp_df[col].astype(str)
    return temp_df.agg('|'.join', axis=1)

# --- INSERTED: Load the Exclusion List ---
print(f"Step 1: Loading exclusion list from '{exclusion_file_path}'...")
try:
    df_to_exclude = pd.read_csv(exclusion_file_path)
    df_to_exclude.dropna(how='all', inplace=True)
    exclude_fingerprints = create_fingerprint(df_to_exclude, group_cols)
    groups_to_remove_fingerprints = set(exclude_fingerprints)
    print(f"Loaded {len(groups_to_remove_fingerprints)} unique group fingerprints to exclude.")
except FileNotFoundError:
    print(f"Warning: Exclusion file not found at '{exclusion_file_path}'. No groups will be excluded.")
    groups_to_remove_fingerprints = set()
except Exception as e:
    print(f"Error loading exclusion file: {e}. No groups will be excluded.")
    groups_to_remove_fingerprints = set()

# --- 4. MODIFIED: Read Files, Pre-Filter Rows, and Then Group ---
groups_data = defaultdict(list)
print(f"\nStep 2: Reading files, filtering rows, and gathering groups...")
for file_path in file_list:
    columns_to_read = list(set(['timestamp', 'voltage_28v_dc1_cal', 'voltage_28v_dc2_cal'] + group_cols))
    try:
        file_schema = pq.read_schema(file_path)
        final_columns = [col for col in columns_to_read if col in file_schema.names]
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=500_000, columns=final_columns):
            chunk = batch.to_pandas()
            if chunk.empty:
                continue

            # --- PRE-FILTERING LOGIC ---
            # 1. Create fingerprints for the incoming data chunk
            chunk_fingerprints = create_fingerprint(chunk, group_cols)
            
            # 2. Find which rows to KEEP
            mask_to_keep = ~chunk_fingerprints.isin(groups_to_remove_fingerprints)
            
            # 3. Create a new DataFrame containing only the rows we want to keep
            filtered_chunk = chunk[mask_to_keep]

            # 4. Now, group the CLEAN, pre-filtered data
            if not filtered_chunk.empty:
                for group_key, group_df in filtered_chunk.groupby(group_cols):
                    groups_data[group_key].append(group_df)
                    
    except Exception as e:
        print(f"Could not read {file_path}, error: {e}")
print(f"✅ Finished gathering data for {len(groups_data)} unique groups.")

# --- 4. Loop Through Each Group, Clean, and Create a Plot ---
print(f"\nStep 3: Cleaning data and generating a plot for each group...")

for group_key, list_of_chunks in tqdm(groups_data.items()):
    # Combine all the small pieces for this one group into a single DataFrame
    full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
    
    # **THIS IS THE NEW STEP**: Run the cleaning function on the group's data
    # NOTE: The user's original code had `clean_one_group`, this has been updated to the new orchestrator
    cleaned_group_df = clean_dc_channels(full_group_df)
    
    # --- Create a unique filename for the plot ---
    filename_parts = [f"{col}_{val}" for col, val in zip(group_cols, group_key)]
    plot_filename = "-".join(filename_parts) + ".png"
    # Clean filename for file system compatibility
    plot_filename = "".join(c for c in plot_filename if c.isalnum() or c in ('-', '_', '.')).rstrip()
    output_path = os.path.join(output_directory, plot_filename)

    # --- Generate the Plot using the CLEANED data ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define colors for each status
    color_map = {
        'De-energized': '#4A90E2', # Blue
        'Stabilizing': '#F5A623',  # Orange
        'Steady State': '#7ED321'  # Green
    }
    
    # Plot each status group from the cleaned data with its specific color
    # This now plots both DC1 and DC2 if present
    for status, color in color_map.items():
        # Plot DC1
        subset_dc1 = cleaned_group_df[cleaned_group_df['dc1_status'] == status]
        if not subset_dc1.empty:
            ax.scatter(subset_dc1['timestamp'], subset_dc1['voltage_28v_dc1_cal'],
                       c=color, label=f'DC1 {status}', s=20, marker='o', alpha=0.7)
        # Plot DC2
        if 'voltage_28v_dc2_cal' in cleaned_group_df.columns:
            subset_dc2 = cleaned_group_df[cleaned_group_df['dc2_status'] == status]
            if not subset_dc2.empty:
                ax.scatter(subset_dc2['timestamp'], subset_dc2['voltage_28v_dc2_cal'],
                           c=color, label=f'DC2 {status}', s=25, marker='x', alpha=0.9)


    ax.axhline(22, color='gray', linestyle='--', linewidth=1, label='Original 22V Threshold')
    
    title_str = "\n".join(filename_parts)
    ax.set_title(f"Cleaned Voltage Profile\n{title_str}", fontsize=10)
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel("Voltage", fontsize=10)
    # Adjust legend to handle potentially many labels from DC1 and DC2
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left")
    
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.savefig(output_path, dpi=100)
    
    # **IMPORTANT**: Close the figure to free up memory
    plt.close(fig)

print(f"\nProcessing complete. All plots of cleaned data saved to the '{output_directory}' folder.")
