import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm

# --- 1. Define a Function to Clean a SINGLE Group with Dynamic Thresholding ---
# This function now applies its logic to every group it receives.
def clean_one_group(group_df):
    
    # You can easily adjust this tolerance. 0.02 means the voltage must be
    # within 2% of the group's mean to be considered 'Steady State'.
    TOLERANCE_PERCENT = 0.02

    # --- First Pass: Calculate the dynamic threshold for THIS group ---
    
    # Find all points initially considered steady state to calculate a target voltage
    initial_steady_points = group_df[group_df['voltage_28v_dc1_cal'] >= 22]
    
    # Default to the fixed 22V if there are no steady points to analyze
    dynamic_threshold = 22.0 
    
    if not initial_steady_points.empty:
        # Calculate the group's specific mean operating voltage
        group_mean_voltage = initial_steady_points['voltage_28v_dc1_cal'].mean()
        # Calculate the new threshold based on the tolerance
        dynamic_threshold = group_mean_voltage * (1 - TOLERANCE_PERCENT)

    # --- Second Pass: Classify ALL data in the group using the new threshold ---
    
    def classify_voltage_dynamic(voltage):
        if voltage < 2.2:
            return 'De-energized'
        # A point is only 'Steady State' if it's above our new dynamic threshold
        elif voltage >= dynamic_threshold:
            return 'Steady State'
        # Otherwise, if it's above 2.2V but below the dynamic threshold, it's stabilizing.
        elif voltage >= 2.2:
            return 'Stabilizing'
        else:
            return 'De-energized'
            
    group_df = group_df.copy()
    
    # Apply the new classification to the entire group to create the 'Cleaned_Status' column
    group_df['Cleaned_Status'] = group_df['voltage_28v_dc1_cal'].apply(classify_voltage_dynamic)
    
    # The old 'unclean block' detection logic has been removed.
    
    return group_df

# --- 2. Read Files in Chunks and Gather Groups ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 
groups_data = defaultdict(list)

print("Step 1: Reading files in chunks and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

# --- 3. Process Each Group and Collect Results ---
cleaned_dfs = []
print("\nStep 2: Cleaning each group one by one...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    full_group_df = pd.concat(list_of_chunks)
    cleaned_group = clean_one_group(full_group_df)
    cleaned_dfs.append(cleaned_group)

# --- 4. Combine All Cleaned Groups ---
print("\nStep 3: Combining all cleaned groups into the final result...")
final_df = pd.concat(cleaned_dfs, ignore_index=True)
print("Processing complete.")

# --- 5. Perform and Export Before-and-After Analysis ---
print("\n" + "="*45)
print("  Performing Final Before-and-After Analysis")
print("="*45)

def perform_detailed_analysis(df, status_col, voltage_col, output_filename):
    """Groups a dataframe and calculates voltage stats, saving to CSV."""
    analysis_df = df[df[status_col].isin(['Stabilizing', 'Steady State'])]
    all_analysis_groups = group_cols + [status_col]
    print(f"\nGrouping and aggregating data based on '{status_col}'...")
    summary = analysis_df.groupby(all_analysis_groups)[voltage_col].agg(
        ['min', 'max', 'mean', 'std']
    ).reset_index()
    summary.to_csv(output_filename, index=False)
    print(f"--> Analysis successful. Results saved to '{output_filename}'")
    return summary

# Run analysis using the ORIGINAL 'dc1_status' column
before_summary = perform_detailed_analysis(
    df=final_df,
    status_col='dc1_status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_before_cleaning.csv'
)

# Run analysis using the NEW 'Cleaned_Status' column
after_summary = perform_detailed_analysis(
    df=final_df,
    status_col='Cleaned_Status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_after_cleaning.csv'
)

print("\n--- Analysis Complete ---")
