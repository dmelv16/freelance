import pandas as pd
import numpy as np
import pyarrow.parquet as pq # Import the pyarrow library
from collections import defaultdict
from tqdm import tqdm # A helpful library for progress bars: pip install tqdm

# --- 1. Define a Function to Clean a SINGLE Group ---
# This function remains exactly the same.
def clean_one_group(group_df):
    
    def classify_voltage(voltage):
        if voltage < 2.2:
            return 'De-energized'
        elif 2.2 <= voltage < 22:
            return 'Stabilizing'
        else: # voltage >= 22
            return 'Steady State'
            
    group_df = group_df.copy()
    
    group_df['block_id'] = (group_df['dc1_status'] != group_df['dc1_status'].shift()).fillna(True).astype(int).cumsum()
    
    group_info = group_df.groupby('block_id').agg(
        min_voltage=('voltage_28v_dc1_cal', 'min'),
        max_voltage=('voltage_28v_dc1_cal', 'max'),
        state=('dc1_status', 'first')
    )
    
    is_unclean_steady = (group_info['state'] == 'Steady State') & (group_info['min_voltage'] < 22)
    is_unclean_stabilizing = (group_info['state'] == 'Stabilizing') & ((group_info['max_voltage'] >= 22) | (group_info['min_voltage'] < 2.2))
    unclean_block_ids = group_info[is_unclean_steady | is_unclean_stabilizing].index.tolist()
    
    group_df['Cleaned_Status'] = group_df['dc1_status']
    is_unclean_mask = group_df['block_id'].isin(unclean_block_ids)
    
    if is_unclean_mask.any():
        voltages_to_fix = group_df.loc[is_unclean_mask, 'voltage_28v_dc1_cal']
        corrections = voltages_to_fix.apply(classify_voltage)
        group_df.loc[is_unclean_mask, 'Cleaned_Status'] = corrections
    
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
# The resulting DataFrame has both original 'dc1_status' and new 'Cleaned_Status'
final_df = pd.concat(cleaned_dfs, ignore_index=True)
print("Processing complete.")

# --- 5. Perform and Export Before-and-After Analysis ---
print("\n" + "="*45)
print("  Performing Final Before-and-After Analysis")
print("="*45)

def perform_detailed_analysis(df, status_col, voltage_col, output_filename):
    """Groups a dataframe and calculates voltage stats, saving to CSV."""
    # Filter for the key states
    analysis_df = df[df[status_col].isin(['Stabilizing', 'Steady State'])]
    
    # Group by all the original grouping columns plus the status
    all_analysis_groups = group_cols + [status_col]
    
    print(f"\nGrouping and aggregating data based on '{status_col}'...")
    # Perform aggregation
    summary = analysis_df.groupby(all_analysis_groups)[voltage_col].agg(
        ['min', 'max', 'mean', 'std']
    ).reset_index()
    
    # Save to CSV
    summary.to_csv(output_filename, index=False)
    print(f"--> Analysis successful. Results saved to '{output_filename}'")
    return summary

# Run analysis on ORIGINAL data
before_summary = perform_detailed_analysis(
    df=final_df,
    status_col='dc1_status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_before_cleaning.csv'
)

# Run analysis on CLEANED data
after_summary = perform_detailed_analysis(
    df=final_df,
    status_col='Cleaned_Status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_after_cleaning.csv'
)

print("\n--- Analysis Complete ---")
print("\n'before_cleaning_summary.csv' contains stats on original data.")
print("'after_cleaning_summary.csv' contains stats on cleaned data.")