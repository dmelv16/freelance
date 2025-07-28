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
# This loop reads each file chunk by chunk without loading the whole file
for file_path in file_list:
    # THIS IS THE CORRECTED LOOP
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        chunk = batch.to_pandas()
        # Group the chunk and append each small group piece to our dictionary
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

# --- 3. Process Each Group and Collect Results ---
cleaned_dfs = []

print("\nStep 2: Cleaning each group one by one...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    # Combine all the small pieces for this one group into a single DataFrame
    full_group_df = pd.concat(list_of_chunks)
    
    # Run our trusted cleaning function on the complete group
    cleaned_group = clean_one_group(full_group_df)
    cleaned_dfs.append(cleaned_group)

# --- 4. Combine All Cleaned Groups ---
print("\nStep 3: Combining all cleaned groups into the final result...")
final_df = pd.concat(cleaned_dfs, ignore_index=True)
print("Processing complete.")

# --- 5. Summarize Changes ---
# (This section remains the same)
print("\n" + "="*35)
print("        Cleaning Summary")
print("="*35)
changes_mask = final_df['dc1_status'] != final_df['Cleaned_Status']
changed_rows = final_df[changes_mask]

if changed_rows.empty:
    print("No classifications were changed.")
else:
    change_counts = changed_rows.groupby(['dc1_status', 'Cleaned_Status']).size().reset_index(name='count')
    change_counts.rename(columns={'dc1_status': 'Original State', 'Cleaned_Status': 'New State'}, inplace=True)
    print("Breakdown of classification changes:")
    print(change_counts.to_string(index=False))
    print(f"\nTotal classifications changed: {len(changed_rows)}")
print("="*35)

# Overwrite original column and drop the temp one
final_df['dc1_status'] = final_df['Cleaned_Status']
final_output = final_df.drop(columns=['Cleaned_Status', 'block_id'])


# --- 6. Final Voltage Analysis ---
print("\n--- Final Analysis of Cleaned Data ---")

# Filter for the statuses we want to verify
analysis_df = final_output[final_output['dc1_status'].isin(['Stabilizing', 'Steady State'])]

# Group by the cleaned status and get the min and max voltage for each
voltage_summary = analysis_df.groupby('dc1_status')['voltage_28v_dc1_cal'].agg(['min', 'max'])

print("Voltage Summary for Key States:")
print(voltage_summary)