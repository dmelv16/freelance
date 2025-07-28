import pandas as pd
import numpy as np
from tqdm import tqdm # A helpful library for progress bars: pip install tqdm

# --- 1. Define a Function to Clean a SINGLE Group ---
# This is the same function from our last attempt. It contains all the
# pandas logic to clean one complete group of data.
def clean_one_group(group_df):
    
    def classify_voltage(voltage):
        if voltage < 2.2:
            return 'De-energized'
        elif 2.2 <= voltage < 22:
            return 'Stabilizing'
        else: # voltage >= 22
            return 'Steady State'
            
    group_df = group_df.copy()
    
    # Create block_id temporarily; it will not be in the final output
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

# --- 2. Get All Unique Groups from Parquet Files (Low Memory) ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 

print("Step 1: Scanning files to find all unique groups...")
# Read ONLY the grouping columns from the files to find unique groups
all_groups_df = pd.concat(
    [pd.read_parquet(f, columns=group_cols) for f in file_list]
).drop_duplicates()
print(f"Found {len(all_groups_df)} unique groups to process.")

# --- 3. Loop, Read, Clean, and Collect ---
cleaned_dfs = [] # We'll store cleaned group DataFrames here

# Using tqdm for a progress bar
for index, group_key in tqdm(all_groups_df.iterrows(), total=len(all_groups_df)):
    # Create a filter expression for pyarrow
    # e.g., [('unit_id', '=', 'A'), ('test_run', '=', 1)]
    filters = [(col, '=', value) for col, value in group_key.items()]
    
    # Read only the data for this specific group from all files
    group_data = pd.concat(
        [pd.read_parquet(f, filters=filters) for f in file_list]
    )
    
    # Run our cleaning function on this small, in-memory DataFrame
    cleaned_group = clean_one_group(group_data)
    cleaned_dfs.append(cleaned_group)

# --- 4. Combine All Cleaned Groups ---
print("\nStep 3: Combining all cleaned groups...")
final_df = pd.concat(cleaned_dfs, ignore_index=True)
print("Processing complete.")

# --- 5. Summarize Changes ---
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