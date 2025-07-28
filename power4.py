import dask.dataframe as dd
import pandas as pd
import numpy as np

# --- 1. Define a Function to Clean a SINGLE Group ---
# This function still needs to create Cleaned_Status temporarily for the summary.
def clean_one_group(group_df):
    
    def classify_voltage(voltage):
        if voltage < 2.2:
            return 'De-energized'
        elif 2.2 <= voltage < 22:
            return 'Stabilizing'
        else:
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
    
    # Return the necessary columns for the next steps
    return group_df[['timestamp', 'voltage_28v_dc1_cal', 'dc1_status', 'Cleaned_Status']]

# --- 2. Read Data and Define Grouping ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
df = dd.read_parquet(file_list)

group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 
values_to_remove = ['Error 0 volt', 'Error missing volt', 'Normal transient', 'Abnormal transient']
df_cleaned = df[~df['dc1_trans'].isin(values_to_remove)]

# --- 3. Apply the Cleaning Function and Compute ---
meta = {
    'timestamp': 'float64', 
    'voltage_28v_dc1_cal': 'float64', 
    'dc1_status': 'object', 
    'Cleaned_Status': 'object'
}

print("Starting computation...")
final_df = df_cleaned.groupby(group_cols).apply(clean_one_group, meta=meta).compute()
print("Computation finished.")

# --- 4. Summarize Changes ---
print("\n" + "="*35)
print("        Cleaning Summary")
print("="*35)

final_df_reset = final_df.reset_index()
changes_mask = final_df_reset['dc1_status'] != final_df_reset['Cleaned_Status']
changed_rows = final_df_reset[changes_mask]

if changed_rows.empty:
    print("No classifications were changed.")
else:
    change_counts = changed_rows.groupby(['dc1_status', 'Cleaned_Status']).size().reset_index(name='count')
    change_counts.rename(columns={'dc1_status': 'Original State', 'Cleaned_Status': 'New State'}, inplace=True)
    print("Breakdown of classification changes:")
    print(change_counts.to_string(index=False))
    print(f"\nTotal classifications changed: {len(changed_rows)}")
print("="*35)

# --- 5. Finalize DataFrame Shape ---
# This new step overwrites the original column and drops the temporary one.
print("\nUpdating original columns and finalizing shape...")
final_df_reset['dc1_status'] = final_df_reset['Cleaned_Status']
final_output = final_df_reset.drop(columns=['Cleaned_Status'])

print("Final DataFrame shape matches original data.")
print(final_output.head())