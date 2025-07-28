import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm

# --- 1. Define the Final Cleaning Function ---
def clean_one_group(group_df):
    
    TOLERANCE_PERCENT = 0.02
    group_df = group_df.copy()

    # First Pass: Calculate dynamic threshold and make initial classification
    initial_steady_points = group_df[group_df['voltage_28v_dc1_cal'] >= 22]
    dynamic_threshold = 22.0 
    
    if not initial_steady_points.empty:
        group_mean_voltage = initial_steady_points['voltage_28v_dc1_cal'].mean()
        dynamic_threshold = group_mean_voltage * (1 - TOLERANCE_PERCENT)

    def initial_classify(voltage):
        if voltage < 2.2:
            return 'De-energized'
        elif voltage >= dynamic_threshold:
            return 'Steady State'
        else:
            return 'Stabilizing'
            
    group_df['Cleaned_Status'] = group_df['voltage_28v_dc1_cal'].apply(initial_classify)

    # Second Pass: Differentiate ramp-up, ramp-down, and dips
    steady_indices = group_df.index[group_df['Cleaned_Status'] == 'Steady State']
    
    if not steady_indices.empty:
        first_steady_index = steady_indices[0]
        last_steady_index = steady_indices[-1]

        is_stabilizing = group_df['Cleaned_Status'] == 'Stabilizing'
        is_between = (group_df.index > first_steady_index) & (group_df.index < last_steady_index)
        
        group_df.loc[is_stabilizing & is_between, 'Cleaned_Status'] = 'Steady State'

    # Overwrite the original status column for the final output
    group_df['dc1_status'] = group_df['Cleaned_Status']
    
    # Return the dataframe with all original columns, plus the updated status
    return group_df.drop(columns=['Cleaned_Status'], errors='ignore')

# --- 2. Read Files in Chunks and Gather Groups ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 
groups_data = defaultdict(list)

print("Step 1: Reading files in chunks and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    # Get all column names from the file schema
    all_columns = parquet_file.schema.names
    for batch in parquet_file.iter_batches(batch_size=500_000, columns=all_columns):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

# --- 3. Process Each Group and Write Directly to a New Parquet File ---
cleaned_dfs_for_analysis = [] # We still collect for final summary, but not for saving
output_filename = 'final_cleaned_data.parquet'
parquet_writer = None

# Get the schema from the first source file to use as our master template
source_file = pq.ParquetFile(file_list[0])
master_schema = source_file.schema
original_columns = master_schema.names # Get the original column order

print("Step 1: Reading files in chunks and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000, columns=original_columns):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

print(f"\nStep 2: Cleaning each group and writing to '{output_filename}'...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    # Combine chunks for one full group
    full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
    
    # Run the cleaning function
    cleaned_group_df = clean_one_group(full_group_df, original_columns)
    
    # Convert the cleaned pandas DataFrame to a pyarrow Table, ENFORCING the master schema
    table = pa.Table.from_pandas(cleaned_group_df, schema=master_schema)
    
    # If this is the first group, create the Parquet file and writer with the master schema
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_filename, master_schema)
    
    # Write the cleaned group's data to the file
    parquet_writer.write_table(table)

# Close the file writer after the loop is finished
if parquet_writer:
    parquet_writer.close()

print(f"\nProcessing complete. Final cleaned data saved to '{output_filename}'.")
