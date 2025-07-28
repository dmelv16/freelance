import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm

# --- 1. Define the Final Cleaning Function ---
def clean_one_group(group_df):
    
    TOLERANCE_PERCENT = 0.02
    df = group_df.copy() # Use a new variable name to avoid confusion

    # --- First Pass: Calculate dynamic threshold ---
    initial_steady_points = df[df['voltage_28v_dc1_cal'] >= 22]
    dynamic_threshold = 22.0 
    
    if not initial_steady_points.empty:
        group_mean_voltage = initial_steady_points['voltage_28v_dc1_cal'].mean()
        dynamic_threshold = group_mean_voltage * (1 - TOLERANCE_PERCENT)

    def initial_classify(voltage):
        if voltage < 2.2: return 'De-energized'
        elif voltage >= dynamic_threshold: return 'Steady State'
        else: return 'Stabilizing'
            
    df['Cleaned_Status'] = df['voltage_28v_dc1_cal'].apply(initial_classify)

    # --- Second Pass: Differentiate ramp-up, ramp-down, and dips ---
    steady_indices = df.index[df['Cleaned_Status'] == 'Steady State']
    
    if not steady_indices.empty:
        first_steady_index, last_steady_index = steady_indices[0], steady_indices[-1]
        is_stabilizing = df['Cleaned_Status'] == 'Stabilizing'
        is_between = (df.index > first_steady_index) & (df.index < last_steady_index)
        df.loc[is_stabilizing & is_between, 'Cleaned_Status'] = 'Steady State'

    # Overwrite the original status column with the cleaned values
    df['dc1_status'] = df['Cleaned_Status']
    
    # Return a DataFrame with the exact same columns as the original input
    return df[original_columns]

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

# --- THIS IS THE NEW LOGIC TO CREATE A UNIVERSAL SCHEMA ---
print("Scanning all files to create a universal master schema...")
all_columns = set()
for file_path in file_list:
    schema = pq.ParquetFile(file_path).schema_arrow
    for col_name in schema.names:
        all_columns.add(col_name)

# We can rebuild a master schema from a reference file and add missing fields
# For simplicity, we will ensure columns are present in each chunk later.
source_file = pq.ParquetFile(file_list[0])
master_schema = source_file.schema_arrow
master_columns = master_schema.names
# Find any columns that are not in our master file's schema
additional_columns = all_columns - set(master_columns)
# Note: This script assumes additional columns can be safely added.
# For a fully robust solution, you would merge schemas. Here we handle it at the chunk level.
universal_column_list = master_columns + list(additional_columns)


groups_data = defaultdict(list)
print("Step 1: Reading files in chunks and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        chunk = batch.to_pandas()
        # --- NEW: Ensure chunk has all universal columns ---
        for col in universal_column_list:
            if col not in chunk.columns:
                chunk[col] = None # Add missing columns with a null value
        
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

print(f"\nStep 2: Cleaning each group and writing to '{output_filename}'...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    # Combine chunks for one full group
    full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
    
    # Run the cleaning function
    cleaned_group_df = clean_one_group(full_group_df)
    
    # Ensure the cleaned group has all columns before creating the table
    for col in universal_column_list:
        if col not in cleaned_group_df.columns:
            cleaned_group_df[col] = None
    
    # Convert to pyarrow Table
    table = pa.Table.from_pandas(cleaned_group_df, preserve_index=False)
    
    # If this is the first group, create the Parquet file and writer
    if parquet_writer is None:
        # Use the schema from the first processed table as the definitive master schema
        master_schema = table.schema
        parquet_writer = pq.ParquetWriter(output_filename, master_schema)
    
    # Ensure the table schema matches the master before writing
    if not table.schema.equals(master_schema):
        table = table.cast(master_schema)
        
    parquet_writer.write_table(table)

# Close the file writer after the loop is finished
if parquet_writer:
    parquet_writer.close()

print(f"\nProcessing complete. Final cleaned data saved to '{output_filename}'.")

