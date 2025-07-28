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
output_filename = 'final_cleaned_data.parquet'
parquet_writer = None

# Define columns that MUST be numeric
numeric_cols = ['voltage_28v_dc1_cal', 'current_28v_dc1_cal', 'timestamp'] # Add any other known numeric cols

print("Step 1: Scanning all source files to build a master schema...")
all_fields = {}
for file_path in file_list:
    schema = pq.ParquetFile(file_path).schema_arrow
    for field in schema:
        if field.name not in all_fields:
            all_fields[field.name] = field
master_schema = pa.schema([field for name, field in sorted(all_fields.items())])
print("Master schema created successfully.")


# --- 3. Read Files, Harmonize, Sanitize, and Gather Groups ---
groups_data = defaultdict(list)
# Get the list of all column names in the correct order from our master schema
master_column_names = master_schema.names

print("\nStep 2: Reading files and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        # Convert the raw batch to pandas first
        chunk = batch.to_pandas()
        
        # --- FIX STARTS HERE ---
        # 1. Harmonize Schema: Add any columns from the master list that are missing in this chunk
        for col_name in master_column_names:
            if col_name not in chunk.columns:
                chunk[col_name] = None # Add missing column with nulls
        
        # 2. Ensure the column order is exactly the same as the master schema
        chunk = chunk[master_column_names]
        # --- FIX ENDS HERE ---

        # Sanitize numeric types to fix mixed-type issues
        for col in numeric_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

print(f"\nStep 3: Cleaning each group and writing to '{output_filename}'...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
    cleaned_group_df = clean_one_group(full_group_df)
    
    table = pa.Table.from_pandas(cleaned_group_df, schema=master_schema, preserve_index=False)
    
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_filename, master_schema)
    
    parquet_writer.write_table(table)

if parquet_writer:
    parquet_writer.close()

print(f"\nProcessing complete. Final cleaned data saved to '{output_filename}'.")
