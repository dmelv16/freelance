import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import os

# --- 1. CORE HELPER FUNCTION (With Dual-Condition Check) ---
# --- 1. CORRECTED: Core Helper Function ---
def classify_voltage_channel(voltage_series: pd.Series):
    """
    Analyzes a single series of voltage data and returns a series of status strings.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22
    STABILITY_WINDOW = 5
    STABILITY_THRESHOLD = 0.05
    STEADY_VOLTAGE_TOLERANCE = 1.0
    IMMEDIATE_DROP_TOLERANCE = 0.2

    temp_df = pd.DataFrame({'voltage': pd.to_numeric(voltage_series, errors='coerce')})
    temp_df['stability'] = temp_df['voltage'].rolling(window=STABILITY_WINDOW).std()

    # --- Pass 1 & 2: Initial Classification ---
    def classify_row(row):
        if pd.isna(row['voltage']) or row['voltage'] < 2.2:
            return 'De-energized'
        if row['voltage'] >= STEADY_VOLTAGE_THRESHOLD and not pd.isna(row['stability']) and row['stability'] < STABILITY_THRESHOLD:
            return 'Steady State'
        return 'Stabilizing'
    temp_df['status'] = temp_df.apply(classify_row, axis=1)

    # --- REVISED: Pass 3 - Iterative Correction for Dips and End-of-Sequence ---
    steady_indices = temp_df.index[temp_df['status'] == 'Steady State']
    if not steady_indices.empty:
        first_steady_index = steady_indices[0]
        
        # Get initial steady state block for calculating a running mean
        steady_mask = temp_df.index <= first_steady_index
        
        # Loop forward from the first steady point to handle all subsequent points
        for i in range(first_steady_index + 1, len(temp_df)):
            if temp_df.at[i, 'status'] == 'Stabilizing':
                # Update the running mean of the confirmed steady block so far
                mean_steady_voltage = temp_df.loc[steady_mask, 'voltage'].mean()
                
                current_voltage = temp_df.at[i, 'voltage']
                previous_voltage = temp_df.at[i - 1, 'voltage']

                # Perform the dual-condition check
                is_close_to_average = abs(current_voltage - mean_steady_voltage) < STEADY_VOLTAGE_TOLERANCE
                has_no_immediate_drop = current_voltage >= (previous_voltage - IMMEDIATE_DROP_TOLERANCE)

                if is_close_to_average and has_no_immediate_drop:
                    # If conditions are met, expand the steady block
                    temp_df.at[i, 'status'] = 'Steady State'
                    steady_mask.at[i] = True
                else:
                    # If conditions fail, this is a true ramp-down; stop extending the block
                    break
            elif temp_df.at[i, 'status'] == 'Steady State':
                 # If a point was already steady, ensure it's part of our mask
                 steady_mask.at[i] = True
    
    return temp_df['status']

# --- 2. ORCHESTRATOR FUNCTION (No changes needed) ---
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

# --- 3. Setup ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
output_directory = 'output_data_cleaned'
output_filename = os.path.join(output_directory, 'cleaned_data_dual_channel_fixed.parquet')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# --- 4. Read Files and Gather Groups ---
groups_data = defaultdict(list)
print("Step 1: Reading files and gathering all groups...")
for file_path in file_list:
    # Code to read files and gather groups... (Same as before)
    columns_to_read = list(set(['timestamp', 'voltage_28v_dc1_cal', 'voltage_28v_dc2_cal'] + group_cols))
    file_schema = pq.read_schema(file_path)
    final_columns = [col for col in columns_to_read if col in file_schema.names]
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000, columns=final_columns):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"✅ Finished gathering data for {len(groups_data)} unique groups.")

# --- 5. Define Output Schema ---
base_schema = pq.read_schema(file_list[0])
master_names = base_schema.names
if 'dc1_status' not in master_names: master_names.append('dc1_status')
if 'dc2_status' not in master_names: master_names.append('dc2_status')
fields = [f for f in base_schema if f.name in master_names]
if 'dc1_status' not in [f.name for f in fields]: fields.append(pa.field('dc1_status', pa.string()))
if 'dc2_status' not in [f.name for f in fields]: fields.append(pa.field('dc2_status', pa.string()))
master_schema = pa.schema(fields)

# --- 6. SIMPLIFIED: Clean each group and write to file iteratively ---
parquet_writer = None
print(f"\nStep 2: Cleaning each group and writing to '{output_filename}'...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    try:
        # SIMPLIFIED: No threshold lookup needed anymore
        full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
        cleaned_group_df = clean_dc_channels(full_group_df) # Call the simplified function
        
        for col_name in master_schema.names:
            if col_name not in cleaned_group_df.columns:
                cleaned_group_df[col_name] = None
        
        cleaned_group_df = cleaned_group_df[master_schema.names]
        table = pa.Table.from_pandas(cleaned_group_df, schema=master_schema, preserve_index=False)
        
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_filename, master_schema)
        
        parquet_writer.write_table(table)

    except Exception as e:
        print(f"\nWARNING: Skipping group {group_key} due to a processing error: {e}")
        continue

if parquet_writer:
    parquet_writer.close()
    print(f"\n🎉 Successfully finished writing to {output_filename}")
