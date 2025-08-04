import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import os

def classify_voltage_channel(voltage_series: pd.Series):
    """
    Analyzes voltage data using an enhanced "Lock-In" model for the core steady state,
    with improved ramp-down detection using dynamic thresholds and buffer tolerance.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22
    STABILITY_WINDOW = 5
    STABILITY_THRESHOLD = 0.05
    MINIMUM_STEADY_LENGTH = 10  # Minimum samples for a valid steady state region
    FAILURE_TOLERANCE = 3  # Consecutive failures allowed before ending steady state
    RAMP_THRESHOLD = 0.05  # Threshold for gradient-based ramp detection
    
    # Base tolerances (will be adjusted dynamically)
    BASE_STEADY_VOLTAGE_TOLERANCE = 1.0
    BASE_IMMEDIATE_DROP_TOLERANCE = 0.1
    
    temp_df = pd.DataFrame({'voltage': pd.to_numeric(voltage_series, errors='coerce')})
    
    # --- Calculate additional metrics ---
    temp_df['stability'] = temp_df['voltage'].rolling(window=STABILITY_WINDOW).std()
    temp_df['voltage_gradient'] = temp_df['voltage'].diff()
    temp_df['smoothed_gradient'] = temp_df['voltage_gradient'].rolling(window=3, center=True).mean()
    
    # --- Pass 1: Initial Classification ---
    def classify_row(row):
        if pd.isna(row['voltage']) or row['voltage'] < 2.2:
            return 'De-energized'
        if row['voltage'] >= STEADY_VOLTAGE_THRESHOLD and not pd.isna(row['stability']) and row['stability'] < STABILITY_THRESHOLD:
            return 'Steady State'
        return 'Stabilizing'
    
    temp_df['status'] = temp_df.apply(classify_row, axis=1)
    
    # --- Pass 2: Find all potential steady state regions ---
    steady_mask = temp_df['status'] == 'Steady State'
    steady_groups = (steady_mask != steady_mask.shift()).cumsum()
    steady_regions = []
    
    for group_id in steady_groups[steady_mask].unique():
        region_indices = temp_df.index[steady_groups == group_id]
        if len(region_indices) >= MINIMUM_STEADY_LENGTH:
            steady_regions.append((region_indices[0], region_indices[-1]))
    
    if not steady_regions:
        return temp_df['status']
    
    # --- Pass 3: Lock-in the largest/most significant steady state region ---
    # Choose the longest steady state region as the primary one
    primary_region = max(steady_regions, key=lambda x: x[1] - x[0])
    first_steady_index, last_steady_index = primary_region
    
    # Lock-in the core period
    temp_df.loc[first_steady_index:last_steady_index, 'status'] = 'Steady State'
    
    # --- Calculate dynamic thresholds based on steady state characteristics ---
    steady_voltage = temp_df.loc[first_steady_index:last_steady_index, 'voltage']
    mean_steady_voltage = steady_voltage.mean()
    steady_noise = steady_voltage.std()
    
    # Adjust tolerances based on noise level (3-sigma rule)
    STEADY_VOLTAGE_TOLERANCE = max(BASE_STEADY_VOLTAGE_TOLERANCE, 3 * steady_noise)
    IMMEDIATE_DROP_TOLERANCE = max(BASE_IMMEDIATE_DROP_TOLERANCE, steady_noise)
    
    # --- Pass 4: Enhanced Ramp-Down Detection with Buffer ---
    consecutive_failures = 0
    
    for i in range(last_steady_index + 1, len(temp_df)):
        if temp_df.at[i, 'status'] == 'Stabilizing':
            current_voltage = temp_df.at[i, 'voltage']
            previous_voltage = temp_df.at[i - 1, 'voltage']
            
            # Multi-condition check
            is_close_to_average = abs(current_voltage - mean_steady_voltage) < STEADY_VOLTAGE_TOLERANCE
            has_no_immediate_drop = current_voltage >= (previous_voltage - IMMEDIATE_DROP_TOLERANCE)
            
            # Additional gradient check
            smoothed_gradient = temp_df.at[i, 'smoothed_gradient'] if not pd.isna(temp_df.at[i, 'smoothed_gradient']) else 0
            is_not_ramping_down = smoothed_gradient > -RAMP_THRESHOLD
            
            # Combined condition
            if is_close_to_average and has_no_immediate_drop and is_not_ramping_down:
                temp_df.at[i, 'status'] = 'Steady State'
                consecutive_failures = 0  # Reset counter
            else:
                consecutive_failures += 1
                if consecutive_failures >= FAILURE_TOLERANCE:
                    # We've found the start of the ramp-down
                    # Mark remaining points as 'Stabilizing'
                    for j in range(i, len(temp_df)):
                        if temp_df.at[j, 'voltage'] >= 2.2:
                            temp_df.at[j, 'status'] = 'Stabilizing'
                    break
        else:
            # If we encounter 'De-energized', stop
            break
    
    # --- Pass 5: Validation and cleanup ---
    temp_df['status'] = validate_and_cleanup_classification(temp_df)
    
    return temp_df['status']


def validate_and_cleanup_classification(df):
    """
    Validates the classification and performs cleanup to ensure logical consistency.
    """
    status = df['status'].copy()
    
    # Rule 1: Remove isolated steady state points (noise)
    for i in range(1, len(status) - 1):
        if status.iloc[i] == 'Steady State':
            if status.iloc[i-1] != 'Steady State' and status.iloc[i+1] != 'Steady State':
                status.iloc[i] = 'Stabilizing'
    
    # Rule 2: Ensure minimum steady state duration
    MIN_STEADY_DURATION = 5
    steady_start = None
    
    for i in range(len(status)):
        if status.iloc[i] == 'Steady State':
            if steady_start is None:
                steady_start = i
        else:
            if steady_start is not None:
                if i - steady_start < MIN_STEADY_DURATION:
                    # Too short, convert back to stabilizing
                    status.iloc[steady_start:i] = 'Stabilizing'
                steady_start = None
    
    # Handle the case where steady state extends to the end
    if steady_start is not None and len(status) - steady_start < MIN_STEADY_DURATION:
        status.iloc[steady_start:] = 'Stabilizing'
    
    return status


def clean_dc_channels(group_df):
    """
    Orchestrates the cleaning of both DC1 and DC2 channels using the enhanced hybrid model.
    """
    df = group_df.copy()
    
    # Process DC1
    df['dc1_status'] = classify_voltage_channel(df['voltage_28v_dc1_cal'])
    
    # Process DC2 if available
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
