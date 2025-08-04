import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

def classify_voltage_channel(voltage_series: pd.Series):
    """
    Fast classification that properly identifies the short stabilizing period
    and locks in steady state, while handling temporary dips without ending steady state.
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22
    STABILITY_WINDOW = 10  # Window for checking stability
    STABILITY_THRESHOLD = 0.1  # Max std dev for steady state
    STABILIZING_MAX_DURATION = 0.1  # Max 10% of test can be stabilizing
    
    # Ramp-down detection parameters
    RAMP_DOWN_THRESHOLD = 1.0  # Voltage drop to consider ramp-down
    DIP_RECOVERY_WINDOW = 20  # Points to check for recovery after dip
    SUSTAINED_DROP_POINTS = 10  # How many points must stay low for true ramp-down
    MIN_POINTS_FOR_RAMP_CHECK = 5  # Minimum points needed to confirm ramp-down
    END_BUFFER_POINTS = 10  # Points at end to check more carefully
    
    # Vectorized operations for speed
    voltage = pd.to_numeric(voltage_series, errors='coerce').values
    n = len(voltage)
    
    # Initialize status array
    status = np.full(n, 'De-energized', dtype=object)
    
    # Find energized indices (vectorized)
    energized_mask = (voltage >= 2.2) & (~np.isnan(voltage))
    
    if not np.any(energized_mask):
        return pd.Series(status)
    
    # Get energized boundaries
    energized_indices = np.where(energized_mask)[0]
    start_idx = energized_indices[0]
    end_idx = energized_indices[-1]
    
    # Work only with energized portion
    energized_voltage = voltage[start_idx:end_idx + 1]
    energized_length = len(energized_voltage)
    
    # Calculate rolling statistics (vectorized)
    rolling_std = pd.Series(energized_voltage).rolling(
        window=STABILITY_WINDOW, min_periods=1
    ).std().fillna(method='bfill').values
    
    # Find where voltage is stable (low std) and above threshold
    stable_mask = (energized_voltage >= STEADY_VOLTAGE_THRESHOLD) & (rolling_std < STABILITY_THRESHOLD)
    
    # Find first stable point
    stable_indices = np.where(stable_mask)[0]
    
    if len(stable_indices) > 0:
        first_stable = stable_indices[0]
        
        # Limit stabilizing period to max duration
        max_stabilizing = int(energized_length * STABILIZING_MAX_DURATION)
        stabilizing_end = min(first_stable, max_stabilizing)
        
        # Set stabilizing period (short initial period)
        if stabilizing_end > 0:
            status[start_idx:start_idx + stabilizing_end] = 'Stabilizing'
        
        # Calculate steady state characteristics
        steady_sample_end = min(stabilizing_end + 100, energized_length)
        steady_sample = energized_voltage[stabilizing_end:steady_sample_end]
        steady_mean = np.mean(steady_sample[steady_sample >= STEADY_VOLTAGE_THRESHOLD])
        steady_std = np.std(steady_sample[steady_sample >= STEADY_VOLTAGE_THRESHOLD])
        
        # Initially mark everything after stabilizing as steady state
        # (we'll revise this as needed)
        status[start_idx + stabilizing_end:end_idx + 1] = 'Steady State'
        
        # Special handling for the last END_BUFFER_POINTS points
        if energized_length > END_BUFFER_POINTS:
            end_buffer_start = energized_length - END_BUFFER_POINTS
            end_voltages = energized_voltage[end_buffer_start:]
            
            # Check if the end shows signs of ramp-down
            end_mean = np.mean(end_voltages)
            end_trend = np.polyfit(np.arange(len(end_voltages)), end_voltages, 1)[0]
            
            # Criteria for end ramp-down:
            # 1. Mean voltage dropped significantly
            # 2. OR consistent downward trend
            # 3. OR multiple points below threshold
            if (steady_mean - end_mean > RAMP_DOWN_THRESHOLD * 0.7 or 
                end_trend < -0.05 or
                np.sum(end_voltages < steady_mean - RAMP_DOWN_THRESHOLD) > len(end_voltages) * 0.5):
                
                # Find where the drop started in the end buffer
                for j in range(len(end_voltages)):
                    if steady_mean - end_voltages[j] > RAMP_DOWN_THRESHOLD * 0.5:
                        status[start_idx + end_buffer_start + j:end_idx + 1] = 'Stabilizing'
                        break
        
        # Now check for ramp-downs in the main body (not the end buffer)
        check_start = stabilizing_end + 50
        check_end = max(check_start, energized_length - END_BUFFER_POINTS)
        
        if check_start < check_end:
            for i in range(check_start, check_end):
                current_voltage = energized_voltage[i]
                
                # Check if voltage dropped significantly
                if steady_mean - current_voltage > RAMP_DOWN_THRESHOLD:
                    # Look ahead to see if it recovers
                    look_ahead_window = min(DIP_RECOVERY_WINDOW, check_end - i)
                    future_voltages = energized_voltage[i:i + look_ahead_window]
                    
                    # Check if voltage recovers
                    recovery_mask = future_voltages > (steady_mean - RAMP_DOWN_THRESHOLD * 0.5)
                    if np.sum(recovery_mask) > look_ahead_window * 0.3:
                        # It recovers - this is just a temporary dip
                        continue
                    
                    # Check if drop is sustained
                    sustained_window = min(SUSTAINED_DROP_POINTS, check_end - i)
                    sustained_voltages = energized_voltage[i:i + sustained_window]
                    
                    # If most points in the window are low, it's a true ramp-down
                    if np.mean(sustained_voltages) < steady_mean - RAMP_DOWN_THRESHOLD * 0.7:
                        status[start_idx + i:end_idx + 1] = 'Stabilizing'
                        break
                    
                    # Check for consistent downward trend
                    if sustained_window >= 5:
                        x = np.arange(sustained_window)
                        slope = np.polyfit(x, sustained_voltages, 1)[0]
                        if slope < -0.1:
                            status[start_idx + i:end_idx + 1] = 'Stabilizing'
                            break
        
        # Final check: if the very last point is significantly low, mark it as stabilizing
        if energized_voltage[-1] < steady_mean - RAMP_DOWN_THRESHOLD:
            # Find how far back the drop extends
            drop_start = energized_length - 1
            for j in range(energized_length - 1, max(stabilizing_end, energized_length - 20), -1):
                if energized_voltage[j] >= steady_mean - RAMP_DOWN_THRESHOLD * 0.5:
                    drop_start = j + 1
                    break
            status[start_idx + drop_start:end_idx + 1] = 'Stabilizing'
            
    else:
        # No stable points found - all stabilizing
        status[start_idx:end_idx + 1] = 'Stabilizing'
    
    return pd.Series(status)


def clean_dc_channels(group_df):
    """
    Fast orchestrator for cleaning DC channels.
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

# --- CORRECTED Step 1: Load the Exclusion List and Create Fingerprints ---
print(f"Step 1: Loading exclusion list from '{exclusion_file_path}'...")
try:
    df_to_exclude = pd.read_csv(exclusion_file_path)
    
    # Create the unique fingerprint for each row in the exclusion file
    # This is done by converting every value to a string and joining them with a separator
    df_to_exclude['fingerprint'] = df_to_exclude[group_cols].astype(str).agg('|'.join, axis=1)
    
    # Convert the fingerprints into a set for very fast lookups
    groups_to_remove_fingerprints = set(df_to_exclude['fingerprint'])
    
    print(f"Loaded {len(groups_to_remove_fingerprints)} unique group fingerprints to exclude.")
except FileNotFoundError:
    print(f"Warning: Exclusion file not found at '{exclusion_file_path}'. No groups will be excluded.")
    groups_to_remove_fingerprints = set()
except Exception as e:
    print(f"Error loading exclusion file: {e}. No groups will be excluded.")
    groups_to_remove_fingerprints = set()


# --- CORRECTED Step 2: Read Files and Filter During Ingestion ---
groups_data = defaultdict(list)
print(f"\nStep 2: Reading files and filtering groups during ingestion...")
total_groups_seen = 0
groups_kept = 0
for file_path in file_list:
    columns_to_read = list(set(['timestamp', 'voltage_28v_dc1_cal', 'voltage_28v_dc2_cal'] + group_cols))
    try:
        file_schema = pq.read_schema(file_path)
        final_columns = [col for col in columns_to_read if col in file_schema.names]
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(batch_size=500_000, columns=final_columns):
            chunk = batch.to_pandas()
            
            if chunk.empty:
                continue

            # Create a temporary fingerprint column in the data being read
            chunk['fingerprint'] = chunk[group_cols].astype(str).agg('|'.join, axis=1)

            for group_fingerprint, group_df in chunk.groupby('fingerprint'):
                # THE FILTERING HAPPENS HERE, using the fingerprint
                if group_fingerprint not in groups_to_remove_fingerprints:
                    # We need the original tuple key for our dictionary
                    # Get the first row of the group to reconstruct the key
                    first_row = group_df.iloc[0]
                    group_key = tuple(first_row[col] for col in group_cols)

                    if group_key not in groups_data:
                        total_groups_seen +=1
                        groups_kept += 1

                    groups_data[group_key].append(group_df.drop(columns=['fingerprint']))
                else:
                    # This is a group we want to skip. We find its original key to count it.
                     first_row = group_df.iloc[0]
                     group_key = tuple(first_row[col] for col in group_cols)
                     if group_key not in groups_data:
                         total_groups_seen +=1
                         
    except Exception as e:
         print(f"Could not read {file_path}, error: {e}")

print(f"✅ Scan complete. Seen {total_groups_seen} unique groups.")
print(f"Filtered out {total_groups_seen - groups_kept} groups. {groups_kept} groups will be processed.")


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
