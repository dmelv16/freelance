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
        
        # Calculate steady state mean from a good sample
        # Use points after stabilization that meet steady criteria
        steady_sample_end = min(stabilizing_end + 100, energized_length)
        steady_sample = energized_voltage[stabilizing_end:steady_sample_end]
        steady_mean = np.mean(steady_sample[steady_sample >= STEADY_VOLTAGE_THRESHOLD])
        
        # Lock in steady state initially
        status[start_idx + stabilizing_end:end_idx + 1] = 'Steady State'
        
        # Now check for TRUE ramp-down (not just temporary dips)
        # Start checking from after initial steady state establishment
        check_start = stabilizing_end + 50  # Give steady state time to establish
        
        if check_start < energized_length - SUSTAINED_DROP_POINTS:
            for i in range(check_start, energized_length - SUSTAINED_DROP_POINTS):
                current_voltage = energized_voltage[i]
                
                # Check if voltage dropped significantly
                if steady_mean - current_voltage > RAMP_DOWN_THRESHOLD:
                    # This could be a dip or ramp-down - need to check
                    
                    # Look ahead to see if it recovers (temporary dip) or stays down (ramp-down)
                    look_ahead_window = min(DIP_RECOVERY_WINDOW, energized_length - i)
                    future_voltages = energized_voltage[i:i + look_ahead_window]
                    
                    # Check if voltage recovers
                    recovery_mask = future_voltages > (steady_mean - RAMP_DOWN_THRESHOLD * 0.5)
                    if np.sum(recovery_mask) > look_ahead_window * 0.3:
                        # It recovers - this is just a temporary dip, keep as steady state
                        continue
                    
                    # Check if drop is sustained
                    sustained_window = min(SUSTAINED_DROP_POINTS, energized_length - i)
                    sustained_voltages = energized_voltage[i:i + sustained_window]
                    
                    # If most points in the window are low, it's a true ramp-down
                    if np.mean(sustained_voltages) < steady_mean - RAMP_DOWN_THRESHOLD * 0.7:
                        # This is a true ramp-down
                        status[start_idx + i:end_idx + 1] = 'Stabilizing'
                        break
                    
                    # Also check for consistent downward trend
                    if sustained_window >= 5:
                        x = np.arange(sustained_window)
                        slope = np.polyfit(x, sustained_voltages, 1)[0]
                        if slope < -0.1:  # Consistent downward trend
                            status[start_idx + i:end_idx + 1] = 'Stabilizing'
                            break
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
