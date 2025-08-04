import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import os

import pandas as pd
import numpy as np
from scipy import signal

def classify_voltage_channel(voltage_series: pd.Series):
    """
    Analyzes voltage data by finding the true steady state region and locking it in.
    Handles both cases: with clear ramp-up elbow and without (already stable).
    """
    # --- Tunable Parameters ---
    STEADY_VOLTAGE_THRESHOLD = 22
    WINDOW_SIZE = 20  # Window for calculating statistics
    ELBOW_DETECTION_WINDOW = 50  # Window for detecting the ramp-up elbow
    VARIANCE_RATIO_THRESHOLD = 0.3  # Threshold for detecting stabilization
    MIN_STEADY_STATE_LENGTH = 30  # Minimum length for steady state region
    
    temp_df = pd.DataFrame({'voltage': pd.to_numeric(voltage_series, errors='coerce')})
    
    # Initialize all as De-energized
    temp_df['status'] = 'De-energized'
    
    # Find where voltage is actually present
    energized_mask = (temp_df['voltage'] >= 2.2) & (~temp_df['voltage'].isna())
    energized_indices = temp_df.index[energized_mask]
    
    if len(energized_indices) == 0:
        return temp_df['status']
    
    # Work only with energized portion
    start_idx = energized_indices[0]
    end_idx = energized_indices[-1]
    
    # Extract energized voltage data
    energized_voltage = temp_df.loc[start_idx:end_idx, 'voltage'].values
    
    # --- Step 1: Find the overall characteristics ---
    overall_mean = np.nanmean(energized_voltage)
    overall_std = np.nanstd(energized_voltage)
    
    # --- Step 2: Detect if there's a ramp-up phase (elbow detection) ---
    elbow_idx = detect_elbow(energized_voltage, ELBOW_DETECTION_WINDOW)
    
    if elbow_idx is None:
        # No clear elbow - might already be in steady state from the beginning
        stabilizing_end = find_stabilization_point(energized_voltage, WINDOW_SIZE)
    else:
        stabilizing_end = elbow_idx
    
    # --- Step 3: Find the steady state region ---
    # Look for the most stable continuous region after stabilization
    if stabilizing_end < len(energized_voltage) - MIN_STEADY_STATE_LENGTH:
        steady_start, steady_end = find_steady_state_region(
            energized_voltage[stabilizing_end:], 
            MIN_STEADY_STATE_LENGTH,
            overall_mean,
            overall_std
        )
        
        if steady_start is not None:
            # Adjust indices relative to original data
            steady_start += stabilizing_end
            steady_end += stabilizing_end
            
            # --- Step 4: Lock in the classifications ---
            # Everything before steady state is stabilizing
            temp_df.loc[start_idx:start_idx + steady_start - 1, 'status'] = 'Stabilizing'
            
            # Lock in steady state
            temp_df.loc[start_idx + steady_start:start_idx + steady_end, 'status'] = 'Steady State'
            
            # --- Step 5: Check for ramp-down after steady state ---
            if steady_end < len(energized_voltage) - 1:
                # Analyze remaining data for ramp-down
                remaining_voltage = energized_voltage[steady_end + 1:]
                steady_mean = np.mean(energized_voltage[steady_start:steady_end + 1])
                
                # Check if voltage drops significantly
                is_ramp_down = check_ramp_down(remaining_voltage, steady_mean, overall_std)
                
                if is_ramp_down:
                    temp_df.loc[start_idx + steady_end + 1:end_idx, 'status'] = 'Stabilizing'
                else:
                    # Extend steady state to the end
                    temp_df.loc[start_idx + steady_end + 1:end_idx, 'status'] = 'Steady State'
        else:
            # No clear steady state found - all is stabilizing
            temp_df.loc[start_idx:end_idx, 'status'] = 'Stabilizing'
    else:
        # Not enough data after stabilization
        temp_df.loc[start_idx:end_idx, 'status'] = 'Stabilizing'
    
    return temp_df['status']


def detect_elbow(voltage_data, window_size):
    """
    Detects the elbow point where voltage transitions from ramp-up to stable.
    Returns None if no clear elbow is found (already stable from start).
    """
    if len(voltage_data) < window_size * 2:
        return None
    
    # Calculate rolling statistics
    rolling_mean = pd.Series(voltage_data).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(voltage_data).rolling(window=window_size, center=True).std()
    
    # Calculate rate of change in mean
    mean_gradient = np.gradient(rolling_mean.fillna(method='bfill').fillna(method='ffill'))
    
    # Look for where gradient drops significantly
    gradient_threshold = np.nanstd(mean_gradient) * 0.5
    
    # Find where system stabilizes (low gradient and low variance)
    stable_mask = (np.abs(mean_gradient) < gradient_threshold) & (rolling_std < rolling_std.quantile(0.3))
    
    # Find first substantial stable region
    stable_groups = (stable_mask != stable_mask.shift()).cumsum()
    
    for group in stable_groups[stable_mask].unique():
        group_indices = stable_groups[stable_groups == group].index
        if len(group_indices) >= window_size // 2:
            return group_indices[0]
    
    return None


def find_stabilization_point(voltage_data, window_size):
    """
    Finds where the signal stabilizes by analyzing variance patterns.
    """
    if len(voltage_data) < window_size * 2:
        return 0
    
    # Calculate rolling variance
    rolling_var = pd.Series(voltage_data).rolling(window=window_size).var()
    
    # Find where variance stabilizes
    var_gradient = np.gradient(rolling_var.fillna(method='bfill'))
    
    # Look for where variance stops decreasing significantly
    for i in range(window_size, len(voltage_data) - window_size):
        if np.mean(np.abs(var_gradient[i:i+window_size])) < rolling_var.std() * 0.1:
            return i
    
    return window_size


def find_steady_state_region(voltage_data, min_length, overall_mean, overall_std):
    """
    Finds the most stable continuous region that represents steady state.
    """
    best_start = None
    best_end = None
    best_score = float('inf')
    
    # Scan for stable regions
    for start in range(0, len(voltage_data) - min_length):
        for end in range(start + min_length, min(start + len(voltage_data), len(voltage_data))):
            segment = voltage_data[start:end]
            
            # Calculate stability metrics
            segment_std = np.std(segment)
            segment_mean = np.mean(segment)
            
            # Check if this segment is around the expected steady state voltage
            if abs(segment_mean - overall_mean) > overall_std * 2:
                continue
            
            # Score based on stability (lower is better)
            score = segment_std / overall_std
            
            # Additional penalty for being too far from overall mean
            mean_penalty = abs(segment_mean - overall_mean) / overall_mean
            score += mean_penalty
            
            if score < best_score:
                best_score = score
                best_start = start
                best_end = end - 1
    
    # Verify the best region is actually stable enough
    if best_start is not None:
        segment = voltage_data[best_start:best_end + 1]
        if np.std(segment) < overall_std * 0.5:  # Must be more stable than overall
            return best_start, best_end
    
    return None, None


def check_ramp_down(remaining_voltage, steady_mean, overall_std):
    """
    Checks if the remaining voltage represents a ramp-down from steady state.
    """
    if len(remaining_voltage) < 5:
        return False
    
    # Check for consistent drop
    mean_remaining = np.mean(remaining_voltage)
    
    # Significant drop from steady state
    if steady_mean - mean_remaining > overall_std:
        return True
    
    # Check for negative trend
    x = np.arange(len(remaining_voltage))
    coeffs = np.polyfit(x, remaining_voltage, 1)
    slope = coeffs[0]
    
    # Negative slope indicates ramp-down
    if slope < -overall_std / len(remaining_voltage):
        return True
    
    return False


def clean_dc_channels(group_df):
    """
    Orchestrates the cleaning of both DC1 and DC2 channels.
    This version properly locks in steady state regions without intermingling.
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
