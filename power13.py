import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
from tqdm import tqdm
import os

# ============================================================================
# PART 1: ENHANCED VOLTAGE CLASSIFICATION FUNCTIONS
# ============================================================================

def detect_elbow_points(voltage_data: np.ndarray, window_size=10):
    """
    Detects elbow points using curvature analysis.
    Works for both ramp-up and settle-down scenarios.
    """
    if len(voltage_data) < window_size * 3:
        return []
    
    # Smooth the signal first
    smoothed = uniform_filter1d(voltage_data, size=5, mode='nearest')
    
    # Calculate first and second derivatives
    first_deriv = np.gradient(smoothed)
    second_deriv = np.gradient(first_deriv)
    
    # Calculate curvature magnitude (absolute value to catch both directions)
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**1.5
    
    # Find peaks in curvature (these are potential elbows)
    peaks, properties = signal.find_peaks(curvature, 
                                         height=np.percentile(curvature, 75),
                                         distance=window_size)
    
    return peaks

def calculate_stability_score(segment: np.ndarray, window_size=5):
    """
    Calculate how stable a segment is using multiple metrics.
    Lower score = more stable.
    """
    if len(segment) < window_size:
        return float('inf')
    
    # Multiple stability metrics
    std_dev = np.std(segment)
    range_val = np.ptp(segment)  # peak-to-peak range
    
    # Rolling variance to detect local instabilities
    if len(segment) > window_size:
        rolling_std = pd.Series(segment).rolling(window_size, center=True).std()
        max_local_std = np.nanmax(rolling_std)
    else:
        max_local_std = std_dev
    
    # Trend component (penalize segments with strong trends)
    if len(segment) > 1:
        trend = np.abs(np.polyfit(np.arange(len(segment)), segment, 1)[0])
    else:
        trend = 0
    
    # Combined stability score (lower is better)
    stability_score = std_dev + 0.5 * range_val + 0.3 * max_local_std + 2 * trend
    return stability_score

def find_stable_region(voltage_data: np.ndarray, min_length=20):
    """
    Find the most stable continuous region in the voltage data.
    This helps identify the true steady-state period.
    """
    n = len(voltage_data)
    if n < min_length:
        return 0, n
    
    best_score = float('inf')
    best_start, best_end = 0, n
    
    # Use a sliding window to find the most stable region
    for window_size in [min_length, min_length * 2, min_length * 3]:
        if window_size > n:
            continue
            
        for i in range(n - window_size + 1):
            segment = voltage_data[i:i + window_size]
            score = calculate_stability_score(segment)
            
            if score < best_score:
                best_score = score
                best_start = i
                best_end = i + window_size
    
    return best_start, best_end

def classify_voltage_channel(voltage_series: pd.Series):
    """
    Enhanced classifier that accurately handles all stabilization patterns:
    1. Ramp-up to steady state
    2. Settle-down from high to steady state  
    3. Complex patterns with multiple transitions
    4. End-of-test ramp-downs
    """
    # --- Tunable Parameters ---
    MINIMUM_VOLTAGE = 2.2  # Below this is de-energized
    STEADY_VOLTAGE_THRESHOLD = 22.0  # Minimum for steady state consideration
    STABILITY_WINDOW = 15
    TREND_THRESHOLD = 0.02  # Max acceptable trend for steady state (V/sample)
    VARIANCE_MULTIPLIER = 3.5  # For tolerance bands
    SETTLE_THRESHOLD = 0.3  # Voltage difference to detect settling
    END_BUFFER_POINTS = 15
    CURVATURE_WINDOW = 10
    
    # --- Setup ---
    voltage = pd.to_numeric(voltage_series, errors='coerce').values
    n = len(voltage)
    status = np.full(n, 'De-energized', dtype=object)
    
    # Find energized region
    energized_mask = (voltage >= MINIMUM_VOLTAGE) & (~np.isnan(voltage))
    if not np.any(energized_mask):
        return pd.Series(status)
    
    energized_indices = np.where(energized_mask)[0]
    start_idx, end_idx = energized_indices[0], energized_indices[-1]
    energized_voltage = voltage[start_idx : end_idx + 1]
    energized_length = len(energized_voltage)
    
    if energized_length < 30:
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)
    
    # --- STEP 1: FIND THE MOST STABLE REGION ---
    stable_start, stable_end = find_stable_region(energized_voltage, min_length=20)
    
    if stable_start >= stable_end:
        # No stable region found
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)
    
    stable_voltage = energized_voltage[stable_start:stable_end]
    stable_mean = np.mean(stable_voltage)
    stable_std = max(np.std(stable_voltage), 0.01)
    
    # --- STEP 2: DETECT ELBOW POINTS ---
    elbow_points = detect_elbow_points(energized_voltage, window_size=CURVATURE_WINDOW)
    
    # --- STEP 3: DETERMINE STABILIZATION REGIONS ---
    # Initialize with the stable region as steady state
    local_status = np.full(energized_length, 'Stabilizing', dtype=object)
    local_status[stable_start:stable_end] = 'Steady State'
    
    # Check beginning of signal for stabilization
    if stable_start > 0:
        # Determine if it's ramping up or settling down
        early_voltage = energized_voltage[:min(20, stable_start)]
        early_mean = np.mean(early_voltage)
        
        if abs(early_mean - stable_mean) > SETTLE_THRESHOLD:
            # Significant difference - this is stabilizing
            # Find the exact transition point
            transition_found = False
            
            # Check if any detected elbow is in this region
            for elbow in elbow_points:
                if elbow < stable_start and elbow > 10:
                    local_status[:elbow] = 'Stabilizing'
                    local_status[elbow:stable_start] = 'Steady State' if calculate_stability_score(energized_voltage[elbow:stable_start]) < stable_std * 2 else 'Stabilizing'
                    transition_found = True
                    break
            
            if not transition_found:
                # Use tolerance band method to find transition
                tolerance = VARIANCE_MULTIPLIER * stable_std
                for i in range(min(stable_start, energized_length - STABILITY_WINDOW)):
                    window = energized_voltage[i : i + STABILITY_WINDOW]
                    window_mean = np.mean(window)
                    window_std = np.std(window)
                    
                    # Check if window is stable and close to steady state mean
                    if (abs(window_mean - stable_mean) < tolerance and 
                        window_std < stable_std * 2):
                        local_status[i:stable_start] = 'Steady State'
                        break
    
    # --- STEP 4: CHECK FOR SETTLE-DOWN PATTERN ---
    # Special handling for high-start settle-down (example: 27.62V → 27.55V)
    if energized_length > 50:
        initial_segment = energized_voltage[:30]
        middle_segment = energized_voltage[30:60] if energized_length > 60 else energized_voltage[30:]
        
        initial_mean = np.mean(initial_segment)
        middle_mean = np.mean(middle_segment)
        
        # Detect consistent downward trend in beginning
        if initial_mean > middle_mean + SETTLE_THRESHOLD:
            # Find where it actually settles
            for i in range(30, min(energized_length, 100)):
                # Use a sliding window to find stable point
                if i + STABILITY_WINDOW <= energized_length:
                    window = energized_voltage[i : i + STABILITY_WINDOW]
                    if calculate_stability_score(window) < stable_std * 1.5:
                        local_status[:i] = 'Stabilizing'
                        local_status[i:stable_end] = 'Steady State'
                        break
    
    # --- STEP 5: CHECK END OF SIGNAL FOR RAMP-DOWN ---
    if stable_end < energized_length:
        end_segment = energized_voltage[stable_end:]
        
        # Check for significant drop or trend
        if len(end_segment) > 5:
            end_mean = np.mean(end_segment)
            end_trend = np.polyfit(np.arange(len(end_segment)), end_segment, 1)[0]
            
            if (stable_mean - end_mean > SETTLE_THRESHOLD or 
                end_trend < -TREND_THRESHOLD):
                # This is a ramp-down
                local_status[stable_end:] = 'Stabilizing'
    
    # Special check for end buffer
    if energized_length > END_BUFFER_POINTS:
        end_buffer_start = energized_length - END_BUFFER_POINTS
        end_voltages = energized_voltage[end_buffer_start:]
        
        # Check if end shows instability
        end_stability = calculate_stability_score(end_voltages)
        if end_stability > stable_std * 3:
            # Find where instability starts
            for i in range(len(end_voltages)):
                if abs(end_voltages[i] - stable_mean) > stable_std * VARIANCE_MULTIPLIER:
                    local_status[end_buffer_start + i:] = 'Stabilizing'
                    break
    
    # --- STEP 6: HANDLE BRIEF DIPS AND SPIKES ---
    # Don't reclassify brief dips in steady state as stabilizing
    steady_indices = np.where(local_status == 'Steady State')[0]
    if len(steady_indices) > 0:
        # Group consecutive steady state regions
        groups = np.split(steady_indices, np.where(np.diff(steady_indices) != 1)[0] + 1)
        
        for group in groups:
            if len(group) < 10:  # Small isolated steady state region
                # Check if it should actually be stabilizing
                group_voltage = energized_voltage[group]
                if calculate_stability_score(group_voltage) > stable_std * 2:
                    local_status[group] = 'Stabilizing'
    
    # --- STEP 7: FINAL SMOOTHING ---
    # Remove very short state changes (noise reduction)
    for i in range(1, energized_length - 1):
        if local_status[i] != local_status[i-1] and local_status[i] != local_status[i+1]:
            local_status[i] = local_status[i-1]
    
    # Map local status back to full array
    status[start_idx : end_idx + 1] = local_status
    
    return pd.Series(status)

# ============================================================================
# PART 2: DATA PROCESSING PIPELINE
# ============================================================================

def clean_dc_channels(group_df):
    """
    Fast orchestrator for cleaning DC channels using enhanced classification.
    """
    df = group_df.copy()
    
    # Process DC1 with enhanced classifier
    df['dc1_status'] = classify_voltage_channel(df['voltage_28v_dc1_cal'])
    
    # Process DC2 if available with enhanced classifier
    if 'voltage_28v_dc2_cal' in df.columns:
        df['dc2_status'] = classify_voltage_channel(df['voltage_28v_dc2_cal'])
    else:
        df['dc2_status'] = 'De-energized'
    
    return df

def analyze_voltage_profile(voltage_series: pd.Series):
    """
    Provides detailed analysis of the voltage profile to help tune parameters.
    Useful for debugging and understanding your data patterns.
    """
    voltage = pd.to_numeric(voltage_series, errors='coerce').values
    voltage_clean = voltage[~np.isnan(voltage)]
    
    if len(voltage_clean) == 0:
        return "No valid voltage data"
    
    # Find stable region
    energized_mask = voltage_clean >= 2.2
    if not np.any(energized_mask):
        return "No energized data found"
    
    energized = voltage_clean[energized_mask]
    stable_start, stable_end = find_stable_region(energized)
    
    analysis = {
        'total_points': len(voltage_clean),
        'energized_points': len(energized),
        'min_voltage': np.min(energized),
        'max_voltage': np.max(energized),
        'mean_voltage': np.mean(energized),
        'std_voltage': np.std(energized),
        'stable_region': (stable_start, stable_end),
        'stable_mean': np.mean(energized[stable_start:stable_end]) if stable_start < stable_end else None,
        'stable_std': np.std(energized[stable_start:stable_end]) if stable_start < stable_end else None,
        'elbows_detected': len(detect_elbow_points(energized))
    }
    
    # Detect pattern type
    if len(energized) > 30:
        early_mean = np.mean(energized[:15])
        late_mean = np.mean(energized[-15:])
        middle_mean = np.mean(energized[len(energized)//2 - 10 : len(energized)//2 + 10])
        
        if early_mean < middle_mean - 0.5:
            analysis['pattern'] = 'Ramp-up'
        elif early_mean > middle_mean + 0.5:
            analysis['pattern'] = 'Settle-down'
        elif late_mean < middle_mean - 0.5:
            analysis['pattern'] = 'End ramp-down'
        else:
            analysis['pattern'] = 'Stable throughout'
    
    return analysis

# ============================================================================
# MAIN PROCESSING SCRIPT
# ============================================================================

def main():
    # --- Configuration ---
    file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']  # Update with your file paths
    group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
    output_directory = 'output_data_cleaned'
    exclusion_file_path = 'path/to/exclusion_list.csv'  # Update with your exclusion file path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    output_filename = os.path.join(output_directory, 'cleaned_data_dual_channel_enhanced.parquet')
    
    # --- Step 1: Load the Exclusion List and Create Fingerprints ---
    print(f"Step 1: Loading exclusion list from '{exclusion_file_path}'...")
    try:
        df_to_exclude = pd.read_csv(exclusion_file_path)
        
        # Create the unique fingerprint for each row in the exclusion file
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
    
    # --- Step 2: Read Files and Filter During Ingestion ---
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
                        first_row = group_df.iloc[0]
                        group_key = tuple(first_row[col] for col in group_cols)
    
                        if group_key not in groups_data:
                            total_groups_seen += 1
                            groups_kept += 1
    
                        groups_data[group_key].append(group_df.drop(columns=['fingerprint']))
                    else:
                        # This is a group we want to skip
                        first_row = group_df.iloc[0]
                        group_key = tuple(first_row[col] for col in group_cols)
                        if group_key not in groups_data:
                            total_groups_seen += 1
                            
        except Exception as e:
            print(f"Could not read {file_path}, error: {e}")
    
    print(f"✅ Scan complete. Seen {total_groups_seen} unique groups.")
    print(f"Filtered out {total_groups_seen - groups_kept} groups. {groups_kept} groups will be processed.")
    
    # --- Step 3: Define Output Schema ---
    base_schema = pq.read_schema(file_list[0])
    master_names = base_schema.names
    if 'dc1_status' not in master_names: 
        master_names.append('dc1_status')
    if 'dc2_status' not in master_names: 
        master_names.append('dc2_status')
    
    fields = [f for f in base_schema if f.name in master_names]
    if 'dc1_status' not in [f.name for f in fields]: 
        fields.append(pa.field('dc1_status', pa.string()))
    if 'dc2_status' not in [f.name for f in fields]: 
        fields.append(pa.field('dc2_status', pa.string()))
    
    master_schema = pa.schema(fields)
    
    # --- Step 4: Clean each group and write to file iteratively ---
    parquet_writer = None
    print(f"\nStep 3: Cleaning each group with enhanced classification and writing to '{output_filename}'...")
    
    # Optional: Add progress tracking for analysis
    analysis_results = []
    
    for group_key, list_of_chunks in tqdm(groups_data.items(), desc="Processing groups"):
        try:
            # Combine chunks for this group
            full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
            
            # Optional: Analyze voltage profiles before classification (for debugging)
            # Uncomment the following lines to collect analysis data
            # dc1_analysis = analyze_voltage_profile(full_group_df['voltage_28v_dc1_cal'])
            # analysis_results.append({'group': group_key, 'channel': 'DC1', 'analysis': dc1_analysis})
            
            # Clean the channels with enhanced classification
            cleaned_group_df = clean_dc_channels(full_group_df)
            
            # Ensure all required columns are present
            for col_name in master_schema.names:
                if col_name not in cleaned_group_df.columns:
                    cleaned_group_df[col_name] = None
            
            # Reorder columns to match schema
            cleaned_group_df = cleaned_group_df[master_schema.names]
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(cleaned_group_df, schema=master_schema, preserve_index=False)
            
            # Initialize writer on first iteration
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_filename, master_schema)
            
            # Write the table
            parquet_writer.write_table(table)
    
        except Exception as e:
            print(f"\nWARNING: Skipping group {group_key} due to a processing error: {e}")
            continue
    
    # Close the writer
    if parquet_writer:
        parquet_writer.close()
        print(f"\n🎉 Successfully finished writing to {output_filename}")
    
    # Optional: Save analysis results for review
    # if analysis_results:
    #     pd.DataFrame(analysis_results).to_csv('voltage_analysis_results.csv', index=False)
    #     print(f"Saved voltage profile analysis to 'voltage_analysis_results.csv'")

if __name__ == "__main__":
    main()
