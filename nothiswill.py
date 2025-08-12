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

def find_variance_transition(voltage_data: np.ndarray, window_size=20):
    """
    Find where variance stabilizes - the transition from high to low variance.
    This is often the best indicator of the stabilizing->steady transition.
    """
    if len(voltage_data) < window_size * 2:
        return None
    
    # Calculate rolling variance
    variances = []
    for i in range(len(voltage_data) - window_size):
        window = voltage_data[i:i+window_size]
        variances.append(np.var(window))
    
    variances = np.array(variances)
    
    # Find where variance drops and stays low
    median_var = np.median(variances)
    low_var_threshold = median_var * 0.5  # Points with variance below this are "stable"
    
    # Find the first sustained low-variance region
    consecutive_low = 0
    min_consecutive = window_size // 2  # Need at least this many consecutive low-variance windows
    
    for i in range(len(variances)):
        if variances[i] < low_var_threshold:
            consecutive_low += 1
            if consecutive_low >= min_consecutive:
                # Found stable region, return where it started
                return i - consecutive_low + window_size // 2
        else:
            consecutive_low = 0
    
    return None

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
    Simplified classifier focused on variance-based detection.
    Finds where data transitions from variable to stable.
    """
    # --- Parameters ---
    MINIMUM_VOLTAGE = 2.2  
    VARIANCE_WINDOW = 20  # Window for variance calculation
    STABILITY_RATIO = 0.3  # Variance must drop to this ratio of initial variance
    MIN_STABILIZING_POINTS = 15  # Minimum points to classify as stabilizing
    END_CHECK_WINDOW = 30  # Check last N points for end destabilization
    
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
    
    # Too short - all stabilizing
    if energized_length < VARIANCE_WINDOW * 2:
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)
    
    # --- MAIN ALGORITHM: Find where variance drops and stays low ---
    
    # Step 1: Calculate rolling variance
    variances = []
    for i in range(energized_length - VARIANCE_WINDOW + 1):
        window = energized_voltage[i:i+VARIANCE_WINDOW]
        variances.append(np.var(window))
    
    if len(variances) == 0:
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)
    
    variances = np.array(variances)
    
    # Step 2: Find the baseline stable variance (from the most stable part)
    # Use the 25th percentile of variances as our "stable" benchmark
    stable_variance = np.percentile(variances, 25)
    
    # Step 3: Find transition point where variance drops below threshold
    # We're looking for where variance becomes consistently low
    variance_threshold = stable_variance * 3  # Allow 3x the stable variance
    
    # Initialize all as stabilizing
    local_status = np.full(energized_length, 'Stabilizing', dtype=object)
    
    # Find the first point where variance drops and stays low
    transition_found = False
    consecutive_stable = 0
    required_consecutive = 5  # Need 5 consecutive low-variance windows
    
    for i in range(len(variances)):
        if variances[i] <= variance_threshold:
            consecutive_stable += 1
            if consecutive_stable >= required_consecutive and not transition_found:
                # Found the transition point
                transition_point = i  # This corresponds to the start of the window
                transition_found = True
                # Mark everything from here as steady state
                local_status[transition_point:] = 'Steady State'
                break
        else:
            consecutive_stable = 0
    
    # Step 4: Refine the transition for specific patterns
    
    # Check for settle-down pattern (starts high, decreases to steady)
    if transition_found and transition_point > MIN_STABILIZING_POINTS:
        # Look at the slope in the beginning
        early_segment = energized_voltage[:transition_point]
        if len(early_segment) > 10:
            early_slope = np.polyfit(np.arange(len(early_segment)), early_segment, 1)[0]
            
            # If clearly decreasing, find where it actually levels off
            if early_slope < -0.01:  # Significant downward slope
                # Find where the decrease stops
                for i in range(min(transition_point, energized_length - 10)):
                    window = energized_voltage[i:i+10]
                    window_slope = np.polyfit(np.arange(10), window, 1)[0]
                    
                    # Slope has flattened
                    if abs(window_slope) < 0.005:
                        local_status[:i] = 'Stabilizing'
                        local_status[i:] = 'Steady State'
                        transition_point = i
                        break
    
    # Check for ramp-up pattern (starts low, increases to steady)
    if transition_found and transition_point > MIN_STABILIZING_POINTS:
        early_segment = energized_voltage[:transition_point]
        if len(early_segment) > 10:
            early_slope = np.polyfit(np.arange(len(early_segment)), early_segment, 1)[0]
            
            # If clearly increasing, find where it levels off
            if early_slope > 0.01:  # Significant upward slope
                # Find where the increase stops
                for i in range(min(transition_point, energized_length - 10)):
                    window = energized_voltage[i:i+10]
                    window_slope = np.polyfit(np.arange(10), window, 1)[0]
                    
                    # Slope has flattened
                    if abs(window_slope) < 0.005:
                        local_status[:i] = 'Stabilizing'
                        local_status[i:] = 'Steady State'
                        transition_point = i
                        break
    
    # Step 5: Check the end for destabilization
    if energized_length > END_CHECK_WINDOW and transition_found:
        end_segment = energized_voltage[-END_CHECK_WINDOW:]
        end_variance = np.var(end_segment)
        
        # Also check for significant voltage drop at the end
        middle_voltage = np.mean(energized_voltage[energized_length//2:energized_length//2+20])
        end_voltage = np.mean(end_segment)
        
        # If end variance is high OR voltage dropped significantly
        if end_variance > variance_threshold * 2 or (middle_voltage - end_voltage) > 0.5:
            # Find where the destabilization starts
            for i in range(energized_length - END_CHECK_WINDOW, energized_length):
                remaining = energized_voltage[i:]
                if len(remaining) > 5:
                    remaining_var = np.var(remaining)
                    if remaining_var > variance_threshold * 1.5:
                        local_status[i:] = 'Stabilizing'
                        break
    
    # Step 6: Handle edge cases
    
    # If no transition was found but variance is generally low, it might all be steady
    if not transition_found:
        overall_variance = np.var(energized_voltage)
        if overall_variance < stable_variance * 2:
            # Low variance throughout - probably all steady state
            local_status[:] = 'Steady State'
        else:
            # High variance throughout - probably all stabilizing
            local_status[:] = 'Stabilizing'
    
    # Ensure minimum stabilizing period at the start unless it's already stable
    if transition_found and transition_point < MIN_STABILIZING_POINTS:
        # Check if the beginning is actually already stable
        initial_variance = np.var(energized_voltage[:MIN_STABILIZING_POINTS])
        if initial_variance > stable_variance * 3:
            # High initial variance, enforce minimum stabilizing period
            local_status[:MIN_STABILIZING_POINTS] = 'Stabilizing'
            local_status[MIN_STABILIZING_POINTS:] = 'Steady State'
    
    # Map back to full array
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
