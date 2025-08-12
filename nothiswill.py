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
    Enhanced elbow detection using multiple methods for better accuracy.
    """
    if len(voltage_data) < window_size * 3:
        return []
    
    elbows = []
    
    # Method 1: Curvature analysis (original)
    smoothed = uniform_filter1d(voltage_data, size=5, mode='nearest')
    first_deriv = np.gradient(smoothed)
    second_deriv = np.gradient(first_deriv)
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**1.5
    
    # More selective peak detection
    curvature_threshold = np.percentile(curvature[curvature > 0], 80)
    peaks1, _ = signal.find_peaks(curvature, 
                                  height=curvature_threshold,
                                  distance=window_size,
                                  prominence=curvature_threshold/2)
    elbows.extend(peaks1)
    
    # Method 2: Slope change detection
    # Look for points where the rate of change significantly shifts
    slope_window = min(15, len(voltage_data) // 10)
    if len(voltage_data) > slope_window * 2:
        slopes = []
        for i in range(len(voltage_data) - slope_window):
            segment = voltage_data[i:i+slope_window]
            slope = np.polyfit(np.arange(len(segment)), segment, 1)[0]
            slopes.append(slope)
        
        slopes = np.array(slopes)
        slope_change = np.abs(np.diff(slopes))
        
        # Find significant slope changes
        slope_threshold = np.percentile(slope_change[slope_change > 0], 70)
        peaks2, _ = signal.find_peaks(slope_change,
                                      height=slope_threshold,
                                      distance=window_size)
        elbows.extend(peaks2 + slope_window // 2)  # Adjust for window offset
    
    # Method 3: Variance change detection
    # Look for transitions between high and low variance regions
    var_window = min(20, len(voltage_data) // 8)
    if len(voltage_data) > var_window * 2:
        variances = []
        for i in range(len(voltage_data) - var_window):
            segment = voltage_data[i:i+var_window]
            variances.append(np.var(segment))
        
        variances = np.array(variances)
        var_change = np.abs(np.diff(variances))
        
        # Find significant variance changes
        if np.any(var_change > 0):
            var_threshold = np.percentile(var_change[var_change > 0], 60)
            peaks3, _ = signal.find_peaks(var_change,
                                          height=var_threshold,
                                          distance=window_size)
            elbows.extend(peaks3 + var_window // 2)
    
    # Remove duplicates and sort
    elbows = list(set(elbows))
    elbows.sort()
    
    # Filter out elbows that are too close to each other
    filtered_elbows = []
    for elbow in elbows:
        if not filtered_elbows or elbow - filtered_elbows[-1] > window_size:
            filtered_elbows.append(elbow)
    
    return filtered_elbows

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
    Enhanced classifier with better elbow precision and noise tolerance.
    """
    # --- Tunable Parameters (ADJUSTED FOR BETTER ACCURACY) ---
    MINIMUM_VOLTAGE = 2.2  
    STEADY_VOLTAGE_THRESHOLD = 22.0  
    STABILITY_WINDOW = 15
    TREND_THRESHOLD = 0.015  # Reduced for more tolerance
    VARIANCE_MULTIPLIER = 4.0  # Increased for more tolerance to minor fluctuations
    SETTLE_THRESHOLD = 0.2  # More sensitive to subtle settling
    END_BUFFER_POINTS = 15
    CURVATURE_WINDOW = 10
    NOISE_TOLERANCE = 0.05  # New: tolerance for minor fluctuations
    MIN_STABLE_LENGTH = 30  # Minimum length to consider as steady state
    
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
        status[start_idx : end_idx + 1] = 'Stabilizing'
        return pd.Series(status)
    
    stable_voltage = energized_voltage[stable_start:stable_end]
    stable_mean = np.mean(stable_voltage)
    stable_std = max(np.std(stable_voltage), 0.01)
    
    # Calculate noise floor - the typical variation in the stable region
    noise_floor = stable_std * 2  # Adaptive noise threshold
    
    # --- STEP 2: DETECT ELBOW POINTS WITH ENHANCED METHOD ---
    elbow_points = detect_elbow_points(energized_voltage, window_size=CURVATURE_WINDOW)
    
    # --- STEP 3: INITIAL CLASSIFICATION ---
    local_status = np.full(energized_length, 'Stabilizing', dtype=object)
    local_status[stable_start:stable_end] = 'Steady State'
    
    # --- STEP 4: REFINE BEGINNING TRANSITION ---
    if stable_start > 0:
        early_voltage = energized_voltage[:min(30, stable_start)]
        early_mean = np.mean(early_voltage)
        
        # Check for settling pattern (high to low)
        if early_mean > stable_mean + SETTLE_THRESHOLD:
            # This is a settle-down pattern
            best_transition = 0
            
            # Find the best transition point
            for i in range(min(stable_start, energized_length - STABILITY_WINDOW)):
                window = energized_voltage[i : i + STABILITY_WINDOW]
                window_mean = np.mean(window)
                window_std = np.std(window)
                
                # Check if we've reached steady state
                if (abs(window_mean - stable_mean) < noise_floor and 
                    window_std < noise_floor):
                    best_transition = i
                    break
            
            # Check if any elbow point is better
            for elbow in elbow_points:
                if 0 < elbow < stable_start:
                    # Verify this elbow is actually the transition
                    before_elbow = energized_voltage[max(0, elbow-10):elbow]
                    after_elbow = energized_voltage[elbow:min(elbow+10, energized_length)]
                    
                    if (len(before_elbow) > 0 and len(after_elbow) > 0):
                        before_mean = np.mean(before_elbow)
                        after_mean = np.mean(after_elbow)
                        
                        # Significant change at elbow
                        if abs(before_mean - after_mean) > NOISE_TOLERANCE:
                            best_transition = elbow
            
            local_status[:best_transition] = 'Stabilizing'
            local_status[best_transition:stable_end] = 'Steady State'
            
        # Check for ramp-up pattern (low to high)
        elif early_mean < stable_mean - SETTLE_THRESHOLD:
            # This is a ramp-up pattern
            best_transition = stable_start
            
            # Find where voltage reaches near steady state
            for i in range(stable_start):
                if energized_voltage[i] >= stable_mean - noise_floor:
                    best_transition = i
                    break
            
            # Check elbows for better transition
            for elbow in elbow_points:
                if 0 < elbow < stable_start:
                    if energized_voltage[elbow] >= stable_mean - noise_floor * 1.5:
                        best_transition = elbow
                        break
            
            local_status[:best_transition] = 'Stabilizing'
            local_status[best_transition:stable_end] = 'Steady State'
    
    # --- STEP 5: PRECISE END-OF-SIGNAL CHECK ---
    if stable_end < energized_length - END_BUFFER_POINTS:
        end_segment = energized_voltage[stable_end:]
        
        if len(end_segment) > 10:
            end_mean = np.mean(end_segment)
            end_std = np.std(end_segment)
            
            # Only mark as stabilizing if there's significant change
            if (abs(stable_mean - end_mean) > noise_floor or 
                end_std > noise_floor * 2):
                
                # Find where the change begins
                for i in range(stable_end, energized_length):
                    if abs(energized_voltage[i] - stable_mean) > noise_floor:
                        local_status[i:] = 'Stabilizing'
                        break
    
    # --- STEP 6: SMOOTH OUT NOISE ---
    # Prevent over-classification of minor fluctuations
    # Use a sliding window to smooth classifications
    smooth_window = 10
    for i in range(smooth_window, energized_length - smooth_window):
        if local_status[i] == 'Stabilizing':
            # Check if this is just noise in an otherwise stable region
            surrounding = local_status[i-smooth_window:i+smooth_window+1]
            steady_count = np.sum(surrounding == 'Steady State')
            
            # If mostly surrounded by steady state, this is probably noise
            if steady_count > len(surrounding) * 0.7:
                # Verify it's actually stable
                segment = energized_voltage[i-5:i+6] if i >= 5 and i < energized_length-5 else energized_voltage[i:i+1]
                if np.std(segment) < noise_floor:
                    local_status[i] = 'Steady State'
    
    # --- STEP 7: ENFORCE MINIMUM LENGTHS ---
    # Prevent very short steady state regions
    current_state = local_status[0]
    state_start = 0
    
    for i in range(1, energized_length):
        if local_status[i] != current_state:
            state_length = i - state_start
            
            # If steady state region is too short, convert to stabilizing
            if current_state == 'Steady State' and state_length < MIN_STABLE_LENGTH:
                # Unless it's in the main stable region
                if not (state_start <= stable_start and i >= stable_end):
                    local_status[state_start:i] = 'Stabilizing'
            
            current_state = local_status[i]
            state_start = i
    
    # --- STEP 8: FINAL VALIDATION ---
    # Ensure we're not over-classifying the beginning or end
    # The first 10 points should be stabilizing unless voltage is already at steady state
    if energized_length > 10:
        first_10 = energized_voltage[:10]
        if np.std(first_10) > noise_floor or abs(np.mean(first_10) - stable_mean) > noise_floor:
            for i in range(min(10, energized_length)):
                if abs(energized_voltage[i] - stable_mean) > noise_floor:
                    local_status[i] = 'Stabilizing'
                else:
                    break  # Stop when we reach steady state
    
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
