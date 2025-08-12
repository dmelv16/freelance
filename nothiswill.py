import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
from tqdm import tqdm
import os

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
    
    # Find ALL points that meet the energized criteria
    energized_mask = (voltage >= MINIMUM_VOLTAGE) & (~np.isnan(voltage))
    
    if not np.any(energized_mask):
        return pd.Series(status)
    
    # Get indices of energized points
    energized_indices = np.where(energized_mask)[0]
    
    # If there are gaps in energized data, we should handle continuous segments
    # For now, let's work with the continuous segment from first to last energized point
    start_idx, end_idx = energized_indices[0], energized_indices[-1]
    
    # Create a mask for the continuous region (including any gaps)
    continuous_region_mask = np.zeros(n, dtype=bool)
    continuous_region_mask[start_idx:end_idx + 1] = True
    
    # Within this region, mark all points >= MINIMUM_VOLTAGE as needing classification
    # This ensures ALL points meeting voltage criteria get classified
    points_to_classify = continuous_region_mask & energized_mask
    
    # Extract the continuous voltage segment for analysis
    energized_voltage = voltage[start_idx : end_idx + 1]
    energized_length = len(energized_voltage)
    
    # Create local status array for the continuous segment
    local_status = np.full(energized_length, 'De-energized', dtype=object)
    
    # Mark energized points within the local segment
    local_energized_mask = energized_mask[start_idx : end_idx + 1]
    
    # Too short - all stabilizing for energized points
    if energized_length < VARIANCE_WINDOW * 2:
        local_status[local_energized_mask] = 'Stabilizing'
        status[start_idx : end_idx + 1] = local_status
        return pd.Series(status)
    
    # --- MAIN ALGORITHM: Find where variance drops and stays low ---
    
    # Step 1: Calculate rolling variance (only for valid/energized windows)
    variances = []
    variance_positions = []
    
    for i in range(energized_length - VARIANCE_WINDOW + 1):
        window = energized_voltage[i:i+VARIANCE_WINDOW]
        window_mask = local_energized_mask[i:i+VARIANCE_WINDOW]
        
        # Only calculate variance if enough valid points in window
        if np.sum(window_mask) >= VARIANCE_WINDOW * 0.8:  # At least 80% valid points
            valid_window = window[window_mask]
            variances.append(np.var(valid_window))
            variance_positions.append(i)
    
    if len(variances) == 0:
        local_status[local_energized_mask] = 'Stabilizing'
        status[start_idx : end_idx + 1] = local_status
        return pd.Series(status)
    
    variances = np.array(variances)
    
    # Step 2: Find the baseline stable variance
    stable_variance = np.percentile(variances, 25)
    
    # Step 3: Find transition point where variance drops below threshold
    variance_threshold = stable_variance * 3
    
    # Initialize all energized points as stabilizing
    local_status[local_energized_mask] = 'Stabilizing'
    
    # Find the first point where variance drops and stays low
    transition_found = False
    consecutive_stable = 0
    required_consecutive = 5
    transition_point = -1
    
    for i, pos in enumerate(variance_positions):
        if variances[i] <= variance_threshold:
            consecutive_stable += 1
            if consecutive_stable >= required_consecutive and not transition_found:
                transition_point = pos
                transition_found = True
                # Mark energized points from here as steady state
                steady_mask = local_energized_mask.copy()
                steady_mask[:transition_point] = False
                local_status[steady_mask] = 'Steady State'
                break
        else:
            consecutive_stable = 0
    
    # Step 4: Refine the transition for specific patterns
    if transition_found and transition_point > MIN_STABILIZING_POINTS:
        # Get only the energized values for slope calculation
        energized_only_indices = np.where(local_energized_mask)[0]
        if len(energized_only_indices) > 10:
            # Check for settle-down or ramp-up patterns
            early_indices = energized_only_indices[energized_only_indices < transition_point]
            
            if len(early_indices) > 10:
                early_values = energized_voltage[early_indices]
                early_slope = np.polyfit(np.arange(len(early_values)), early_values, 1)[0]
                
                # If significant slope (up or down), find where it levels off
                if abs(early_slope) > 0.01:
                    for j in range(len(energized_only_indices) - 10):
                        window_indices = energized_only_indices[j:j+10]
                        window_values = energized_voltage[window_indices]
                        window_slope = np.polyfit(np.arange(10), window_values, 1)[0]
                        
                        if abs(window_slope) < 0.005:  # Slope has flattened
                            actual_transition = window_indices[0]
                            # Update classification
                            for idx in energized_only_indices:
                                if idx < actual_transition:
                                    local_status[idx] = 'Stabilizing'
                                else:
                                    local_status[idx] = 'Steady State'
                            transition_point = actual_transition
                            break
    
    # Step 5: Check the end for destabilization
    if energized_length > END_CHECK_WINDOW and transition_found:
        end_indices = np.where(local_energized_mask[-END_CHECK_WINDOW:])[0]
        if len(end_indices) > 5:
            end_values = energized_voltage[-END_CHECK_WINDOW:][end_indices]
            end_variance = np.var(end_values)
            
            # Check for destabilization at the end
            if end_variance > variance_threshold * 2:
                # Mark the end portion as stabilizing
                destab_start = energized_length - END_CHECK_WINDOW
                for idx in range(destab_start, energized_length):
                    if local_energized_mask[idx]:
                        local_status[idx] = 'Stabilizing'
    
    # Step 6: Handle edge cases
    if not transition_found:
        # Calculate overall variance for energized points only
        energized_values = energized_voltage[local_energized_mask]
        if len(energized_values) > 0:
            overall_variance = np.var(energized_values)
            if overall_variance < stable_variance * 2:
                local_status[local_energized_mask] = 'Steady State'
            else:
                local_status[local_energized_mask] = 'Stabilizing'
    
    # Ensure minimum stabilizing period at the start
    if transition_found and transition_point < MIN_STABILIZING_POINTS:
        energized_indices_local = np.where(local_energized_mask)[0]
        if len(energized_indices_local) >= MIN_STABILIZING_POINTS:
            initial_indices = energized_indices_local[:MIN_STABILIZING_POINTS]
            initial_values = energized_voltage[initial_indices]
            initial_variance = np.var(initial_values)
            
            if initial_variance > stable_variance * 3:
                # Enforce minimum stabilizing period
                for idx in initial_indices:
                    local_status[idx] = 'Stabilizing'
    
    # Map back to full array
    status[start_idx : end_idx + 1] = local_status
    
    # Final verification: ensure ALL points >= MINIMUM_VOLTAGE are classified
    # This is the critical fix - make sure no energized points remain "De-energized"
    for i in range(n):
        if voltage[i] >= MINIMUM_VOLTAGE and not np.isnan(voltage[i]):
            if status[i] == 'De-energized':
                # This point should not be de-energized - assign based on context
                # Look for nearby classified points
                window_start = max(0, i - 10)
                window_end = min(n, i + 10)
                nearby_statuses = status[window_start:window_end]
                
                # If there are steady state points nearby, use steady state
                if 'Steady State' in nearby_statuses:
                    status[i] = 'Steady State'
                else:
                    status[i] = 'Stabilizing'
    
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
