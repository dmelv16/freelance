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
    
    # IMPORTANT: First, mark ALL energized points as at least "Stabilizing"
    # This ensures no point with voltage >= MINIMUM_VOLTAGE remains "De-energized"
    status[energized_mask] = 'Stabilizing'
    
    # Get indices of energized points
    energized_indices = np.where(energized_mask)[0]
    
    # Find continuous segments of energized points
    # (allowing for small gaps of up to 5 points)
    segments = []
    current_segment_start = energized_indices[0]
    last_idx = energized_indices[0]
    
    for idx in energized_indices[1:]:
        if idx - last_idx > 5:  # Gap larger than 5 points - new segment
            segments.append((current_segment_start, last_idx))
            current_segment_start = idx
        last_idx = idx
    segments.append((current_segment_start, last_idx))
    
    # Process each continuous segment
    for seg_start, seg_end in segments:
        segment_length = seg_end - seg_start + 1
        
        # Skip very short segments
        if segment_length < VARIANCE_WINDOW:
            # Already marked as Stabilizing, so continue
            continue
        
        # Extract segment voltage
        segment_voltage = voltage[seg_start:seg_end + 1]
        segment_mask = energized_mask[seg_start:seg_end + 1]
        
        # Calculate rolling variance for this segment
        variances = []
        variance_positions = []
        
        for i in range(segment_length - VARIANCE_WINDOW + 1):
            window = segment_voltage[i:i+VARIANCE_WINDOW]
            window_mask = segment_mask[i:i+VARIANCE_WINDOW]
            
            # Only calculate variance if enough valid points in window
            if np.sum(window_mask) >= VARIANCE_WINDOW * 0.7:  # At least 70% valid points
                valid_window = window[window_mask]
                if len(valid_window) > 1:  # Need at least 2 points for variance
                    variances.append(np.var(valid_window))
                    variance_positions.append(i)
        
        if len(variances) == 0:
            continue  # Keep as Stabilizing
        
        variances = np.array(variances)
        
        # Find the baseline stable variance
        stable_variance = np.percentile(variances, 25)
        variance_threshold = stable_variance * 3
        
        # Find transition point where variance drops and stays low
        consecutive_stable = 0
        required_consecutive = 5
        transition_found = False
        transition_point = -1
        
        for i, pos in enumerate(variance_positions):
            if variances[i] <= variance_threshold:
                consecutive_stable += 1
                if consecutive_stable >= required_consecutive and not transition_found:
                    transition_point = seg_start + pos
                    transition_found = True
                    break
            else:
                consecutive_stable = 0
        
        # If transition found, mark steady state region
        if transition_found:
            # Mark all energized points from transition point onward as steady state
            for idx in range(transition_point, seg_end + 1):
                if energized_mask[idx]:
                    status[idx] = 'Steady State'
            
            # Check for destabilization at the end
            if segment_length > END_CHECK_WINDOW:
                end_start = seg_end - END_CHECK_WINDOW + 1
                end_indices = np.where(energized_mask[end_start:seg_end + 1])[0]
                
                if len(end_indices) > 5:
                    end_values = voltage[end_start:seg_end + 1][end_indices]
                    end_variance = np.var(end_values)
                    
                    if end_variance > variance_threshold * 2:
                        # Mark end as stabilizing
                        for idx in range(end_start, seg_end + 1):
                            if energized_mask[idx]:
                                status[idx] = 'Stabilizing'
        
        else:
            # No clear transition found - check if it's all stable or all variable
            energized_values = segment_voltage[segment_mask]
            if len(energized_values) > 1:
                overall_variance = np.var(energized_values)
                
                if overall_variance < stable_variance * 2:
                    # Low variance throughout - mark all as steady state
                    for idx in range(seg_start, seg_end + 1):
                        if energized_mask[idx]:
                            status[idx] = 'Steady State'
                # Otherwise, keep as Stabilizing (already set)
    
    # Handle isolated energized points in gaps between segments
    # These should inherit the classification from surrounding points
    for i in np.where(energized_mask)[0]:
        # Check if this point is isolated (not part of a continuous segment)
        is_isolated = True
        
        # Check if there are other energized points very close by
        for j in range(max(0, i-2), min(n, i+3)):
            if j != i and energized_mask[j]:
                is_isolated = False
                break
        
        if is_isolated or status[i] == 'Stabilizing':
            # Look at a wider window to determine what this point should be
            window_size = 50
            window_start = max(0, i - window_size)
            window_end = min(n, i + window_size)
            
            # Count the states in the surrounding window
            window_statuses = status[window_start:window_end]
            steady_count = np.sum(window_statuses == 'Steady State')
            stabilizing_count = np.sum(window_statuses == 'Stabilizing')
            
            # If surrounded mostly by steady state, make it steady state
            if steady_count > stabilizing_count and steady_count > 0:
                status[i] = 'Steady State'
            
            # Alternative: if it's between two steady state regions, make it steady state
            # Look for steady state before and after
            steady_before = False
            steady_after = False
            
            # Check before
            for j in range(i-1, max(0, i-20), -1):
                if status[j] == 'Steady State':
                    steady_before = True
                    break
                elif status[j] == 'Stabilizing' and energized_mask[j]:
                    break  # Hit a stabilizing region, stop looking
            
            # Check after
            for j in range(i+1, min(n, i+20)):
                if status[j] == 'Steady State':
                    steady_after = True
                    break
                elif status[j] == 'Stabilizing' and energized_mask[j]:
                    break  # Hit a stabilizing region, stop looking
            
            # If between steady states, make it steady state
            if steady_before and steady_after:
                status[i] = 'Steady State'
    
    # FINAL SAFETY CHECK: Absolutely ensure no energized point is "De-energized"
    final_check = (voltage >= MINIMUM_VOLTAGE) & (~np.isnan(voltage))
    de_energized_but_should_not_be = final_check & (status == 'De-energized')
    
    if np.any(de_energized_but_should_not_be):
        print(f"WARNING: Found {np.sum(de_energized_but_should_not_be)} points with voltage >= {MINIMUM_VOLTAGE}V still marked as De-energized")
        print(f"Voltage range: {voltage[de_energized_but_should_not_be].min():.1f} - {voltage[de_energized_but_should_not_be].max():.1f}V")
        print(f"Indices: {np.where(de_energized_but_should_not_be)[0][:10]}...")  # Show first 10 indices
        
        # Force these to be at least Stabilizing
        status[de_energized_but_should_not_be] = 'Stabilizing'
    
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
