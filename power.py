import pandas as pd
import numpy as np
import gc  # For garbage collection

def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by downcasting numeric types and converting
    low-cardinality strings to categories.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                # Integer downcasting logic
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # Float downcasting logic
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        # Convert low-cardinality string columns to category
        elif col_type == 'object':
            # If the number of unique values is less than 50% of the total, convert to category
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# Configuration
STEADY_STATE_MIN = 22.0
STEADY_STATE_MAX = 29.0
STABILIZING_MIN = 2.2
STABILIZING_MAX = 22.0
STEADY_STATE_DELAY_MS = 3
VOLTAGE_STABILITY_WINDOW_MS = 5
VOLTAGE_STABILITY_THRESHOLD = 0.5
DE_ENERGIZED_MAX = 2.2

# Define grouping columns
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run']

# ===================================
# LOAD DATA ONCE AND OPTIMIZE
# ===================================
print("Loading main dataset...")
df = pd.read_csv('power_data.csv')

# Immediately reduce memory footprint
df = reduce_mem_usage(df)

# Convert timestamp once
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Add row_id for tracking
df['row_id'] = range(len(df))

# Get unique groups from the in-memory DataFrame
unique_groups = df[group_cols].drop_duplicates().values.tolist()
print(f"Found {len(unique_groups)} unique groups to process")

# ===================================
# OPTIMIZED STATE DETECTION FUNCTIONS
# ===================================
def identify_system_states(df_group):
    """Identify true system states using a fast, vectorized approach."""
    df_group = df_group.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate helper columns
    df_group['time_diff_ms'] = df_group['timestamp'].diff().dt.total_seconds() * 1000
    df_group['voltage_diff'] = df_group['voltage_28v_dc1_cal'].diff()
    
    # Identify turn-on events
    df_group['turn_on_event'] = (
        (df_group['voltage_28v_dc1_cal'].shift(1) < STABILIZING_MIN) &
        (df_group['voltage_28v_dc1_cal'] >= STABILIZING_MIN) &
        (df_group['voltage_diff'] > 1.0)
    )
    df_group['system_cycle'] = df_group['turn_on_event'].cumsum()
    
    # Vectorized time since turn-on
    if df_group['system_cycle'].max() > 0:
        turn_on_times = df_group.groupby('system_cycle')['timestamp'].transform('first')
        df_group['time_since_turn_on_ms'] = (df_group['timestamp'] - turn_on_times).dt.total_seconds() * 1000
        df_group.loc[df_group['system_cycle'] == 0, 'time_since_turn_on_ms'] = np.nan
    else:
        df_group['time_since_turn_on_ms'] = np.nan
    
    # Vectorized rolling stats
    if len(df_group) > 3 and df_group['time_diff_ms'].median() > 0:
        window_size = max(3, int(VOLTAGE_STABILITY_WINDOW_MS / df_group['time_diff_ms'].median()))
        df_group['voltage_rolling_std'] = df_group['voltage_28v_dc1_cal'].rolling(
            window=window_size, center=True, min_periods=1
        ).std()
    else:
        df_group['voltage_rolling_std'] = np.nan
    
    # Vectorized state classification using np.select
    voltage = df_group['voltage_28v_dc1_cal']
    time_since = df_group['time_since_turn_on_ms']
    voltage_std = df_group['voltage_rolling_std']
    
    conditions = [
        voltage.isna() | (voltage < 0.1),
        voltage < DE_ENERGIZED_MAX,
        time_since.notna() & (time_since < STEADY_STATE_DELAY_MS),
        (voltage >= STEADY_STATE_MIN) & (voltage <= STEADY_STATE_MAX) & 
        voltage_std.notna() & (voltage_std < VOLTAGE_STABILITY_THRESHOLD) & 
        time_since.notna() & (time_since >= STEADY_STATE_DELAY_MS),
        (voltage >= STABILIZING_MIN) & (voltage < STEADY_STATE_MIN),
        (voltage >= STEADY_STATE_MIN) & voltage_std.notna() & 
        (voltage_std >= VOLTAGE_STABILITY_THRESHOLD),
        voltage > STEADY_STATE_MAX
    ]
    
    choices = [
        'Tester Error',
        'De-energized',
        'Stabilizing',
        'Steady State',
        'Stabilizing',
        'Stabilizing',
        'Overvoltage'
    ]
    
    df_group['true_state'] = np.select(conditions, choices, default='Transient')
    
    # Handle edge cases where time_since_turn_on is NaN but voltage is high
    mask_possible_steady = (
        time_since.isna() & 
        (voltage >= STEADY_STATE_MIN) & 
        (df_group['true_state'] == 'Transient')
    )
    df_group.loc[mask_possible_steady, 'true_state'] = 'Possible Steady State'
    
    return df_group

def refine_transient_detection_vectorized(df_group):
    """Vectorized version of transient refinement"""
    df_group = df_group.sort_values('timestamp').reset_index(drop=True)
    
    # Only process if we have steady states
    if not (df_group['true_state'] == 'Steady State').any():
        return df_group
    
    # Create a rolling window count of steady states
    window_size = 11  # 5 before + current + 5 after
    steady_state_mask = (df_group['true_state'] == 'Steady State').astype(int)
    steady_state_ratio = steady_state_mask.rolling(
        window=window_size, center=True, min_periods=1
    ).mean()
    
    # Find steady state readings that are actually transients
    transient_mask = (
        (df_group['true_state'] == 'Steady State') &
        (steady_state_ratio > 0.7) &
        ((df_group['voltage_28v_dc1_cal'] < STEADY_STATE_MIN) | 
         (df_group['voltage_28v_dc1_cal'] > STEADY_STATE_MAX))
    )
    
    df_group.loc[transient_mask, 'true_state'] = 'Transient'
    
    # Add transient type if available
    if 'dc1_trans' in df_group.columns:
        df_group.loc[transient_mask & df_group['dc1_trans'].str.contains('Normal', na=False), 
                     'transient_type'] = 'Normal Transient'
        df_group.loc[transient_mask & df_group['dc1_trans'].str.contains('Abnormal', na=False), 
                     'transient_type'] = 'Abnormal Transient'
    
    return df_group

# ===================================
# MAIN PROCESSING LOOP
# ===================================
print("\nStarting main processing loop...")

# Initialize collection lists
aggregated_results_list = []
cleaned_groups_list = []
unchanged_groups_list = []
state_durations_list = []
change_logs_list = []
group_change_summary_list = []

# Track overall statistics
total_rows_changed = 0

# Process each group
for group_num, group_values in enumerate(unique_groups, 1):
    if group_num % 100 == 0:
        print(f"  Processing group {group_num}/{len(unique_groups)}...")
    
    # Create group DataFrame by filtering main df (very fast)
    mask = True
    for col, val in zip(group_cols, group_values):
        mask &= (df[col] == val)
    df_group = df[mask].copy()
    
    if len(df_group) == 0:
        continue
    
    # ===== Part 1: Aggregation =====
    agg_result = {
        'save': group_values[0],
        'unit_id': group_values[1],
        'ofp': group_values[2],
        'station': group_values[3],
        'test_case': group_values[4],
        'test_run': group_values[5],
        'voltage_min': df_group['voltage_28v_dc1_cal'].min(),
        'voltage_max': df_group['voltage_28v_dc1_cal'].max(),
        'voltage_median': df_group['voltage_28v_dc1_cal'].median(),
        'voltage_sd': df_group['voltage_28v_dc1_cal'].std(),
        'voltage_range': df_group['voltage_28v_dc1_cal'].max() - df_group['voltage_28v_dc1_cal'].min(),
        'current_min': df_group['current_28v_dc1_cal'].min(),
        'current_max': df_group['current_28v_dc1_cal'].max(),
        'current_median': df_group['current_28v_dc1_cal'].median(),
        'current_sd': df_group['current_28v_dc1_cal'].std(),
        'current_range': df_group['current_28v_dc1_cal'].max() - df_group['current_28v_dc1_cal'].min(),
        'timestamp_min': df_group['timestamp'].min(),
        'timestamp_max': df_group['timestamp'].max(),
        'norm_trans': (df_group['dc1_trans'] == 'Normal Transient').sum(),
        'abnorm_trans': (df_group['dc1_trans'] == 'Abnormal Transient').sum()
    }
    agg_result['duration'] = agg_result['timestamp_max'] - agg_result['timestamp_min']
    aggregated_results_list.append(agg_result)
    
    # ===== Part 2: State Detection (only for problematic groups) =====
    is_problematic = agg_result['voltage_min'] < STEADY_STATE_MIN
    
    if is_problematic:
        # Store original status
        df_group['dc1_status_original'] = df_group['dc1_status']
        
        # Apply state detection
        df_group = identify_system_states(df_group)
        df_group = refine_transient_detection_vectorized(df_group)
        
        # Update dc1_status with corrected states
        df_group['dc1_status'] = df_group['true_state']
        df_group['state_mismatch'] = df_group['dc1_status_original'] != df_group['dc1_status']
        
        # Count changes
        rows_changed = df_group['state_mismatch'].sum()
        total_rows_changed += rows_changed
        
        # Collect state durations (fully vectorized)
        if 'true_state' in df_group.columns:
            state_changes = df_group['true_state'] != df_group['true_state'].shift()
            state_groups = state_changes.cumsum()
            
            duration_data = df_group.groupby(['true_state', state_groups]).agg({
                'timestamp': ['min', 'max'],
                'voltage_28v_dc1_cal': ['mean', 'std']
            }).reset_index()
            
            if not duration_data.empty:
                # Handle MultiIndex columns properly
                # The columns will be: ('true_state', ''), ('state_groups', ''), 
                # ('timestamp', 'min'), ('timestamp', 'max'), 
                # ('voltage_28v_dc1_cal', 'mean'), ('voltage_28v_dc1_cal', 'std')
                
                # Calculate duration using the MultiIndex columns
                duration_data['duration_seconds'] = (
                    duration_data[('timestamp', 'max')] - 
                    duration_data[('timestamp', 'min')]
                ).dt.total_seconds()
                
                # Create a new DataFrame with flat column names
                duration_flat = pd.DataFrame({
                    'true_state': duration_data[('true_state', '')],
                    'duration_seconds': duration_data['duration_seconds'],
                    'voltage_mean': duration_data[('voltage_28v_dc1_cal', 'mean')],
                    'voltage_std': duration_data[('voltage_28v_dc1_cal', 'std')]
                })
                
                # Filter for desired states
                duration_flat = duration_flat[
                    duration_flat['true_state'].isin(['Steady State', 'Stabilizing', 'Transient'])
                ]
                
                if not duration_flat.empty:
                    # Add group identifier columns
                    for i, col in enumerate(group_cols):
                        duration_flat[col] = group_values[i]
                    
                    # Rename for consistency
                    duration_flat = duration_flat.rename(columns={'true_state': 'state'})
                    
                    # Append the entire DataFrame to the list
                    state_durations_list.append(duration_flat)
        
        # Collect change logs
        if rows_changed > 0:
            change_log = df_group[df_group['state_mismatch']][
                ['row_id'] + group_cols + ['timestamp', 'voltage_28v_dc1_cal',
                 'dc1_status_original', 'dc1_status', 'system_cycle', 'time_since_turn_on_ms']
            ]
            change_logs_list.append(change_log)
        
        # Create group summary
        group_change_summary_list.append({
            'save': group_values[0], 'unit_id': group_values[1],
            'ofp': group_values[2], 'station': group_values[3],
            'test_case': group_values[4], 'test_run': group_values[5],
            'total_rows': len(df_group),
            'rows_changed': rows_changed,
            'percent_changed': (rows_changed / len(df_group) * 100) if len(df_group) > 0 else 0,
            'voltage_min': agg_result['voltage_min'],
            'voltage_max': agg_result['voltage_max'],
            'voltage_mean': df_group['voltage_28v_dc1_cal'].mean(),
            'original_steady_state_count': (df_group['dc1_status_original'] == 'Steady State').sum(),
            'new_steady_state_count': (df_group['true_state'] == 'Steady State').sum()
        })
        
        cleaned_groups_list.append(df_group)
    else:
        # Group is unchanged
        unchanged_groups_list.append(df_group)
    
    # Clean up memory periodically
    del df_group
    if group_num % 500 == 0:
        gc.collect()

# ===================================
# SAVE ALL RESULTS
# ===================================
print("\nProcessing complete. Saving all files...")

# Aggregated results
result_df = pd.DataFrame(aggregated_results_list)
result_df.to_csv('aggregated_results.csv', index=False)
print(f"Aggregation results saved: {len(result_df)} groups")

# State durations
if state_durations_list:
    # Concatenate all duration DataFrames
    duration_df = pd.concat(state_durations_list, ignore_index=True)
    duration_df.to_csv('state_duration_analysis.csv', index=False)
    print(f"State duration analysis saved: {len(duration_df)} records")

# Change logs
if change_logs_list:
    change_log_all = pd.concat(change_logs_list, ignore_index=True)
    change_log_all.to_csv('state_changes_detailed.csv', index=False)
    print(f"State changes log saved: {len(change_log_all)} changes")

# Group change summary
if group_change_summary_list:
    change_summary_df = pd.DataFrame(group_change_summary_list)
    change_summary_df = change_summary_df.sort_values('rows_changed', ascending=False)
    change_summary_df.to_csv('group_change_summary.csv', index=False)
    print(f"Group change summary saved: {len(change_summary_df)} problematic groups")

# Combine all processed data
print("\nCombining all processed data...")

# Create the cleaned and unchanged DataFrames first
df_cleaned = pd.concat(cleaned_groups_list, ignore_index=True) if cleaned_groups_list else pd.DataFrame()
df_unchanged = pd.concat(unchanged_groups_list, ignore_index=True) if unchanged_groups_list else pd.DataFrame()

# Create final DataFrame
df_final = pd.concat([df_cleaned, df_unchanged], ignore_index=True)
if not df_final.empty:
    df_final = df_final.sort_values('row_id').reset_index(drop=True)
    
    # Create comparison statistics using already-created df_cleaned
    if not df_cleaned.empty:
        comparison = pd.crosstab(df_cleaned['dc1_status_original'], 
                               df_cleaned['true_state'], margins=True)
        print("\nState Classification Comparison:")
        print(comparison)
        
        print("\nTrue State Distribution in cleaned data:")
        print(df_cleaned['true_state'].value_counts())
    
    # Save final dataset
    df_final.to_csv('power_data_complete_cleaned.csv', index=False)
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Total rows processed: {len(df_final):,}")
    print(f"Problematic groups processed: {len([g for g in group_change_summary_list])}")
    print(f"Total groups processed: {len(unique_groups)}")
    print(f"Total rows with state changes: {total_rows_changed:,}")
    print(f"Percentage of data corrected: {total_rows_changed/len(df_final)*100:.2f}%")
else:
    print("No data was processed!")

print("\nAll processing complete!")