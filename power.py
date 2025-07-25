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

# Add row_id for tracking original order
df['row_id'] = range(len(df))

# Store original columns for later
original_cols = df.columns.tolist()

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
    df_group['system_cycle'] = df_group['turn_on_event'].fillna(False).astype(int).cumsum()

    # Vectorized time since turn-on
    if df_group['system_cycle'].max() > 0:
        turn_on_times = df_group.groupby('system_cycle')['timestamp'].transform('first')
        df_group['time_since_turn_on_ms'] = (df_group['timestamp'] - turn_on_times).dt.total_seconds() * 1000
        df_group.loc[df_group['system_cycle'] == 0, 'time_since_turn_on_ms'] = np.nan
    else:
        df_group['time_since_turn_on_ms'] = np.nan

    # Vectorized rolling stats for stability check
    df_group['voltage_rolling_std'] = np.nan
    median_time_diff = df_group['time_diff_ms'].median()
    if pd.notna(median_time_diff) and median_time_diff > 0 and len(df_group) > 3:
        window_size = max(3, int(VOLTAGE_STABILITY_WINDOW_MS / median_time_diff))
        df_group['voltage_rolling_std'] = df_group['voltage_28v_dc1_cal'].rolling(
            window=window_size, center=True, min_periods=1
        ).std()

    # Define conditions for state classification
    voltage = df_group['voltage_28v_dc1_cal']
    time_since = df_group['time_since_turn_on_ms']
    voltage_std = df_group['voltage_rolling_std']

    cond_tester_error = voltage.isna() | (voltage < 0.1)
    cond_de_energized = voltage < DE_ENERGIZED_MAX
    cond_stabilizing_time = time_since.notna() & (time_since < STEADY_STATE_DELAY_MS)
    cond_steady_state = ((voltage >= STEADY_STATE_MIN) & (voltage <= STEADY_STATE_MAX) &
                         voltage_std.notna() & (voltage_std < VOLTAGE_STABILITY_THRESHOLD) &
                         time_since.notna() & (time_since >= STEADY_STATE_DELAY_MS))
    cond_stabilizing_volt = (voltage >= STABILIZING_MIN) & (voltage < STEADY_STATE_MIN)
    cond_stabilizing_std = ((voltage >= STEADY_STATE_MIN) & voltage_std.notna() &
                            (voltage_std >= VOLTAGE_STABILITY_THRESHOLD))
    cond_overvoltage = voltage > STEADY_STATE_MAX

    # Assign states in order of priority
    df_group['true_state'] = 'Transient'
    df_group.loc[cond_overvoltage, 'true_state'] = 'Overvoltage'
    df_group.loc[cond_stabilizing_std, 'true_state'] = 'Stabilizing'
    df_group.loc[cond_stabilizing_volt, 'true_state'] = 'Stabilizing'
    df_group.loc[cond_steady_state, 'true_state'] = 'Steady State'
    df_group.loc[cond_stabilizing_time, 'true_state'] = 'Stabilizing'
    df_group.loc[cond_de_energized, 'true_state'] = 'De-energized'
    df_group.loc[cond_tester_error, 'true_state'] = 'Tester Error'
    
    return df_group

def refine_steady_state_dips(df_group, window_size=7, threshold=0.75):
    """Reclassifies brief dips from 'Steady State' back to 'Steady State'."""
    is_steady_mask = (df_group['true_state'] == 'Steady State').astype(int)
    steady_neighbor_ratio = is_steady_mask.rolling(
        window=window_size, center=True, min_periods=1
    ).mean()
    reclassify_mask = (
        (df_group['true_state'] == 'Stabilizing') &
        (steady_neighbor_ratio >= threshold)
    )
    df_group.loc[reclassify_mask, 'true_state'] = 'Steady State'
    return df_group

# ===================================
# UPDATED FUNCTION FOR PROBLEMATIC GROUPS
# ===================================
def process_and_filter_problematic_group(df_group, final_cols):
    """
    For a problematic group, identify states, keep all 'Stabilizing' data,
    and keep the first and last 50 'Steady State' readings.
    Discard all other states (e.g., Transient, Overvoltage).
    """
    # Run state detection to classify each row
    df_group = identify_system_states(df_group)
    df_group = refine_steady_state_dips(df_group)

    # 1. Isolate all 'Stabilizing' rows
    stabilizing_rows = df_group[df_group['true_state'] == 'Stabilizing']

    # 2. Isolate and filter 'Steady State' rows
    steady_state_rows = df_group[df_group['true_state'] == 'Steady State']
    if len(steady_state_rows) > 100:
        first_50 = steady_state_rows.head(50)
        last_50 = steady_state_rows.tail(50)
        kept_steady_rows = pd.concat([first_50, last_50])
    else:
        # If 100 or fewer steady state rows, keep all of them
        kept_steady_rows = steady_state_rows

    # 3. Combine the two sets of rows we want to keep
    rows_to_keep = pd.concat([stabilizing_rows, kept_steady_rows])

    # If we have any rows left, update their status and select original columns
    if not rows_to_keep.empty:
        # Sort by original row_id to maintain chronological order
        rows_to_keep = rows_to_keep.sort_values('row_id')
        rows_to_keep = rows_to_keep.copy()

        # Update the status column to reflect the new classification
        rows_to_keep['dc1_status'] = rows_to_keep['true_state']

        # Return only the columns present in the original DataFrame
        return rows_to_keep[final_cols]
    else:
        # If no rows are kept, return an empty DataFrame with correct columns
        return pd.DataFrame(columns=final_cols)

# ===================================
# MAIN PROCESSING LOOP
# ===================================
print("\nStarting main processing loop...")

# Initialize collection lists
aggregated_results_list = []
cleaned_groups_list = []
unchanged_groups_list = []

# Process each group
for group_num, group_values in enumerate(unique_groups, 1):
    if group_num % 100 == 0:
        print(f"  Processing group {group_num}/{len(unique_groups)}...")
    
    # Create group DataFrame by filtering main df
    mask = True
    for col, val in zip(group_cols, group_values):
        mask &= (df[col] == val)
    df_group = df[mask].copy()
    
    if len(df_group) == 0:
        continue
    
    # ===== Part 1: Aggregation (Done for all groups) =====
    agg_result = {
        'save': group_values[0], 'unit_id': group_values[1], 'ofp': group_values[2],
        'station': group_values[3], 'test_case': group_values[4], 'test_run': group_values[5],
        'voltage_min': df_group['voltage_28v_dc1_cal'].min(),
        'voltage_max': df_group['voltage_28v_dc1_cal'].max(),
        'voltage_median': df_group['voltage_28v_dc1_cal'].median(),
        'timestamp_min': df_group['timestamp'].min(),
        'timestamp_max': df_group['timestamp'].max(),
    }
    aggregated_results_list.append(agg_result)
    
    # ===== Part 2: State Filtering (only for problematic groups) =====
    is_problematic = agg_result['voltage_min'] < STEADY_STATE_MIN
    
    if is_problematic:
        # Process the group using the new filtering logic
        filtered_group = process_and_filter_problematic_group(df_group, original_cols)
        
        # Add the filtered data to the list of cleaned groups
        if not filtered_group.empty:
            cleaned_groups_list.append(filtered_group)
    else:
        # Group is not problematic, add it directly to the unchanged list
        unchanged_groups_list.append(df_group)
    
    # Clean up memory periodically
    del df_group
    if group_num % 500 == 0:
        gc.collect()

# ===================================
# SAVE ALL RESULTS
# ===================================
print("\nProcessing complete. Saving all files...")

# Aggregated results (summarizes groups before filtering)
result_df = pd.DataFrame(aggregated_results_list)
result_df.to_csv('aggregated_results.csv', index=False)
print(f"Aggregation results saved: {len(result_df)} groups summarized")

# Combine all processed data
print("\nCombining all processed data for final output...")

# Create DataFrames from the lists of processed groups
df_cleaned = pd.concat(cleaned_groups_list, ignore_index=True) if cleaned_groups_list else pd.DataFrame()
df_unchanged = pd.concat(unchanged_groups_list, ignore_index=True) if unchanged_groups_list else pd.DataFrame()

# Create the final, complete DataFrame
df_final = pd.concat([df_cleaned, df_unchanged], ignore_index=True)

if not df_final.empty:
    # Sort by the original row_id to restore the dataset's original order
    df_final = df_final.sort_values('row_id').reset_index(drop=True)
    
    # Save the final, cleaned dataset
    df_final.to_csv('power_data_complete_cleaned.csv', index=False)
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Total rows in original dataset: {len(df):,}")
    print(f"Total rows in final dataset: {len(df_final):,}")
    print(f"Total groups processed: {len(unique_groups)}")
    num_problematic = len(unique_groups) - len(unchanged_groups_list)
    print(f"Problematic groups filtered: {num_problematic}")
    rows_dropped = len(df) - len(df_final)
    print(f"Total rows dropped: {rows_dropped:,} ({rows_dropped/len(df)*100:.2f}%)")
else:
    print("No data was processed or kept!")

print("\nAll processing complete!")