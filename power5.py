import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm

# --- 1. Define the Final Cleaning Function ---
def clean_one_group(group_df):
    
    TOLERANCE_PERCENT = 0.02
    group_df = group_df.copy()

    # --- First Pass: Calculate dynamic threshold and make initial classification ---
    initial_steady_points = group_df[group_df['voltage_28v_dc1_cal'] >= 22]
    dynamic_threshold = 22.0 
    
    if not initial_steady_points.empty:
        group_mean_voltage = initial_steady_points['voltage_28v_dc1_cal'].mean()
        dynamic_threshold = group_mean_voltage * (1 - TOLERANCE_PERCENT)

    # Make an initial classification based purely on voltage thresholds
    def initial_classify(voltage):
        if voltage < 2.2:
            return 'De-energized'
        elif voltage >= dynamic_threshold:
            return 'Steady State'
        else:
            return 'Stabilizing'
            
    group_df['Cleaned_Status'] = group_df['voltage_28v_dc1_cal'].apply(initial_classify)

    # --- Second Pass: Differentiate ramp-up, ramp-down, and dips ---
    # Find the index of the first and last true steady state points
    steady_indices = group_df.index[group_df['Cleaned_Status'] == 'Steady State']
    
    if not steady_indices.empty:
        first_steady_index = steady_indices[0]
        last_steady_index = steady_indices[-1]

        # Identify points that are stabilizing but are between the first and last steady state points
        is_stabilizing = group_df['Cleaned_Status'] == 'Stabilizing'
        is_between = (group_df.index > first_steady_index) & (group_df.index < last_steady_index)
        
        # Re-classify these in-between dips as 'Steady State'
        group_df.loc[is_stabilizing & is_between, 'Cleaned_Status'] = 'Steady State'

    return group_df
# --- 2. Read Files in Chunks and Gather Groups ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 
groups_data = defaultdict(list)

print("Step 1: Reading files in chunks and gathering groups...")
for file_path in file_list:
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"Finished gathering data for {len(groups_data)} unique groups.")

# --- 3. Process Each Group and Collect Results ---
cleaned_dfs = []
print("\nStep 2: Cleaning each group one by one...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    full_group_df = pd.concat(list_of_chunks)
    cleaned_group = clean_one_group(full_group_df)
    cleaned_dfs.append(cleaned_group)

# --- 4. Combine All Cleaned Groups ---
print("\nStep 3: Combining all cleaned groups into the final result...")
final_df = pd.concat(cleaned_dfs, ignore_index=True)
print("Processing complete.")

# --- 5. Perform and Export Before-and-After Analysis ---
print("\n" + "="*45)
print("  Performing Final Before-and-After Analysis")
print("="*45)

def perform_detailed_analysis(df, status_col, voltage_col, output_filename):
    """Groups a dataframe and calculates voltage stats, saving to CSV."""
    analysis_df = df[df[status_col].isin(['Stabilizing', 'Steady State'])]
    all_analysis_groups = group_cols + [status_col]
    print(f"\nGrouping and aggregating data based on '{status_col}'...")
    summary = analysis_df.groupby(all_analysis_groups)[voltage_col].agg(
        ['min', 'max', 'mean', 'std']
    ).reset_index()
    summary.to_csv(output_filename, index=False)
    print(f"--> Analysis successful. Results saved to '{output_filename}'")
    return summary

# Run analysis using the ORIGINAL 'dc1_status' column
before_summary = perform_detailed_analysis(
    df=final_df,
    status_col='dc1_status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_before_cleaning.csv'
)

# Run analysis using the NEW 'Cleaned_Status' column
after_summary = perform_detailed_analysis(
    df=final_df,
    status_col='Cleaned_Status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_after_cleaning.csv'
)

print("\n--- Analysis Complete ---")
