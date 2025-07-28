import pandas as pd
import numpy as np
import pyarrow.parquet as pq # Import the pyarrow library
from collections import defaultdict
from tqdm import tqdm # A helpful library for progress bars: pip install tqdm
import matplotlib.pyplot as plt

# --- 1. Define a Function to Clean a SINGLE Group ---
# This function remains exactly the same.
def clean_one_group(group_df):
    
    def classify_voltage(voltage):
        if voltage < 2.2:
            return 'De-energized'
        elif 2.2 <= voltage < 22:
            return 'Stabilizing'
        else: # voltage >= 22
            return 'Steady State'
            
    group_df = group_df.copy()
    
    group_df['block_id'] = (group_df['dc1_status'] != group_df['dc1_status'].shift()).fillna(True).astype(int).cumsum()
    
    group_info = group_df.groupby('block_id').agg(
        min_voltage=('voltage_28v_dc1_cal', 'min'),
        max_voltage=('voltage_28v_dc1_cal', 'max'),
        state=('dc1_status', 'first')
    )
    
    is_unclean_steady = (group_info['state'] == 'Steady State') & (group_info['min_voltage'] < 22)
    is_unclean_stabilizing = (group_info['state'] == 'Stabilizing') & ((group_info['max_voltage'] >= 22) | (group_info['min_voltage'] < 2.2))
    unclean_block_ids = group_info[is_unclean_steady | is_unclean_stabilizing].index.tolist()
    
    group_df['Cleaned_Status'] = group_df['dc1_status']
    is_unclean_mask = group_df['block_id'].isin(unclean_block_ids)
    
    if is_unclean_mask.any():
        voltages_to_fix = group_df.loc[is_unclean_mask, 'voltage_28v_dc1_cal']
        corrections = voltages_to_fix.apply(classify_voltage)
        group_df.loc[is_unclean_mask, 'Cleaned_Status'] = corrections
    
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
# The resulting DataFrame has both original 'dc1_status' and new 'Cleaned_Status'
final_df = pd.concat(cleaned_dfs, ignore_index=True)
print("Processing complete.")

# --- 5. Perform and Export Before-and-After Analysis ---
print("\n" + "="*45)
print("  Performing Final Before-and-After Analysis")
print("="*45)

def perform_detailed_analysis(df, status_col, voltage_col, output_filename):
    """Groups a dataframe and calculates voltage stats, saving to CSV."""
    # Filter for the key states
    analysis_df = df[df[status_col].isin(['Stabilizing', 'Steady State'])]
    
    # Group by all the original grouping columns plus the status
    all_analysis_groups = group_cols + [status_col]
    
    print(f"\nGrouping and aggregating data based on '{status_col}'...")
    # Perform aggregation
    summary = analysis_df.groupby(all_analysis_groups)[voltage_col].agg(
        ['min', 'max', 'mean', 'std']
    ).reset_index()
    
    # Save to CSV
    summary.to_csv(output_filename, index=False)
    print(f"--> Analysis successful. Results saved to '{output_filename}'")
    return summary

# Run analysis on ORIGINAL data
before_summary = perform_detailed_analysis(
    df=final_df,
    status_col='dc1_status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_before_cleaning.csv'
)

# Run analysis on CLEANED data
after_summary = perform_detailed_analysis(
    df=final_df,
    status_col='Cleaned_Status',
    voltage_col='voltage_28v_dc1_cal',
    output_filename='analysis_after_cleaning.csv'
)

print("\n--- Analysis Complete ---")
print("\n'before_cleaning_summary.csv' contains stats on original data.")
print("'after_cleaning_summary.csv' contains stats on cleaned data.")


# --- 5. Finalize the DataFrame ---
# We now have 'dc1_status' (original) and 'Cleaned_Status'
final_df.drop(columns=['block_id'], inplace=True, errors='ignore')


# --- 6. Create Overlay Visualization ---
print("\n--- Generating Overlay Visualization ---")

# Use the cleaned status for our analysis
final_df['status_for_plot'] = final_df['Cleaned_Status']

# Step A: Find the start time for each group (first 'Stabilizing' OR 'Steady State' timestamp)
print("Step A: Finding ramp-up start time for each group...")
# THIS IS THE CORRECTED LOGIC
ramp_up_states = ['Stabilizing', 'Steady State']
ramp_up_rows = final_df[final_df['status_for_plot'].isin(ramp_up_states)]
start_times = ramp_up_rows.groupby(group_cols)['timestamp'].min().to_frame('start_time').reset_index()

# Step B: Merge start times back to the main DataFrame
plot_df = pd.merge(final_df, start_times, on=group_cols, how='inner')

# Step C: Calculate normalized time 
plot_df['normalized_time'] = plot_df['timestamp'] - plot_df['start_time']

# Step D: We can now plot all states, including 'De-energized' before T=0
# plot_df = plot_df[plot_df['status_for_plot'].isin(['Stabilizing', 'Steady State'])] # This filter is no longer necessary

# Step E: Create the plot
print("Step B: Generating plot...")
fig, ax = plt.subplots(figsize=(15, 8))

# Loop through each group and plot its voltage points
for name, group in plot_df.groupby(group_cols):
    ax.scatter(group['normalized_time'], group['voltage_28v_dc1_cal'], alpha=0.5, s=10)

# Add reference lines for clarity
ax.axhline(2.2, color='orange', linestyle='--', label='Stabilizing Threshold (2.2V)')
ax.axhline(22, color='green', linestyle='--', label='Steady State Threshold (22V)')
ax.axvline(0, color='gray', linestyle=':', label='T=0 (Start of Ramp-Up)')


ax.set_title('Overlayed Voltage Ramp-Up Profiles for All Test Groups', fontsize=16)
ax.set_xlabel('Normalized Time (from start of ramp-up)', fontsize=12)
ax.set_ylabel('Voltage (voltage_28v_dc1_cal)', fontsize=12)
ax.legend()
ax.grid(True)

# Save the figure to a file
plt.savefig('voltage_ramp_overlay_scatter.png', dpi=300)
print("\nVisualization complete. Plot saved to 'voltage_ramp_overlay_scatter.png'")