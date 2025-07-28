import pandas as pd
import numpy as np

# --- 1. Load and Prepare Sample Data ---
# The sample data now includes all your specified grouping columns.
data = {
    # Your full grouping columns from the provided script
    'save':       ['s1', 's1', 's1', 's1', 's1', 's2', 's2', 's2', 's2'],
    'unit_id':    ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
    'ofp':        ['v1', 'v1', 'v1', 'v1', 'v1', 'v2', 'v2', 'v2', 'v2'],
    'station':    ['st1', 'st1', 'st1', 'st1', 'st1', 'st2', 'st2', 'st2', 'st2'],
    'test_case':  ['tc1', 'tc1', 'tc1', 'tc1', 'tc1', 'tc5', 'tc5', 'tc5', 'tc5'],
    'test_run':   [1, 1, 1, 1, 1, 4, 4, 4, 4],
    'Aircraft':   ['p1', 'p1', 'p1', 'p1', 'p1', 'p2', 'p2', 'p2', 'p2'],

    'timestamp':  [100, 101, 102, 103, 104, 200, 201, 202, 203],
    'voltage_28v_dc1_cal': [
        23.1, 24.5, 21.9, 25.2, 24.9, # Test 1: 'Steady State' with a voltage dip
        18.0, 19.5, 22.5, 21.0        # Test 2: 'Stabilizing' with a voltage spike
    ],
    'dc1_status': [
        'Steady State', 'Steady State', 'Stabilizing', 'Steady State', 'Steady State',
        'Stabilizing', 'Stabilizing', 'Stabilizing', 'Stabilizing'
    ],
    'dc1_trans': ['None'] * 9
}
df = pd.DataFrame(data)
print("--- Original Data ---")
print(df.head())


# --- 2. Define Grouping and Remove Transients ---
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
values_to_remove = ['Error 0 volt', 'Error missing volt', 'Normal transient', 'Abnormal transient']
df_cleaned = df[~df['dc1_trans'].isin(values_to_remove)]

# --- 3. Identify Misclassified Groups ---
# Create the block_id, which is unique WITHIN each test run group
df_cleaned['block_id'] = df_cleaned.groupby(group_cols)['dc1_status'].transform(
    lambda x: (x != x.shift()).fillna(True).astype(int).cumsum(),
    meta=('block_id', 'int64')
)

# Define the columns that uniquely identify each block
block_identifier_cols = group_cols + ['block_id']

# Perform the small aggregation to find which blocks are unclean
group_info = df_cleaned.groupby(block_identifier_cols).agg(
    min_voltage=('voltage_28v_dc1_cal', 'min'),
    max_voltage=('voltage_28v_dc1_cal', 'max'),
    state=('dc1_status', 'first')
).compute() # Compute this small summary table

# Determine which groups are unclean based on the rules
is_unclean_steady = (group_info['state'] == 'Steady State') & (group_info['min_voltage'] < 22)
is_unclean_stabilizing = (group_info['state'] == 'Stabilizing') & ((group_info['max_voltage'] >= 22) | (group_info['min_voltage'] < 2.2))
group_info['is_unclean'] = is_unclean_steady | is_unclean_stabilizing

# Get just the list of unclean blocks to join back
unclean_blocks = group_info[group_info['is_unclean']].reset_index()[block_identifier_cols]

# --- 4. Map Unclean Status to All Rows using Merge ---
# This is the key change. We are "broadcasting" the unclean status to all rows
# that belong to an unclean block. The 'how="left"' ensures we keep all original rows.
df_merged = dd.merge(
    df_cleaned,
    unclean_blocks,
    on=block_identifier_cols,
    how='left',
    indicator=True # Adds a column '_merge' to show which rows matched
)

# --- 5. Apply Corrections ---
def classify_voltage(row):
    # If the row was part of an unclean block (_merge == 'both'), re-classify it.
    # Otherwise, keep the original status.
    if row['_merge'] == 'both':
        voltage = row['voltage_28v_dc1_cal']
        if voltage < 2.2:
            return 'De-energized'
        elif 2.2 <= voltage < 22:
            return 'Stabilizing'
        else: # voltage >= 22
            return 'Steady State'
    else:
        return row['dc1_status']

# Apply the function row-by-row. Dask optimizes this across partitions.
# We must provide 'meta' to tell Dask the output type.
df_merged['Cleaned_Status'] = df_merged.apply(
    classify_voltage,
    axis=1,
    meta=('Cleaned_Status', 'str')
)

# --- 6. Get the Final Result ---
print("Starting computation... this may take a while.")
# Select only the columns we need before the final compute
final_cols_to_keep = group_cols + ['timestamp', 'voltage_28v_dc1_cal', 'dc1_status', 'Cleaned_Status']
final_tasks = df_merged[final_cols_to_keep]
final_df = final_tasks.compute()
print("Computation finished.")

# --- 7. Summarize Changes ---
print("\n" + "="*35)
print("        Cleaning Summary")
print("="*35)
changes_mask = final_df['dc1_status'] != final_df['Cleaned_Status']
changed_rows = final_df[changes_mask]

if changed_rows.empty:
    print("No classifications were changed.")
else:
    change_counts = changed_rows.groupby(['dc1_status', 'Cleaned_Status']).size().reset_index(name='count')
    change_counts.rename(columns={'dc1_status': 'Original State', 'Cleaned_Status': 'New State'}, inplace=True)
    print("Breakdown of classification changes:")
    print(change_counts.to_string(index=False))
    print(f"\nTotal classifications changed: {len(changed_rows)}")
print("="*35)