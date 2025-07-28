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
df_cleaned['block_id'] = df_cleaned.groupby(group_cols)['dc1_status'].transform(
    lambda x: (x != x.shift()).fillna(True).astype(int).cumsum(),
    meta=('block_id', 'int64')
)
block_identifier_cols = group_cols + ['block_id']

# Perform the small aggregation to find which blocks are unclean
group_info = df_cleaned.groupby(block_identifier_cols).agg(
    min_voltage=('voltage_28v_dc1_cal', 'min'),
    max_voltage=('voltage_28v_dc1_cal', 'max'),
    state=('dc1_status', 'first')
).compute()

# Determine which groups are unclean
is_unclean_steady = (group_info['state'] == 'Steady State') & (group_info['min_voltage'] < 22)
is_unclean_stabilizing = (group_info['state'] == 'Stabilizing') & ((group_info['max_voltage'] >= 22) | (group_info['min_voltage'] < 2.2))
group_info['is_unclean'] = is_unclean_steady | is_unclean_stabilizing

# Get just the identifiers of the unclean blocks
unclean_block_identifiers = group_info[group_info['is_unclean']].reset_index()[block_identifier_cols]

# --- 4. Isolate Unclean Rows and Compute Corrections ---
# Merge to get a Dask DataFrame containing ONLY the rows that need to be fixed
rows_to_fix = dd.merge(
    df_cleaned,
    unclean_block_identifiers,
    on=block_identifier_cols,
    how='inner' # 'inner' join gives us only the matching (unclean) rows
)

# Now, compute this much smaller DataFrame of unclean rows. This should easily fit in memory.
unclean_rows_pd = rows_to_fix.compute()

# Define the classification function
def classify_voltage(voltage):
    if voltage < 2.2:
        return 'De-energized'
    elif 2.2 <= voltage < 22:
        return 'Stabilizing'
    else:
        return 'Steady State'

# On this smaller pandas DataFrame, calculate the correct status
unclean_rows_pd['Correction'] = unclean_rows_pd['voltage_28v_dc1_cal'].apply(classify_voltage)

# Create our final "fix-it" table. It only needs the original index to match rows, and the correction.
# We use the original DataFrame's index for a direct mapping.
fix_it_table = unclean_rows_pd[['Correction']].set_index(unclean_rows_pd.index)


# --- 5. Apply Corrections via Merge ---
# Merge the "fix-it" table back to the original Dask DataFrame.
# This joins the 'Correction' column to the rows that need it.
df_with_fixes = df_cleaned.merge(fix_it_table, how='left', left_index=True, right_index=True)

# Create the final 'Cleaned_Status' column.
# If a 'Correction' exists for a row, use it. Otherwise, use the original status.
df_with_fixes['Cleaned_Status'] = df_with_fixes['Correction'].fillna(df_with_fixes['dc1_status'])


# --- 6. Get the Final Result ---
print("Starting computation... this may take a while.")
final_cols_to_keep = group_cols + ['timestamp', 'voltage_28v_dc1_cal', 'dc1_status', 'Cleaned_Status']
final_tasks = df_with_fixes[final_cols_to_keep]
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