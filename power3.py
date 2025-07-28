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

# THIS IS THE CORRECTED LINE:
# The .copy() is removed. This operation now creates a new Dask DataFrame 'df_cleaned'.
df_cleaned = df[~df['dc1_trans'].isin(values_to_remove)]

# --- 3. Identify Misclassified Groups WITHIN Your Full Setup ---
df_cleaned['block_id'] = df_cleaned.groupby(group_cols)['dc1_status'].transform(
    lambda x: (x != x.shift()).fillna(True).astype(int).cumsum(),
    meta=('block_id', 'int64')
)
df_cleaned['unique_block_id'] = df_cleaned.groupby(group_cols + ['block_id']).ngroup()

group_info = df_cleaned.groupby('unique_block_id').agg(
    min_voltage=('voltage_28v_dc1_cal', 'min'),
    max_voltage=('voltage_28v_dc1_cal', 'max'),
    state=('dc1_status', 'first')
).compute()

unclean_steady_ids = group_info[
    (group_info['state'] == 'Steady State') & (group_info['min_voltage'] < 22)
].index
unclean_stabilizing_ids = group_info[
    (group_info['state'] == 'Stabilizing') &
    ((group_info['max_voltage'] >= 22) | (group_info['min_voltage'] < 2.2))
].index
all_unclean_ids = unclean_steady_ids.union(unclean_stabilizing_ids).tolist()

# --- 4. Correct ALL Misclassified Data ---
def classify_voltage(voltage):
    if voltage < 2.2:
        return 'De-energized'
    elif 2.2 <= voltage < 22:
        return 'Stabilizing'
    else:
        return 'Steady State'

df_cleaned['Cleaned_Status'] = df_cleaned.map_partitions(
    lambda pdf: pdf['dc1_status'].where(
        ~pdf['unique_block_id'].isin(all_unclean_ids),
        pdf['voltage_28v_dc1_cal'].apply(classify_voltage)
    ),
    meta=('Cleaned_Status', 'str')
)

# --- 5. Get the Final Result ---
print("Starting computation... this may take a while.")
final_df = df_cleaned.compute()
print("Computation finished.")

# --- 6. Summarize Changes (on the final pandas DataFrame) ---
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