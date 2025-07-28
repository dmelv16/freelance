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
# Your specific columns from power2.py that define a unique test setup
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']

# Remove transient data first
values_to_remove = ['Error 0 volt', 'Error missing volt', 'Normal transient', 'Abnormal transient']
df_cleaned = df[~df['dc1_trans'].isin(values_to_remove)].copy()


# --- 3. Identify Misclassified Groups WITHIN Your Full Setup ---
# First, group by YOUR full list of columns.
# Then, find the contiguous state blocks inside each of those unique test groups.
df_cleaned['block_id'] = df_cleaned.groupby(group_cols)['dc1_status'].transform(lambda x: (x != x.shift()).cumsum())

# Create a globally unique ID for each block to simplify final grouping.
df_cleaned['unique_block_id'] = df_cleaned.groupby(group_cols + ['block_id']).ngroup()

# Aggregate info for each unique block to check for rule violations.
group_info = df_cleaned.groupby('unique_block_id').agg(
    min_voltage=('voltage_28v_dc1_cal', 'min'),
    max_voltage=('voltage_28v_dc1_cal', 'max'),
    state=('dc1_status', 'first')
)

# FLAG 1: Find 'Steady State' blocks with minimums below 22V.
unclean_steady_ids = group_info[
    (group_info['state'] == 'Steady State') & (group_info['min_voltage'] < 22)
].index

# FLAG 2: Find 'Stabilizing' blocks with maximums >= 22V or minimums < 2.2V.
unclean_stabilizing_ids = group_info[
    (group_info['state'] == 'Stabilizing') &
    ((group_info['max_voltage'] >= 22) | (group_info['min_voltage'] < 2.2))
].index

# Combine all unclean block IDs
all_unclean_ids = unclean_steady_ids.union(unclean_stabilizing_ids)
print(f"\nIdentified unclean block IDs to fix: {all_unclean_ids.tolist()}")


# --- 4. Correct ALL Misclassified Data ---
# Initialize the new column with the original data.
df_cleaned['Cleaned_Status'] = df_cleaned['dc1_status']

# Define the simple voltage-based classification rules.
def classify_voltage(voltage):
    if voltage < 2.2:
        return 'De-energized'
    elif 2.2 <= voltage < 22:
        return 'Stabilizing'
    else: # voltage >= 22
        return 'Steady State'

# Isolate all rows belonging to any unclean block.
indices_to_fix = df_cleaned[df_cleaned['unique_block_id'].isin(all_unclean_ids)].index

# Apply the correction ONLY to those specific rows.
df_cleaned.loc[indices_to_fix, 'Cleaned_Status'] = df_cleaned.loc[indices_to_fix, 'voltage_28v_dc1_cal'].apply(classify_voltage)


# --- 5. Final Result ---
print("\n--- Final Cleaned Data ---")
# Select relevant columns for final display.
final_cols = group_cols + ['timestamp', 'voltage_28v_dc1_cal', 'dc1_status', 'Cleaned_Status']
print(df_cleaned[final_cols])