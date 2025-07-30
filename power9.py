import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from tqdm import tqdm
import os

# --- 1. CLEANING FUNCTION ---
# MODIFIED: to accept a configurable voltage threshold.
def clean_one_group(group_df, steady_voltage_threshold=22):
    """
    Cleans data for a single group, classifying each row based on voltage and stability.
    
    Args:
        group_df (pd.DataFrame): The input dataframe for a single group.
        steady_voltage_threshold (float): The voltage level to consider as 'steady'.
    """
    STABILITY_WINDOW = 5
    STABILITY_THRESHOLD = 0.1

    df = group_df.copy()
    df['voltage_28v_dc1_cal'] = pd.to_numeric(df['voltage_28v_dc1_cal'], errors='coerce')
    df['voltage_std'] = df['voltage_28v_dc1_cal'].rolling(window=STABILITY_WINDOW).std()

    def classify_with_stability(row):
        voltage = row['voltage_28v_dc1_cal']
        stability = row['voltage_std']
        
        if pd.isna(voltage) or voltage < 2.2:
            return 'De-energized'
        
        if voltage >= steady_voltage_threshold and not pd.isna(stability) and stability < STABILITY_THRESHOLD:
            return 'Steady State'
        
        return 'Stabilizing'

    df['Cleaned_Status'] = df.apply(classify_with_stability, axis=1)

    steady_indices = df.index[df['Cleaned_Status'] == 'Steady State']
    if not steady_indices.empty:
        first_steady_index, last_steady_index = steady_indices[0], steady_indices[-1]
        is_stabilizing = df['Cleaned_Status'] == 'Stabilizing'
        is_between = (df.index > first_steady_index) & (df.index < last_steady_index)
        df.loc[is_stabilizing & is_between, 'Cleaned_Status'] = 'Steady State'

    df['dc1_status'] = df['Cleaned_Status']
    # Return the final DataFrame without the temporary columns
    return df.drop(columns=['Cleaned_Status', 'voltage_std'])

# --- 2. SETUP ---
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
output_directory = 'output_data_cleaned'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")

# --- 3. NEW: Configuration for Automatic Detection ---
try:
    OFP_COLUMN_INDEX = group_cols.index('ofp')
except ValueError:
    raise ValueError("'ofp' not found in group_cols. Please add it.")

DEFAULT_VOLTAGE_THRESHOLD = 22.0
ANALYSIS_VOLTAGE_FLOOR = 15.0
THRESHOLD_BUFFER = 2.0
HISTOGRAM_BINS = 50

# --- 4. Read Files and Gather Groups (Pass 1) ---
groups_data = defaultdict(list)
print("Step 1: Reading files and gathering all groups...")
for file_path in file_list:
    columns_to_read = list(set(['voltage_28v_dc1_cal', 'timestamp'] + group_cols))
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=500_000, columns=columns_to_read):
        chunk = batch.to_pandas()
        for group_key, group_df in chunk.groupby(group_cols):
            groups_data[group_key].append(group_df)
print(f"✅ Finished gathering data for {len(groups_data)} unique groups.")

# --- 5. Analyze Voltage Distributions (Pass 2) ---
print("\nStep 2: Analyzing voltage distributions to auto-detect thresholds...")
ofp_thresholds = {}
ofp_voltages = defaultdict(list)

for group_key, df_chunks in groups_data.items():
    ofp_id = group_key[OFP_COLUMN_INDEX]
    for chunk in df_chunks:
        ofp_voltages[ofp_id].extend(chunk['voltage_28v_dc1_cal'].dropna().tolist())

for ofp_id, voltages in ofp_voltages.items():
    voltages_np = np.array(voltages)
    steady_state_candidates = voltages_np[voltages_np > ANALYSIS_VOLTAGE_FLOOR]

    if steady_state_candidates.size < 100:
        ofp_thresholds[ofp_id] = DEFAULT_VOLTAGE_THRESHOLD
        print(f"  - OFP '{ofp_id}': Not enough data. Using default threshold ({DEFAULT_VOLTAGE_THRESHOLD}V).")
        continue

    counts, bin_edges = np.histogram(steady_state_candidates, bins=HISTOGRAM_BINS)
    modal_bin_index = np.argmax(counts)
    steady_voltage_estimate = (bin_edges[modal_bin_index] + bin_edges[modal_bin_index + 1]) / 2
    
    calculated_threshold = steady_voltage_estimate - THRESHOLD_BUFFER
    ofp_thresholds[ofp_id] = calculated_threshold
    print(f"  - OFP '{ofp_id}': Detected steady voltage ~{steady_voltage_estimate:.1f}V. Set threshold to {calculated_threshold:.1f}V.")

# --- 6. NEW: Define Output Schema and Filename ---
output_filename = os.path.join(output_directory, 'cleaned_data_final.parquet')
# Discover schema from the first file and add our new column
base_schema = pq.read_schema(file_list[0])
master_column_names = base_schema.names
if 'dc1_status' not in master_column_names:
    master_column_names.append('dc1_status')

# Create the final schema object for the writer
fields_to_keep = [field for field in base_schema if field.name in master_column_names]
if 'dc1_status' not in [f.name for f in fields_to_keep]:
    fields_to_keep.append(pa.field('dc1_status', pa.string()))
master_schema = pa.schema(fields_to_keep)


# --- 7. MODIFIED: Clean each group and write to file iteratively (Pass 3) ---
parquet_writer = None
print(f"\nStep 3: Cleaning each group and writing to '{output_filename}'...")
for group_key, list_of_chunks in tqdm(groups_data.items()):
    try:
        # Get the correct threshold for this group's OFP
        ofp_id = group_key[OFP_COLUMN_INDEX]
        threshold_to_use = ofp_thresholds.get(ofp_id, DEFAULT_VOLTAGE_THRESHOLD)

        # Process the group with the correct threshold
        full_group_df = pd.concat(list_of_chunks).sort_values('timestamp').reset_index(drop=True)
        cleaned_group_df = clean_one_group(full_group_df, steady_voltage_threshold=threshold_to_use)
        
        # Add any missing columns from the master schema
        for col_name in master_schema.names:
            if col_name not in cleaned_group_df.columns:
                cleaned_group_df[col_name] = None
        
        # Ensure column order matches the master schema before creating the table
        cleaned_group_df = cleaned_group_df[master_schema.names]

        table = pa.Table.from_pandas(cleaned_group_df, schema=master_schema, preserve_index=False)
        
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_filename, master_schema)
        
        parquet_writer.write_table(table)

    except Exception as e:
        print(f"\nWARNING: Skipping group {group_key} due to a processing error: {e}")
        continue

if parquet_writer:
    parquet_writer.close()
    print(f"\n🎉 Successfully finished writing to {output_filename}")
else:
    print("\nNo data was processed or written.")
