# filter_parquet.py
import pandas as pd
import pyarrow.parquet as pq
import os

# --- Configuration ---
# 1. The large, unfiltered Parquet file created by your main script
input_parquet_path = 'output_data_cleaned/cleaned_data_final.parquet'

# 2. The CSV file containing the exact groups you want to remove
exclusion_file_path = 'groups_to_exclude.csv'

# 3. The name of the final, filtered output file
output_parquet_path = 'output_data_cleaned/final_filtered_data.parquet'

# 4. The columns that define a unique group
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']

# 5. How many rows to process at a time (adjust based on your RAM)
BATCH_SIZE = 500_000

# --- Step 1: Load the Exclusion List and Create Fingerprints ---
print(f"Step 1: Loading exclusion list from '{exclusion_file_path}'...")
try:
    df_to_exclude = pd.read_csv(exclusion_file_path)
    # Create the robust fingerprint for the exclusion file
    fingerprint_cols = []
    for col in group_cols:
        if pd.api.types.is_numeric_dtype(df_to_exclude[col]):
            fingerprint_cols.append(df_to_exclude[col].map('{:.6f}'.format).str.rstrip('0').str.rstrip('.'))
        else:
            fingerprint_cols.append(df_to_exclude[col].astype(str))
    df_to_exclude['fingerprint'] = pd.concat(fingerprint_cols, axis=1).agg('|'.join, axis=1)
    fingerprints_to_remove = set(df_to_exclude['fingerprint'])
    print(f"Loaded {len(fingerprints_to_remove)} unique group fingerprints to exclude.")
except FileNotFoundError:
    print(f"Error: Exclusion file not found at '{exclusion_file_path}'. Cannot filter.")
    exit()
except Exception as e:
    print(f"Error loading exclusion file: {e}")
    exit()

# --- Step 2: Process the Large Parquet File in Chunks ---
print(f"\nStep 2: Processing '{input_parquet_path}' in chunks...")

# Open the source Parquet file
try:
    parquet_file = pq.ParquetFile(input_parquet_path)
except Exception as e:
    print(f"Error opening input Parquet file: {e}")
    exit()

# Use the schema from the source file for the new, filtered file
schema = parquet_file.schema

# Open a writer for the new, filtered Parquet file
with pq.ParquetWriter(output_parquet_path, schema=schema) as writer:
    # Use iter_batches to read the file in memory-efficient chunks
    total_rows_processed = 0
    total_rows_written = 0
    
    # Create a progress bar with the total number of row groups
    for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE), total=parquet_file.num_row_groups):
        # Convert the chunk (batch) to a Pandas DataFrame
        chunk_df = batch.to_pandas()
        total_rows_processed += len(chunk_df)

        # Create the fingerprint for the current chunk
        fingerprint_cols_chunk = []
        for col in group_cols:
            if pd.api.types.is_numeric_dtype(chunk_df[col]):
                fingerprint_cols_chunk.append(chunk_df[col].map('{:.6f}'.format).str.rstrip('0').str.rstrip('.'))
            else:
                fingerprint_cols_chunk.append(chunk_df[col].astype(str))
        chunk_df['fingerprint'] = pd.concat(fingerprint_cols_chunk, axis=1).agg('|'.join, axis=1)
        
        # Filter the chunk
        # The '~' inverts the mask, keeping rows NOT in the removal set
        filtered_chunk = chunk_df[~chunk_df['fingerprint'].isin(fingerprints_to_remove)]

        # If there's anything left after filtering, write it to the new file
        if not filtered_chunk.empty:
            # Remove the temporary fingerprint column
            filtered_chunk = filtered_chunk.drop(columns=['fingerprint'])
            total_rows_written += len(filtered_chunk)
            # Convert the filtered chunk back to a PyArrow Table
            table = pa.Table.from_pandas(filtered_chunk, schema=schema, preserve_index=False)
            # Write the filtered table to the output file
            writer.write_table(table)

print("\n--- Processing Complete ---")
print(f"Total rows read: {total_rows_processed}")
print(f"Total rows written (after filtering): {total_rows_written}")
print(f"✅ Successfully saved filtered data to '{output_parquet_path}'")
