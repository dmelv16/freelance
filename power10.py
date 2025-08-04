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

# 4. The list of columns that define a unique group
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']

# --- Main Script ---
print(f"Step 1: Loading the main Parquet file: {input_parquet_path}")
try:
    df = pd.read_parquet(input_parquet_path)
    print(f"Successfully loaded {len(df)} rows.")
except Exception as e:
    print(f"Error: Could not read input Parquet file. {e}")
    exit()

print(f"\nStep 2: Loading exclusion list from '{exclusion_file_path}'...")
try:
    df_to_exclude = pd.read_csv(exclusion_file_path)
    print(f"Found {len(df_to_exclude)} groups to exclude.")
except FileNotFoundError:
    print(f"Error: Exclusion file not found at '{exclusion_file_path}'. Cannot filter.")
    exit()
except Exception as e:
    print(f"Error loading exclusion file: {e}")
    exit()

# --- Create a reliable "fingerprint" to handle data type differences ---
print("\nStep 3: Creating robust fingerprints for filtering...")

# Create fingerprint for the main data
# Convert all group columns to string to avoid type mismatches
for col in group_cols:
    df[col] = df[col].astype(str)
df['fingerprint'] = df[group_cols].agg('|'.join, axis=1)

# Create fingerprint for the exclusion data
for col in group_cols:
    df_to_exclude[col] = df_to_exclude[col].astype(str)
df_to_exclude['fingerprint'] = df_to_exclude[group_cols].agg('|'.join, axis=1)

# Get the set of fingerprints to remove
fingerprints_to_remove = set(df_to_exclude['fingerprint'])

# --- Filter the DataFrame ---
print(f"\nStep 4: Filtering out {len(fingerprints_to_remove)} groups...")
initial_rows = len(df)
# The '~' inverts the boolean mask, keeping everything NOT in the removal set
df_filtered = df[~df['fingerprint'].isin(fingerprints_to_remove)].copy()
final_rows = len(df_filtered)

# Clean up the temporary fingerprint column
df_filtered.drop(columns=['fingerprint'], inplace=True)

print(f"Filtering complete. Removed {initial_rows - final_rows} rows. {final_rows} rows remaining.")

# --- Save the new, filtered Parquet file ---
print(f"\nStep 5: Saving filtered data to '{output_parquet_path}'...")
try:
    df_filtered.to_parquet(output_parquet_path, index=False)
    print("✅ Successfully saved the filtered Parquet file.")
except Exception as e:
    print(f"Error saving filtered file: {e}")
