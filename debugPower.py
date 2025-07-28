import dask.dataframe as dd
import pandas as pd

# --- Step 1: VERIFY YOUR SETUP ---
# Please double-check that this list is 100% correct and does NOT contain 'cert_id'.
group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft'] 

# Your file paths
file_list = ['path/to/p1.parquet', 'path/to/p2.parquet']


# --- Step 2: READ DATA AND PRINT INFO ---
# This section will print the ground truth of what's in your files and what you're trying to group by.
print("--- DEBUGGING INFORMATION ---")
try:
    df = dd.read_parquet(file_list)
    
    print("\n1. Columns Dask found in your Parquet files:")
    print(list(df.columns))
    
    print("\n2. Columns being used for grouping:")
    print(group_cols)
    
    # --- Step 3: ATTEMPT THE GROUPBY OPERATION ---
    # This is the simplest possible groupby. If this fails, the problem is confirmed
    # to be a mismatch between the columns in the file and the group_cols list.
    print("\n3. Attempting the groupby operation...")
    
    # We are isolating the groupby to prove it's the source of the error.
    result = df.groupby(group_cols).head(1)
    result.compute() # .compute() is what triggers the actual work and the error
    
    print("\nSUCCESS: The groupby operation worked with the provided columns.")

except KeyError as e:
    print("\nERROR: The groupby operation failed as expected.")
    print("This confirms the error is happening because a column in your 'group_cols' list is not present in your data files.")
    print("Please carefully compare list #1 and list #2 above to find the typo or incorrect column name.")
    print(f"The exact column it failed to find is: {e}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

print("\n--- END OF DEBUG ---")