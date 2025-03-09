import os
import glob
import re
import pandas as pd


def format_season(x):
    """
    Force the season code to be a 4-digit string with leading zeros.
    """
    try:
        # Convert to int and then format as 4-digit with leading zeros
        return "{:04d}".format(int(x))
    except Exception:
        s = str(x).strip()
        # If numeric string and less than 4 digits, pad with leading zeros
        if s.isdigit() and len(s) < 4:
            return "{:0>4}".format(s)
        return s


def merge_processed_files_common():
    """
    Merge all processed CSV files from the processed folder.
    Steps:
    1. Load each CSV file and print the number of data rows (excluding header).
    2. For each file, extract the season code from the filename using a regex pattern
       (e.g., "E0_0001_processed.csv" -> "0001"), and insert a new column "SeasonSource"
       as the first column, ensuring it is stored as a 4-digit string.
    3. Remove the original "Div" column if it exists.
    4. Determine the common columns across all files.
    5. Subset each DataFrame to the common columns.
    6. Vertically merge (concatenate) all DataFrames.
    7. Reorder columns so that the first columns are:
         SeasonSource, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Referee,
       and the remaining columns are sorted alphabetically.
    8. Convert the "Date" column to datetime (format "dd/mm/YYYY") and sort descending.
    9. Remove rows that are completely empty except for SeasonSource.
    10. Save the merged, reordered, and sorted DataFrame.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "../../data/processed")

    # Find all processed CSV files in the processed folder
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the processed directory.")
        return

    df_list = []
    columns_list = []
    file_row_counts = {}

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            file_name = os.path.basename(file)
            # Use regex to extract season code from filename; expected pattern: E0_XXXX
            match = re.search(r"E0_(\d{4})", file_name)
            if match:
                season_code = format_season(match.group(1))
            else:
                season_code = "unknown"
            # Insert the new column "SeasonSource" at the beginning
            df.insert(0, "SeasonSource", season_code)
            # Ensure SeasonSource is string
            df["SeasonSource"] = df["SeasonSource"].astype(str)
            # Remove the original "Div" column if it exists
            if "Div" in df.columns:
                df.drop(columns=["Div"], inplace=True)
            df_list.append(df)
            columns_list.append(set(df.columns))
            row_count = df.shape[0]
            file_row_counts[file_name] = row_count
            print(f"Loaded {file_name} with {row_count} data rows.")
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    print("\nRow counts for each processed file (excluding header):")
    for fname, count in file_row_counts.items():
        print(f"{fname}: {count} rows")

    # Determine common columns across all files (intersection)
    common_columns = set.intersection(*columns_list)
    common_columns = list(common_columns)
    print("\nCommon columns across all files:")
    print(common_columns)

    # Subset each DataFrame to the common columns
    df_list_common = [df[common_columns] for df in df_list]

    # Merge all DataFrames vertically (row-wise)
    merged_df = pd.concat(df_list_common, ignore_index=True)
    print("\nMerged DataFrame shape before reordering and sorting:", merged_df.shape)

    # Define the desired primary column order
    primary_cols = ['SeasonSource', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Referee']
    # Determine remaining columns from the common columns
    remaining_cols = [col for col in common_columns if col not in primary_cols]
    remaining_cols_sorted = sorted(remaining_cols)
    final_order = primary_cols + remaining_cols_sorted
    # Only keep columns that exist in merged_df
    final_order = [col for col in final_order if col in merged_df.columns]
    merged_df = merged_df[final_order]

    # Convert the 'Date' column to datetime using the expected format "dd/mm/YYYY"
    merged_df['Date_dt'] = pd.to_datetime(merged_df['Date'], format="%d/%m/%Y", errors="coerce")
    # Sort descending by Date (newest first)
    merged_df.sort_values(by='Date_dt', ascending=False, inplace=True)
    merged_df.drop(columns=['Date_dt'], inplace=True)

    # Drop rows that are completely empty except for SeasonSource
    cols_to_check = [col for col in merged_df.columns if col != "SeasonSource"]
    merged_df = merged_df.dropna(subset=cols_to_check, how='all')

    merged_file_path = os.path.join(processed_dir, "merged_E0_common_sorted.csv")
    merged_df.to_csv(merged_file_path, index=False)
    print("\nMerged, reordered, and sorted data saved to", merged_file_path)


if __name__ == "__main__":
    merge_processed_files_common()
