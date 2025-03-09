import os
import glob
import pandas as pd


def analyze_processed_columns():
    """
    Analyze and print the column names for each CSV file in the processed directory.
    """
    # Determine the base directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the processed data directory
    processed_dir = os.path.join(base_dir, "../../data/processed")

    # Use glob to find all CSV files in the processed directory
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the processed directory.")
        return {}

    file_columns = {}
    for file in csv_files:
        try:
            # Read only the header of the CSV file
            df = pd.read_csv(file, nrows=0)
            columns = list(df.columns)
            file_name = os.path.basename(file)
            file_columns[file_name] = columns
            print(f"File: {file_name}")
            print("Columns:", columns)
            print("-" * 40)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    return file_columns


if __name__ == "__main__":
    analyze_processed_columns()
