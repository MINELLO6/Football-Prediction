import os
import glob
import pandas as pd

def standardize_date(date_str):
    """
    Standardize a date string according to the following rules:
    1. Split the string by '/' into three parts: part1, part2, part3.
    2. If part1 has 2 characters and part3 has 2 characters, assume the format is "dd/mm/yy".
       Prepend "20" to part3 to form "dd/mm/YYYY".
    3. If part1 has 4 characters, then assume the date is mis-formatted:
       - Replace part1 with its last two characters (the intended day).
       - For part3, if its length is less than 4, pad it by prepending "2" and left-zero-fill to make it 4 characters.
         For example, "2025/12/1" becomes "25/12/2001".
    4. If parts are 2, 2, 4 (already "dd/mm/YYYY"), do nothing.
    5. Finally, pad day and month to 2 digits.
    Returns a standardized date string in the format "dd/mm/YYYY".
    """
    # Check if the input is a string; if not, return it as is
    if not isinstance(date_str, str):
        return date_str

    date_str = date_str.strip()
    parts = date_str.split('/')
    if len(parts) != 3:
        return date_str  # fallback: return original if not 3 parts

    p1, p2, p3 = parts[0], parts[1], parts[2]

    # Case 1: Format is "dd/mm/yy" (both day and year parts are 2 digits)
    if len(p1) == 2 and len(p3) == 2:
        new_year = "20" + p3
        day = p1.zfill(2)
        month = p2.zfill(2)
        return f"{day}/{month}/{new_year}"

    # Case 2: Mis-formatted where part1 has 4 digits.
    if len(p1) == 4:
        try:
            first_val = int(p1)
        except Exception:
            return pd.NaT
        if first_val > 31:
            new_day = p1[-2:]  # take last two digits as day
            if len(p3) < 4:
                new_year = "2" + p3.zfill(3)
            else:
                new_year = p3
            day = new_day.zfill(2)
            month = p2.zfill(2)
            return f"{day}/{month}/{new_year}"

    # Case 3: Already in correct "dd/mm/YYYY" format.
    if len(p1) == 2 and len(p3) == 4:
        day = p1.zfill(2)
        month = p2.zfill(2)
        return f"{day}/{month}/{p3}"

    # Fallback: use default parsing with dayfirst=True and then format to "dd/mm/YYYY"
    try:
        dt = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
        return dt.strftime("%d/%m/%Y") if pd.notnull(dt) else date_str
    except Exception:
        return date_str

def process_csv_file(raw_file_path, processed_file_path, date_column="Date"):
    """
    Process a single CSV file:
    - Read the CSV file from raw_file_path.
    - If a UnicodeDecodeError occurs, try reading with encoding "latin1".
    - Apply standardize_date to the specified date column and overwrite the original column.
    - Print debug information for the first few rows.
    - Save the processed DataFrame to processed_file_path.
    """
    try:
        df = pd.read_csv(raw_file_path)
    except UnicodeDecodeError as e:
        print(f"Unicode error for {raw_file_path}, trying latin1: {e}")
        try:
            df = pd.read_csv(raw_file_path, encoding="latin1")
        except Exception as e2:
            print(f"Failed to read {raw_file_path} with latin1: {e2}")
            return

    print(f"Original {date_column} values from {raw_file_path}:")
    print(df[date_column].head())

    # Overwrite the original date column with standardized dates
    df[date_column] = df[date_column].apply(standardize_date)

    print(f"Overwritten {date_column} values from {raw_file_path}:")
    print(df[date_column].head())

    try:
        df.to_csv(processed_file_path, index=False)
        print(f"Processed file saved to: {processed_file_path}\n")
    except Exception as e:
        print(f"Error saving {processed_file_path}: {e}")

def batch_process_raw_files():
    """
    Batch process all CSV files in the raw data folder:
    - Reads each CSV file from the raw folder.
    - Processes each file to standardize the date column by overwriting it.
    - Saves each processed file into the processed folder with a '_processed' suffix.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "../../data/raw")
    processed_dir = os.path.join(base_dir, "../../data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found in the raw directory.")
        return

    for raw_file in csv_files:
        filename = os.path.basename(raw_file)
        name, ext = os.path.splitext(filename)
        processed_file = os.path.join(processed_dir, f"{name}_processed{ext}")
        print(f"Processing file: {raw_file}")
        process_csv_file(raw_file, processed_file)

if __name__ == "__main__":
    batch_process_raw_files()
