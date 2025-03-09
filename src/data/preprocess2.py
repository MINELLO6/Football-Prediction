import os
import pandas as pd
import shutil
import time


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


def process_e1_file():
    """
    Process the E1 CSV file with proper file path handling and permission issues
    """
    # Use raw string (r prefix) for Windows paths to avoid escape sequence issues
    file_path = r"E:\FootballBayesianPrediction\pythonProject\data\processed\merged_E1.csv"

    # Create output path for new file instead of trying to overwrite
    output_path = r"E:\FootballBayesianPrediction\pythonProject\data\processed\merged_E1_updated.csv"

    try:
        print(f"Reading file: {file_path}")
        # Add low_memory=False to avoid DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)

        # Check if Date column exists
        if "Date" not in df.columns:
            print(f"Error: 'Date' column not found in the file")
            print(f"Available columns: {df.columns.tolist()}")
            return

        print(f"Original Date values from {file_path}:")
        print(df["Date"].head())

        # Overwrite the original Date column with standardized dates
        df["Date"] = df["Date"].apply(standardize_date)

        print(f"Overwritten Date values from {file_path}:")
        print(df["Date"].head())

        # Save to a new file instead of overwriting the original
        try:
            df.to_csv(output_path, index=False)
            print(f"Processed file saved to: {output_path}")

            # If you still want to replace the original after successfully saving,
            # wait a moment and then try to copy the new file over the original
            print("Waiting 2 seconds before attempting to replace original file...")
            time.sleep(2)

            try:
                # Make sure the file is closed first
                df = None  # Release the dataframe

                # Try to replace the original with the updated file
                shutil.copy2(output_path, file_path)
                print(f"Successfully replaced original file at: {file_path}")
            except Exception as e:
                print(f"Could not replace original file, but updated file is available at: {output_path}")
                print(f"Error: {e}")
        except Exception as e:
            print(f"Error saving to new file: {e}")

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    process_e1_file()