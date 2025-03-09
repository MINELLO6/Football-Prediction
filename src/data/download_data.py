import os
import pandas as pd


def read_csv_with_fallback(url):
    """
    Try to read a CSV file from a URL.
    First, try the default (utf-8) encoding.
    If a UnicodeDecodeError occurs, try using latin1 encoding.
    If a ParserError occurs, skip bad lines.
    """
    try:
        return pd.read_csv(url)
    except UnicodeDecodeError as e:
        print(f"Unicode error for {url}, trying latin1: {e}")
        return pd.read_csv(url, encoding="latin1")
    except pd.errors.ParserError as e:
        print(f"Parsing error for {url}, skipping bad lines: {e}")
        return pd.read_csv(url, on_bad_lines='skip')


def download_and_merge_data():
    # Define the directories for raw and processed data
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/raw')
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    dfs = []
    # Loop from 24 down to 0 (representing seasons from 2425 to 0001)
    for start in range(24, -1, -1):
        season_code = f"{start:02d}{(start + 1) % 100:02d}"  # e.g., "2425" for 2024-2025, "0001" for 2000-2001
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"  # 0405: invali byte, download manually
        print(f"Downloading data from: {url}")

        # Set the local path for the raw CSV file
        raw_file_path = os.path.join(raw_dir, f"E0_{season_code}.csv")

        # If the file already exists, read from the local file; otherwise, download it
        if os.path.exists(raw_file_path):
            print(f"File already exists: {raw_file_path}. Skipping download.")
            df = pd.read_csv(raw_file_path)
        else:
            try:
                df = read_csv_with_fallback(url)
                df.to_csv(raw_file_path, index=False)
            except Exception as e:
                print(f"Failed to read {url}: {e}")
                continue

        # Add a new column for the season code
        df["Season"] = season_code
        dfs.append(df)

    if dfs:
        # Concatenate all DataFrames into one large DataFrame
        merged_df = pd.concat(dfs, ignore_index=True)
        print("Merged DataFrame shape:", merged_df.shape)
        # Save the merged DataFrame to a CSV file in the processed directory
        merged_file_path = os.path.join(processed_dir, "merged_E0.csv")
        merged_df.to_csv(merged_file_path, index=False)
        print("Merged data saved to", merged_file_path)
    else:
        print("No data was successfully downloaded.")


if __name__ == "__main__":
    download_and_merge_data()
