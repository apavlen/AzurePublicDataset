import pandas as pd
import argparse
import os
import requests

def download_file(url: str, dest_path: str, chunk_size: int = 8192):
    """
    Download a file from a URL to a local destination.
    """
    print(f"Downloading {url} to {dest_path} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print(f"Download complete: {dest_path}")

def prepare_timeseries(
    raw_data_path: str,
    output_path: str,
    columns: list = None,
    outlier_bounds: dict = None
):
    """
    Prepare and clean a time series dataset for resource utilization.

    Args:
        raw_data_path (str): Path to the raw CSV dataset.
        output_path (str): Path to save the cleaned time series CSV.
        columns (list): List of columns to extract and clean.
        outlier_bounds (dict): Dict of {col: (min, max)} for outlier removal.
    """
    if columns is None:
        columns = ['timestamp', 'vm_id', 'CPU', 'Memory', 'IO', 'Network', 'Disk']
    if outlier_bounds is None:
        outlier_bounds = {col: (0, 100) for col in columns if col not in ['timestamp', 'vm_id']}

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    df = pd.read_csv(raw_data_path)

    # Select relevant columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in raw data: {missing_cols}")
    df = df[columns]

    # Convert timestamp to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['vm_id', 'timestamp'])

    # Handle missing values (forward fill, then drop remaining)
    df = df.fillna(method='ffill').dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove outliers
    for col, (min_val, max_val) in outlier_bounds.items():
        df = df[(df[col] >= min_val) & (df[col] <= max_val)]

    # Save cleaned time series
    df.to_csv(output_path, index=False)
    print(f"Cleaned time series saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare a time series dataset for resource utilization.")
    parser.add_argument("--download_url", type=str, default=None, help="URL to download the raw resource data CSV")
    parser.add_argument("--raw_data", type=str, default="data/raw_resource_data.csv", help="Path to raw resource data CSV")
    parser.add_argument("--output", type=str, default="data/cleaned_resource_timeseries.csv", help="Path to save cleaned time series CSV")
    args = parser.parse_args()

    # Step 1: Download the raw data if a URL is provided
    if args.download_url:
        download_file(args.download_url, args.raw_data)

    # Step 2: Prepare and clean the time series
    prepare_timeseries(
        raw_data_path=args.raw_data,
        output_path=args.output
    )
