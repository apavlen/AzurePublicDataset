import pandas as pd
import argparse
import os
import requests
import glob
import gzip
import shutil
import matplotlib.pyplot as plt

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

def decompress_gz_files(input_dir: str, output_dir: str):
    """
    Decompress all .csv.gz files in input_dir to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    gz_files = glob.glob(os.path.join(input_dir, "*.csv.gz"))
    for gz_file in gz_files:
        out_file = os.path.join(output_dir, os.path.basename(gz_file)[:-3])
        print(f"Decompressing {gz_file} to {out_file}")
        with gzip.open(gz_file, 'rb') as f_in, open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def concatenate_csv_files(input_dir: str, output_csv: str):
    """
    Concatenate all .csv files in input_dir into a single CSV with header.
    """
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    with open(output_csv, 'w') as fout:
        for i, fname in enumerate(csv_files):
            with open(fname) as fin:
                if i == 0:
                    # Write header for the first file
                    fout.write(fin.read())
                else:
                    # Skip header for subsequent files
                    next(fin)
                    fout.write(fin.read())
    print(f"Concatenated {len(csv_files)} files into {output_csv}")

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
    # Try to infer the correct columns from the AzurePublicDataset schema
    # For 2019 trace, the columns are:
    # 1. Encrypted subscription id
    # 2. Encrypted deployment id 
    # 3. Timestamp in seconds (starting from 0) when first VM created
    # 4. Count VMs created
    # 5. Deployment size
    # 6. Encrypted VM id
    # 7. Timestamp VM created
    # 8. Timestamp VM deleted
    # 9. Max CPU utilization
    # 10. Avg CPU utilization
    # 11. P95 of Max CPU utilization
    # 12. VM category
    # 13. VM virtual core count bucket
    # 14. VM memory (GBs) bucket
    # 15. Timestamp in seconds (every 5 minutes)
    # 16. Min CPU utilization during the 5 minutes
    # 17. Max CPU utilization during the 5 minutes
    # 18. Avg CPU utilization during the 5 minutes
    # 19. VM virtual core count bucket definition
    # 20. VM memory (GBs) bucket definition

    # For 2017 trace, the columns are similar but with slight differences.

    # Read the raw data
    df = pd.read_csv(raw_data_path)

    # Try to detect the schema by number of columns
    if len(df.columns) >= 18:
        # 2019 or 2017 schema
        # Assign column names if not present
        if not all(isinstance(c, str) for c in df.columns):
            # Assign names for 2019
            df.columns = [
                "subscription_id", "deployment_id", "first_vm_ts", "count_vms_created", "deployment_size",
                "vm_id", "vm_created_ts", "vm_deleted_ts", "max_cpu", "avg_cpu", "p95_cpu",
                "vm_category", "vm_cores_bucket", "vm_mem_bucket", "reading_ts",
                "min_cpu_5min", "max_cpu_5min", "avg_cpu_5min", "core_bucket_def", "mem_bucket_def"
            ][:len(df.columns)]
        # Use the correct columns for time series
        # We'll use: reading_ts, vm_id, min_cpu_5min, max_cpu_5min, avg_cpu_5min
        columns = ["reading_ts", "vm_id", "min_cpu_5min", "max_cpu_5min", "avg_cpu_5min"]
        df = df[columns]
        # Rename for consistency
        df = df.rename(columns={
            "reading_ts": "timestamp",
            "min_cpu_5min": "CPU_min",
            "max_cpu_5min": "CPU_max",
            "avg_cpu_5min": "CPU_avg"
        })
    else:
        raise ValueError("Unknown schema: cannot find expected columns in raw data.")

    # Convert timestamp to datetime if possible (assume seconds since epoch or since trace start)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    except Exception:
        pass

    df = df.sort_values(['vm_id', 'timestamp'])

    # Handle missing values (forward fill, then drop remaining)
    df = df.fillna(method='ffill').dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove outliers for CPU columns
    for col in ['CPU_min', 'CPU_max', 'CPU_avg']:
        df = df[(df[col] >= 0) & (df[col] <= 100)]

    # Save cleaned time series
    df.to_csv(output_path, index=False)
    print(f"Cleaned time series saved to {output_path}")

def plot_timeseries(csv_path: str, vm_id: str = None, resource: str = "CPU", sample: int = 1000):
    """
    Plot a resource utilization time series for a given VM or all VMs.

    Args:
        csv_path (str): Path to the cleaned time series CSV.
        vm_id (str): VM ID to plot. If None, plot the first VM in the file.
        resource (str): Resource column to plot (e.g., "CPU").
        sample (int): Number of points to plot (for speed).
    """
    df = pd.read_csv(csv_path)
    if vm_id is None:
        vm_id = df['vm_id'].iloc[0]
    df_vm = df[df['vm_id'] == vm_id]
    if df_vm.empty:
        raise ValueError(f"No data found for vm_id={vm_id}")
    plt.figure(figsize=(12, 5))
    plt.plot(df_vm['timestamp'][:sample], df_vm[resource][:sample])
    plt.xlabel("Timestamp")
    plt.ylabel(f"{resource} Utilization")
    plt.title(f"{resource} Utilization for VM {vm_id}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, decompress, concatenate, and prepare a time series dataset for resource utilization.")
    parser.add_argument("--download_url", type=str, default=None, help="URL to download a .csv.gz file")
    parser.add_argument("--download_dir", type=str, default="data/raw_gz", help="Directory to store downloaded .csv.gz files")
    parser.add_argument("--decompressed_dir", type=str, default="data/raw_csv", help="Directory to store decompressed .csv files")
    parser.add_argument("--concat_csv", type=str, default="data/combined_raw.csv", help="Path to concatenated raw CSV")
    parser.add_argument("--output", type=str, default="data/cleaned_resource_timeseries.csv", help="Path to save cleaned time series CSV")
    parser.add_argument("--plot", action="store_true", help="Plot a sample timeseries after processing")
    parser.add_argument("--vm_id", type=str, default=None, help="VM ID to plot (optional)")
    parser.add_argument("--resource", type=str, default="CPU", help="Resource to plot (default: CPU)")
    args = parser.parse_args()

    # Step 1: Download a .csv.gz file if a URL is provided
    if args.download_url:
        os.makedirs(args.download_dir, exist_ok=True)
        filename = os.path.basename(args.download_url)
        dest_path = os.path.join(args.download_dir, filename)
        download_file(args.download_url, dest_path)

    # Step 2: Decompress all .csv.gz files in download_dir
    decompress_gz_files(args.download_dir, args.decompressed_dir)

    # Step 3: Concatenate all .csv files in decompressed_dir
    concatenate_csv_files(args.decompressed_dir, args.concat_csv)

    # Step 4: Prepare and clean the time series
    prepare_timeseries(
        raw_data_path=args.concat_csv,
        output_path=args.output
    )

    # Step 5: Optionally plot a timeseries
    if args.plot:
        plot_timeseries(args.output, vm_id=args.vm_id, resource=args.resource)
