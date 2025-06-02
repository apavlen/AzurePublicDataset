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
    Download a file from a URL to a local destination, unless it already exists.
    """
    if os.path.exists(dest_path):
        print(f"File already exists, skipping download: {dest_path}")
        return
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
    Decompress all .csv.gz files in input_dir to output_dir, unless already decompressed.
    """
    os.makedirs(output_dir, exist_ok=True)
    gz_files = glob.glob(os.path.join(input_dir, "*.csv.gz"))
    for gz_file in gz_files:
        out_file = os.path.join(output_dir, os.path.basename(gz_file)[:-3])
        if os.path.exists(out_file):
            print(f"File already decompressed, skipping: {out_file}")
            continue
        print(f"Decompressing {gz_file} to {out_file}")
        with gzip.open(gz_file, 'rb') as f_in, open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def concatenate_csv_files(input_dir: str, output_csv: str):
    """
    Concatenate all .csv files in input_dir into a single CSV with header.
    If the files do not have a header, add a default header for AzurePublicDataset 2019 schema.
    If the number of columns in the header does not match the data, adjust the header to match the data columns.
    """
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    with open(output_csv, 'w') as fout:
        for i, fname in enumerate(csv_files):
            with open(fname) as fin:
                first_line = fin.readline()
                # Peek at the next line to count columns in data
                data_line = fin.readline()
                fin.seek(0)
                if i == 0:
                    # Check if file has a header
                    if not any(x.isalpha() for x in first_line.split(",")[0]):
                        # No header, add default header
                        data_cols = len(first_line.strip().split(","))
                        default_header = (
                            "subscription_id,deployment_id,first_vm_ts,count_vms_created,deployment_size,"
                            "vm_id,vm_created_ts,vm_deleted_ts,max_cpu,avg_cpu,p95_cpu,"
                            "vm_category,vm_cores_bucket,vm_mem_bucket,reading_ts,"
                            "min_cpu_5min,max_cpu_5min,avg_cpu_5min,core_bucket_def,mem_bucket_def"
                        ).split(",")
                        # Truncate or pad header to match data columns
                        if data_cols < len(default_header):
                            header = ",".join(default_header[:data_cols]) + "\n"
                        elif data_cols > len(default_header):
                            header = ",".join(default_header + [f"extra_col_{j+1}" for j in range(data_cols - len(default_header))]) + "\n"
                        else:
                            header = ",".join(default_header) + "\n"
                        fout.write(header)
                        fout.write(first_line)
                        fout.write(fin.read())
                    else:
                        # File has header, check if header matches data columns
                        header_cols = len(first_line.strip().split(","))
                        data_cols = len(data_line.strip().split(","))
                        if header_cols != data_cols:
                            # Adjust header to match data columns
                            header = first_line.strip().split(",")
                            if data_cols < header_cols:
                                header = header[:data_cols]
                            elif data_cols > header_cols:
                                header = header + [f"extra_col_{j+1}" for j in range(data_cols - header_cols)]
                            fout.write(",".join(header) + "\n")
                            # Write the first data line and the rest
                            fout.write(data_line)
                            fout.write(fin.read())
                        else:
                            fout.write(first_line)
                            fout.write(fin.read())
                else:
                    # Skip header for subsequent files
                    next(fin)
                    fout.write(fin.read())
    print(f"Concatenated {len(csv_files)} files into {output_csv}")
    # Print header of the generated CSV
    with open(output_csv, 'r') as fcheck:
        header = fcheck.readline().strip()
        print(f"Header of generated CSV: {header}")

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

    # Check for header/data mismatch and fix if needed
    if len(df) > 0:
        header_cols = list(df.columns)
        first_row = df.iloc[0]
        if len(header_cols) > len(first_row):
            # Drop extra header columns
            df = df[header_cols[:len(first_row)]]
        elif len(header_cols) < len(first_row):
            # Add generic column names for extra columns
            for j in range(len(first_row) - len(header_cols)):
                df[f"extra_col_{j+1}"] = None

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
        # Keep all columns, but create new columns for plotting
        if 'reading_ts' in df.columns:
            df['timestamp'] = df['reading_ts']
        if 'avg_cpu_5min' in df.columns:
            df['CPU'] = df['avg_cpu_5min']
        if 'max_cpu_5min' in df.columns:
            df['CPU_max'] = df['max_cpu_5min']
        if 'min_cpu_5min' in df.columns:
            df['CPU_min'] = df['min_cpu_5min']
    else:
        raise ValueError("Unknown schema: cannot find expected columns in raw data.")

    # Convert timestamp to datetime if possible (assume seconds since epoch or since trace start)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        except Exception:
            pass

    # Only sort and clean if the required columns exist
    if 'vm_id' in df.columns and 'timestamp' in df.columns:
        df = df.sort_values(['vm_id', 'timestamp'])

    # Handle missing values (forward fill, then drop remaining)
    df = df.ffill().dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove outliers for CPU columns if present
    for col in ['CPU', 'CPU_max', 'CPU_min']:
        if col in df.columns:
            df = df[(df[col] >= 0) & (df[col] <= 100)]

    # Save cleaned time series
    df.to_csv(output_path, index=False)
    print(f"Cleaned time series saved to {output_path}")
    # Print header and a few rows for inspection
    print("Sample of cleaned CSV:")
    print(df.head())
    # Check if generated CSV is per VM
    if 'vm_id' in df.columns:
        unique_vms = df['vm_id'].nunique()
        print(f"Number of unique VMs in cleaned CSV: {unique_vms}")
        print("Rows per VM (first 5 VMs):")
        for vm in df['vm_id'].unique()[:5]:
            count = (df['vm_id'] == vm).sum()
            print(f"  VM {vm}: {count} rows")
    else:
        print("No 'vm_id' column found in cleaned CSV.")

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
    if 'vm_id' not in df.columns:
        raise ValueError("No 'vm_id' column found in the CSV. Cannot plot per-VM timeseries.")
    if vm_id is None:
        if len(df) == 0:
            raise ValueError("No data available to plot.")
        vm_id = df['vm_id'].iloc[0]
    df_vm = df[df['vm_id'] == vm_id]
    if df_vm.empty:
        raise ValueError(f"No data found for vm_id={vm_id}")
    if resource not in df_vm.columns:
        raise ValueError(f"Resource column '{resource}' not found in CSV. Available columns: {list(df_vm.columns)}")
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
