import pandas as pd
import argparse
import os
import glob
import gzip
import re
import logging
import matplotlib.pyplot as plt
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def download_vm_traces(links_file: str, download_dir: str, num_traces: int = 1):
    """
    Download the first N VM CPU reading traces (with vm-id in name) from the AzurePublicDatasetLinksV2.txt file.
    """
    import requests
    os.makedirs(download_dir, exist_ok=True)
    with open(links_file) as f:
        urls = [line.strip() for line in f if "vm_cpu_readings" in line and "vm_cpu_readings-file-" in line]
    urls = sorted(urls)[:num_traces]
    for url in urls:
        fname = os.path.join(download_dir, os.path.basename(url))
        if os.path.exists(fname):
            logging.info(f"File already exists, skipping download: {fname}")
            continue
        logging.info(f"Downloading {url} to {fname}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(fname, "wb") as out:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    out.write(chunk)
        logging.info(f"Downloaded: {fname}")
    logging.info(f"Downloaded {len(urls)} trace files to {download_dir}")

def _decompress_single_gz(gz_file, output_dir, header):
    out_file = os.path.join(output_dir, os.path.basename(gz_file)[:-3])
    if os.path.exists(out_file):
        logging.info(f"File already decompressed, skipping: {out_file}")
        return
    logging.info(f"Decompressing {gz_file} to {out_file}")
    with gzip.open(gz_file, 'rt') as f_in:
        first_line = f_in.readline()
        has_header = any(x.isalpha() for x in first_line.split(",")[0])
    with gzip.open(gz_file, 'rt') as f_in, open(out_file, 'w') as f_out:
        # Always write the correct header
        f_out.write(header)
        f_in.seek(0)
        # If the file has a header, skip it
        if has_header:
            next(f_in)
        for line in f_in:
            # Only keep the first 5 columns
            cols = line.rstrip("\n").split(",")
            if len(cols) >= 5:
                f_out.write(",".join(cols[:5]) + "\n")

def decompress_gz_files(input_dir: str, output_dir: str, max_workers: int = 8):
    """
    Decompress all .csv.gz files in input_dir to output_dir, unless already decompressed.
    Always add the correct header for Azure VM CPU readings if missing.
    Only keep the columns: reading_ts,vm_id,min_cpu_5min,max_cpu_5min,avg_cpu_5min
    Uses parallel processing for speed.
    """
    os.makedirs(output_dir, exist_ok=True)
    gz_files = glob.glob(os.path.join(input_dir, "*.csv.gz"))
    header = "reading_ts,vm_id,min_cpu_5min,max_cpu_5min,avg_cpu_5min\n"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_decompress_single_gz, gz_file, output_dir, header) for gz_file in gz_files]
        for future in concurrent.futures.as_completed(futures):
            # Will raise exceptions if any
            future.result()

def concatenate_csv_files(input_dir: str, output_csv: str):
    """
    Concatenate all .csv files in input_dir into a single CSV with header.
    Assumes all files have the correct header.
    """
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    with open(output_csv, 'w') as fout:
        for i, fname in enumerate(csv_files):
            with open(fname) as fin:
                lines = fin.readlines()
                if i == 0:
                    fout.writelines(lines)
                else:
                    fout.writelines(lines[1:])  # skip header
    print(f"Concatenated {len(csv_files)} files into {output_csv}")
    with open(output_csv, 'r') as fcheck:
        header = fcheck.readline().strip()
        print(f"Header of generated CSV: {header}")

def prepare_timeseries_per_vm(raw_data_path: str, output_dir: str):
    """
    Prepare and clean a time series dataset for Azure VM CPU utilization.
    Write a separate cleaned CSV file for each VM in output_dir.
    """
    expected_cols = ['reading_ts', 'vm_id', 'min_cpu_5min', 'max_cpu_5min', 'avg_cpu_5min']
    df = pd.read_csv(raw_data_path)
    if df.shape[0] == 0 or df.shape[1] == 0:
        print("Warning: No data found in the concatenated CSV.")
        return
    if list(df.columns) != expected_cols:
        df.columns = expected_cols
    df = df.sort_values(['vm_id', 'reading_ts'])
    df = df.ffill().dropna()
    df = df.drop_duplicates()
    for col in ['min_cpu_5min', 'max_cpu_5min', 'avg_cpu_5min']:
        df = df[(df[col] >= 0) & (df[col] <= 100)]
    os.makedirs(output_dir, exist_ok=True)
    vm_ids = df['vm_id'].unique()
    for vm_id in vm_ids:
        vm_df = df[df['vm_id'] == vm_id]
        if vm_df.empty:
            continue
        safe_id = _safe_vm_id(vm_id)
        out_path = os.path.join(output_dir, f"{safe_id}.csv")
        vm_df.to_csv(out_path, index=False)
    print(f"Wrote {len(vm_ids)} per-VM time series files to {output_dir}")

import hashlib

def _safe_vm_id(vm_id: str) -> str:
    """
    Return a filesystem-safe, short, unique string for a VM id.
    """
    # Use first 8 chars of sha256 hash for uniqueness and safety
    return hashlib.sha256(vm_id.encode("utf-8")).hexdigest()[:8]

def plot_timeseries(csv_path: str, vm_ids: list = None, sample: int = 1000, output_dir: str = "plots"):
    """
    Plot all resource columns (except timestamp/vm_id) for each specified VM.
    """
    df = pd.read_csv(csv_path)
    if 'vm_id' not in df.columns:
        raise ValueError("No 'vm_id' column found in the CSV. Cannot plot per-VM timeseries.")
    os.makedirs(output_dir, exist_ok=True)
    if vm_ids is None:
        vm_ids = df['vm_id'].unique()[:5]
    for vm_id in vm_ids:
        df_vm = df[df['vm_id'] == vm_id]
        if df_vm.empty:
            logging.warning(f"No data found for vm_id={vm_id}")
            continue
        x = df_vm['reading_ts'][:sample]
        safe_id = _safe_vm_id(vm_id)
        for col in df_vm.columns:
            if col in ['reading_ts', 'vm_id']:
                continue
            plt.figure(figsize=(12, 5))
            plt.plot(x, df_vm[col][:sample])
            plt.xlabel("reading_ts")
            plt.ylabel(f"{col}")
            plt.title(f"{col} for VM {safe_id}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fname = os.path.join(output_dir, f"{safe_id}_{col}.png")
            plt.savefig(fname)
            plt.close()
            logging.info(f"Saved plot: {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Azure VM CPU time series data. (Plotting is now a separate step.)")
    parser.add_argument("--links_file", type=str, default="AzurePublicDatasetLinksV2.txt", help="Text file with Azure dataset links")
    parser.add_argument("--download_dir", type=str, default="data/raw_gz", help="Directory to store downloaded .csv.gz files")
    parser.add_argument("--num_traces", type=int, default=1, help="Number of VM CPU reading traces to download")
    parser.add_argument("--decompressed_dir", type=str, default="data/raw_csv", help="Directory with decompressed .csv files")
    parser.add_argument("--concat_csv", type=str, default="data/combined_raw.csv", help="Path to concatenated raw CSV")
    parser.add_argument("--per_vm_dir", type=str, default="data/per_vm_timeseries", help="Directory to save per-VM cleaned time series CSVs")
    parser.add_argument("--decompress_workers", type=int, default=8, help="Number of parallel workers for decompression")
    args = parser.parse_args()

    # Step 1: Download N VM CPU reading traces
    download_vm_traces(args.links_file, args.download_dir, num_traces=args.num_traces)

    # Step 2: Decompress all .csv.gz files in download_dir to decompressed_dir (parallel)
    decompress_gz_files(args.download_dir, args.decompressed_dir, max_workers=args.decompress_workers)

    # Step 3: Concatenate all .csv files in decompressed_dir
    concatenate_csv_files(args.decompressed_dir, args.concat_csv)

    # Step 4: Prepare and clean the time series, writing one file per VM
    prepare_timeseries_per_vm(
        raw_data_path=args.concat_csv,
        output_dir=args.per_vm_dir
    )
