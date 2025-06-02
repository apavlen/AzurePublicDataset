import pandas as pd
import argparse
import os
import glob
import gzip
import matplotlib.pyplot as plt

def decompress_gz_files(input_dir: str, output_dir: str):
    """
    Decompress all .csv.gz files in input_dir to output_dir, unless already decompressed.
    Always add the correct header for Azure VM CPU readings if missing.
    """
    os.makedirs(output_dir, exist_ok=True)
    gz_files = glob.glob(os.path.join(input_dir, "*.csv.gz"))
    header = "reading_ts,vm_id,min_cpu_5min,max_cpu_5min,avg_cpu_5min\n"
    for gz_file in gz_files:
        out_file = os.path.join(output_dir, os.path.basename(gz_file)[:-3])
        if os.path.exists(out_file):
            print(f"File already decompressed, skipping: {out_file}")
            continue
        print(f"Decompressing {gz_file} to {out_file}")
        with gzip.open(gz_file, 'rt') as f_in:
            first_line = f_in.readline()
            has_header = any(x.isalpha() for x in first_line.split(",")[0])
        with gzip.open(gz_file, 'rt') as f_in, open(out_file, 'w') as f_out:
            if not has_header:
                f_out.write(header)
            f_in.seek(0)
            f_out.write(f_in.read())

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

def prepare_timeseries(raw_data_path: str, output_path: str):
    """
    Prepare and clean a time series dataset for Azure VM CPU utilization.
    """
    expected_cols = ['reading_ts', 'vm_id', 'min_cpu_5min', 'max_cpu_5min', 'avg_cpu_5min']
    df = pd.read_csv(raw_data_path)
    if df.shape[0] == 0 or df.shape[1] == 0:
        print("Warning: No data found in the concatenated CSV.")
        df.to_csv(output_path, index=False)
        return
    if list(df.columns) != expected_cols:
        df.columns = expected_cols
    df = df.sort_values(['vm_id', 'reading_ts'])
    df = df.ffill().dropna()
    df = df.drop_duplicates()
    for col in ['min_cpu_5min', 'max_cpu_5min', 'avg_cpu_5min']:
        df = df[(df[col] >= 0) & (df[col] <= 100)]
    df.to_csv(output_path, index=False)
    print(f"Cleaned time series saved to {output_path}")
    print("Sample of cleaned CSV:")
    print(df.head())
    if 'vm_id' in df.columns:
        unique_vms = df['vm_id'].nunique()
        print(f"Number of unique VMs in cleaned CSV: {unique_vms}")
        print("Rows per VM (first 5 VMs):")
        for vm in df['vm_id'].unique()[:5]:
            count = (df['vm_id'] == vm).sum()
            print(f"  VM {vm}: {count} rows")
    else:
        print("No 'vm_id' column found in cleaned CSV.")

def plot_timeseries(csv_path: str, vm_id: str = None, resource: str = "avg_cpu_5min", sample: int = 1000):
    """
    Plot a CPU utilization time series for a given VM.
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
    x = df_vm['reading_ts'][:sample]
    plt.figure(figsize=(12, 5))
    plt.plot(x, df_vm[resource][:sample])
    plt.xlabel("reading_ts")
    plt.ylabel(f"{resource} Utilization")
    plt.title(f"{resource} Utilization for VM {vm_id}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and plot Azure VM CPU time series data.")
    parser.add_argument("--decompressed_dir", type=str, default="data/raw_csv", help="Directory with decompressed .csv files")
    parser.add_argument("--concat_csv", type=str, default="data/combined_raw.csv", help="Path to concatenated raw CSV")
    parser.add_argument("--output", type=str, default="data/cleaned_resource_timeseries.csv", help="Path to save cleaned time series CSV")
    parser.add_argument("--plot", action="store_true", help="Plot a sample timeseries after processing")
    parser.add_argument("--vm_id", type=str, default=None, help="VM ID to plot (optional)")
    args = parser.parse_args()

    # Step 1: Decompress all .csv.gz files in raw_gz to raw_csv
    decompress_gz_files("data/raw_gz", args.decompressed_dir)

    # Step 2: Concatenate all .csv files in decompressed_dir
    concatenate_csv_files(args.decompressed_dir, args.concat_csv)

    # Step 3: Prepare and clean the time series
    prepare_timeseries(
        raw_data_path=args.concat_csv,
        output_path=args.output
    )

    # Step 4: Optionally plot a timeseries
    if args.plot:
        plot_timeseries(args.output, vm_id=args.vm_id)
