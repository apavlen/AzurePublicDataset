# TODO: Preparing a Time Series Dataset for Resource Utilization

## Goal
Prepare a clean time series dataset for resource utilization (CPU, memory, IO, etc.) from available raw datasets.

## Plan

1. **Dataset Selection**
   - Identify and select the raw dataset(s) containing resource utilization metrics.

2. **Data Extraction**
   - Extract relevant columns: timestamp, resource name (e.g., CPU, Memory, IO), and utilization value.

3. **Time Series Construction**
   - Transform the extracted data into a time series format:
     - Columns: `timestamp`, `resource_name`, `utilization`
     - Ensure timestamps are in a consistent format and sorted.

4. **Additional Resource Types**
   - Consider including other resource types if available (e.g., network, disk).

5. **Data Cleaning**
   - Handle missing values (e.g., interpolation, forward/backward fill, or removal).
   - Remove or flag outliers.
   - Ensure no duplicate rows.
   - Normalize or standardize utilization values if needed.

6. **Validation**
   - Check for consistency and completeness of the time series.
   - Visualize samples to verify correctness.

7. **Export**
   - Save the cleaned time series dataset in a suitable format (e.g., CSV, Parquet).

## Next Steps

- Implement a Python script to perform steps 2–5.
- Review and iterate on the cleaning process as needed.

---

## Azure VM CPU Readings Time Series Dataset Header

Based on the AzurePublicDataset documentation and the analysis notebook, the time series for VM CPU utilization should be structured as follows:

```
reading_ts,vm_id,min_cpu_5min,max_cpu_5min,avg_cpu_5min
```

- `reading_ts`: The timestamp of the measurement (seconds since deployment start, or as provided in the dataset).
- `vm_id`: The unique identifier for the virtual machine.
- `min_cpu_5min`: Minimum CPU utilization during the 5-minute interval.
- `max_cpu_5min`: Maximum CPU utilization during the 5-minute interval.
- `avg_cpu_5min`: Average CPU utilization during the 5-minute interval.

This structure allows for time series analysis per VM, as required.

---

## Example: Prepare and Clean Azure VM CPU Utilization Time Series

Below is a Python script (`prepare_timeseries.py`) that demonstrates how to extract, construct, and clean a time series dataset for CPU utilization per VM from the Azure 2019 Public Dataset V2.

### Usage

To prepare per-VM time series files from the Azure 2019 Public Dataset V2:

1. Download and decompress the raw Azure VM CPU readings dataset (CSV or CSV.GZ) using the script.
2. Run the script to download, decompress, concatenate, and write per-VM time series files:

```bash
python prepare_timeseries.py --links_file AzurePublicDatasetLinksV2.txt --num_traces 195 --download_dir data/raw_gz --decompressed_dir data/raw_csv --concat_csv data/combined_raw.csv --per_vm_dir data/per_vm_timeseries --decompress_workers 8
```

- This will download all 195 VM CPU reading files, decompress them in parallel, concatenate them, and write a cleaned CSV for each VM to `data/per_vm_timeseries/`.
- You can adjust `--num_traces` to limit the number of files, or `--decompress_workers` for parallelism.

### Example Script

```python
import pandas as pd

# Path to your raw dataset
RAW_DATA_PATH = "data/raw_resource_data.csv"
# Path to save the cleaned time series
OUTPUT_PATH = "data/cleaned_resource_timeseries.csv"

# Load the raw data
df = pd.read_csv(RAW_DATA_PATH)

# Select relevant columns and rename for clarity
df = df[['reading_ts', 'vm_id', 'min_cpu_5min', 'max_cpu_5min', 'avg_cpu_5min']]

# Convert timestamp to datetime if desired (optional)
# df['reading_ts'] = pd.to_datetime(df['reading_ts'], unit='s')

# Sort by VM and timestamp
df = df.sort_values(['vm_id', 'reading_ts'])

# Handle missing values (forward fill, then drop remaining)
df = df.ffill().dropna()

# Remove duplicates
df = df.drop_duplicates()

# Remove outliers (e.g., utilization outside 0-100%)
for col in ['min_cpu_5min', 'max_cpu_5min', 'avg_cpu_5min']:
    df = df[(df[col] >= 0) & (df[col] <= 100)]

# Save cleaned time series
df.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned time series saved to {OUTPUT_PATH}")
```
