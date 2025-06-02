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

## Anticipated Time Series Dataset Header

Based on the datasets described in this repository, the time series for resource utilization should be structured as follows:

```
timestamp,vm_id,CPU,Memory,IO,Network,Disk
```

- `timestamp`: The time of the measurement (in a consistent datetime format).
- `vm_id`: The unique identifier for the virtual machine (or workload).
- `CPU`: CPU utilization value (percentage or normalized).
- `Memory`: Memory utilization value.
- `IO`: Input/Output utilization value.
- `Network`: Network utilization value.
- `Disk`: Disk utilization value.

This structure allows for time series analysis per VM or per workload, as required.

---

## Example: Prepare and Clean Resource Utilization Time Series

Below is a Python script (`prepare_timeseries.py`) that demonstrates how to extract, construct, and clean a time series dataset for CPU, memory, IO, network, and disk utilization per VM from a CSV file.

### Usage

1. Place your raw dataset (CSV) in the working directory.
2. Update the `RAW_DATA_PATH` and `OUTPUT_PATH` variables in the script as needed.
3. Run the script:

```bash
python prepare_timeseries.py
```

### Example Script

```python
import pandas as pd

# Path to your raw dataset
RAW_DATA_PATH = "data/raw_resource_data.csv"
# Path to save the cleaned time series
OUTPUT_PATH = "data/cleaned_resource_timeseries.csv"

# Load the raw data
df = pd.read_csv(RAW_DATA_PATH)

# Select relevant columns and rename if necessary
# Adjust these column names to match your dataset
df = df[['timestamp', 'vm_id', 'CPU', 'Memory', 'IO', 'Network', 'Disk']]

# Convert timestamp to datetime and sort
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['vm_id', 'timestamp'])

# Handle missing values (forward fill, then drop remaining)
df = df.fillna(method='ffill').dropna()

# Remove duplicates
df = df.drop_duplicates()

# Optionally, remove outliers (e.g., utilization outside 0-100%)
for col in ['CPU', 'Memory', 'IO', 'Network', 'Disk']:
    df = df[(df[col] >= 0) & (df[col] <= 100)]

# Save cleaned time series
df.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned time series saved to {OUTPUT_PATH}")
```
