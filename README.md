# Overview

This repository contains public releases of Microsoft Azure traces for the benefit of the research and academic community.

---

## Timeseries Preparation and Plotting

This repository provides a script, `prepare_timeseries.py`, to help you download, decompress, concatenate, clean, and visualize Azure VM resource utilization timeseries data.

### How to Use

#### 1. Download and Prepare Data

You can use the script to download one or more raw `.csv.gz` files from the Azure public dataset, decompress them, concatenate them into a single CSV, and clean the data for analysis.

**Example: Download, process, and plot a sample timeseries**

```bash
python prepare_timeseries.py \
  --download_url https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz \
  --plot
```

- If you want to process more files, download them into the same directory (e.g., with `wget` or by running the script multiple times with different `--download_url` values).
- The script will skip downloading files that already exist.

#### 2. Cleaned Output

The script will:
- Decompress all `.csv.gz` files in `data/raw_gz` to `data/raw_csv`
- Concatenate all `.csv` files in `data/raw_csv` into `data/combined_raw.csv`
- Clean and extract the timeseries into `data/cleaned_resource_timeseries.csv`

The cleaned CSV will have columns:

```
timestamp,vm_id,CPU_min,CPU_max,CPU_avg
```

- `timestamp`: The time of the measurement (as datetime)
- `vm_id`: The unique identifier for the virtual machine
- `CPU_min`, `CPU_max`, `CPU_avg`: CPU utilization statistics for each 5-minute interval

#### 3. Plotting

To plot a timeseries for a specific VM and resource (e.g., CPU_avg):

```bash
python prepare_timeseries.py --plot --vm_id <VM_ID> --resource CPU_avg
```

If `--vm_id` is not specified, the script will plot the first VM in the dataset.

#### 4. Function Descriptions

- `download_file(url, dest_path)`: Downloads a file from a URL if it does not already exist.
- `decompress_gz_files(input_dir, output_dir)`: Decompresses all `.csv.gz` files in a directory.
- `concatenate_csv_files(input_dir, output_csv)`: Concatenates all `.csv` files in a directory into a single CSV with header.
- `prepare_timeseries(raw_data_path, output_path)`: Cleans and extracts the timeseries data, inferring the correct columns from the Azure schema.
- `plot_timeseries(csv_path, vm_id, resource)`: Plots the specified resource timeseries for a given VM.

#### 5. Requirements

You will need the following Python packages:

- pandas
- matplotlib
- requests

Install them with:

```bash
pip install pandas matplotlib requests
```

---

For more details, see the comments and docstrings in `prepare_timeseries.py`.

There are currently four classes of traces:

* VM Traces: two representative traces of the virtual machine (VM) workload of Microsoft Azure collected in 2017 and 2019, and one VM request trace specifically for investigating packing algorithms.
* Azure Functions Traces: representative traces of Azure Functions invocations, collected over two weeks in 2019, and of Azure Functions blob accesses, collected between November and December of 2020.
* Azure LLM Inference Traces: representative traces of LLM inference invocations with input and output tokens, collected in November 2023 and May 2024.
* Azure VM Benchmark Noise Data: longitudinal data on performance variability for some resources and applications for two VM SKUs in two regions in Azure between May 2023 and September 2024.

We provide the traces as they are, but are willing to help researchers understand and use them. So, please let us know of any issues or questions by sending email to our  [mailing list](mailto:azurepublicdataset@service.microsoft.com).

## Quick links by paper:

* Traces ([2017](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md))([2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)) for the paper "Resource Central: Understanding and Predicting Workloads for Improved Resource Management in Large Cloud Platforms" (SOSP'17)
* Traces ([2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)) for the paper "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider" (ATC'19)
* Traces ([2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md)) for the paper "Protean: VM Allocation Service at Scale" (OSDI'20)
* Traces ([2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsBlobDataset2020.md)) for the paper "Faa$T: A Transparent Auto-Scaling Cache for Serverless Applications" (SoCC'21)
* Traces ([2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)) for the paper "Splitwise: Efficient generative LLM inference using phase splitting" (ISCA'24)
* Dataset and code ([2023](AzureGreenSKUFramework2023.md)) for the paper "Designing Cloud Servers for Lower Carbon" (ISCA'24)
* Traces ([2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)) for the paper "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency" (HPCA'25)
* Cloud Benchmarks ([2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureVMNoiseDataset2024.md)) for the paper "TUNA: Tuning Unstable and Noisy Cloud Applications" (EuroSys'25)

## VM Traces

The traces are sanitized subsets of the first-party VM workload in one of Azure’s geographical regions.  We include jupyter notebooks that directly compare the main characteristics of each trace to its corresponding full VM workload, showing that they are qualitatively very similar (except for VM deployment sizes in 2019).  Comparing the characteristics of the two traces illustrates how the workload has changed over this two-year span.

If you do use either of these VM traces in your research, please make sure to cite our SOSP’17 paper ["Resource Central: Understanding and Predicting Workloads for Improved Resource Management in Large Cloud Platforms"](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/Resource-Central-SOSP17.pdf), which includes a full analysis of the Azure VM workload in 2017.

### Trace Locations

* [AzurePublicDatasetV1](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md) - Trace created using data from 2017 Azure VM workload containing information about ~2M VMs and 1.2B utilization readings.
* [AzurePublicDatasetV2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) - Trace created using data from 2019 Azure VM workload containing information about ~2.6M VMs and 1.9B utilization readings.

## Azure Traces for Packing

* [AzureTracesForPacking2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md) - This dataset represents part of the workload on Microsoft's Azure Compute and is specifically intended to evaluate packing algorithms. The dataset includes:

  * VM requests along with their priority
  * The lifetime for each requested VM
  * The (normalized) resources allocated for each VM type.

If you do use the Azure Trace for Packing in your research, please make sure to cite our OSDI'20 paper ["Protean: VM Allocation Service at Scale"](https://www.usenix.org/system/files/osdi20-hadary.pdf), which includes a description of the Azure allocator and related workload analysis.


## Azure Functions Traces

### Function Invocations
* [AzureFunctionsDataset2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) - These traces contain, for a subset of applications running on Azure Functions in July of 2019:

  * how many times per minute each (anonymized) function is invoked and its corresponding trigger group
  * how (anonymized) functions are grouped into (anonymized) applications, and how applications are grouped by (anonymized) owner
  * the distribution of execution times per function
  * the distribution of memory usage per application

If you do use the Azure Functions 2019 traces in your research, please make sure to cite our ATC'20 paper ["Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider"](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/serverless-ATC20.pdf), which includes a full analysis of the Azure Functions workload in July 2019.

* [AzureFunctionsInvocationTrace2021](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md) - This is a trace of function invocations for two weeks starting on 2021-01-31. The trace contains invocation arrival and departure (or compeletion) times, with the folloiwng schema:  

  * app: application id (encrypted)
  * func: function id (encrypted), and unique only within an application 
  * end_timestamp: function invocation end timestamp in millisecond
  * duration: duration of function invocation in millisecond


If you do use the Azure Functions 2021 trace in your research, please cite this SOSP'21 paper ["Faster and Cheaper Serverless Computing on Harvested Resources"](https://www.microsoft.com/en-us/research/publication/faster-and-cheaper-serverless-computing-on-harvested-resources/).

### Functions Blob Accesses

* [AzureFunctionsBlobDataset2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsBlobDataset2020.md) - This is a sample of the blob accesses in Microsoft's Azure Functions, collected between November 23<sup>rd</sup> and December 6<sup>th</sup> 2020. This dataset is the data described and analyzed in the SoCC 2021 paper 'Faa$T: A Transparent Auto-Scaling Cache for Serverless Applications'.


## Azure LLM Inference Traces
* [AzureLLMInferenceDataset2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) - This is a sample of two LLM inference services in Azure containing the input and output tokens. This dataset was collected on November 11<sup>th</sup> 2023. This contains the data described and analyzed in the ISCA 2024 paper 'Splitwise: Efficient generative LLM inference using phase splitting'.
* [AzureLLMInferenceDataset2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) - This is a longer one week sample of two LLM inference services in Azure containing the input and output tokens. This dataset was collected on May 2024. This contains the data described and analyzed in the HPCA 2025 paper 'DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency'.

## Azure Benchmark Traces
* [AzureVMNoiseDataset2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureVMNoiseDataset2024.md) - This is a set of benchmarks that were run repeatedely over a period of 483 days. This dataset was collected from May 2023 to September 2024, and described and used as motivation in the EuroSys 2025 paper 'TUNA: Tuning Unstable and Noisy Cloud Applications'.

### Contact us
Please let us know of any issues or questions by sending email to our [mailing list](mailto:azurepublicdataset@service.microsoft.com).

These traces derive from a collaboration between Azure and Microsoft Research.
