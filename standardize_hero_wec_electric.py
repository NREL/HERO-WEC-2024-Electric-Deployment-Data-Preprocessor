"""
Converts raw measurement data from HERO WEC Electrical Deployment
for engineering analysis.

This script provides functions to process raw measurement data stored in MODAQ format.
It includes functions to extract, filter, concatenate, visualize, perform quality control,
and organize the data for further analysis.
"""

import os

from pathlib import Path

from src import extract, filter, partition, qc, vap, summary


def build_script_path(path_addition: str) -> str:
    data_dir = "data"
    this_script_path = os.path.dirname(os.path.realpath(__file__))
    result_path = Path(this_script_path, data_dir, path_addition).resolve()
    return str(result_path)


if __name__ == "__main__":
    start_message = "Starting HERO WEC Electric Deployment Data Standardization..."
    print(start_message)

    raw_folder = build_script_path("00_raw_data")
    one_to_one_folder = build_script_path("a1_interim_one_to_one_parquet")
    partition_folder = build_script_path("a2_interim_partitioned")
    filter_folder = build_script_path("a3_interim_filtered_parquet")
    standardized_partition_folder = build_script_path("a4_standardized_partition")
    vap_partition_folder = build_script_path("b1_standardized_vap_partition")
    summary_folder = build_script_path("b2_summary_standardized_vap")

    # print("Step 1: Extracting raw data...")

    # # TDMS Extraction
    # tdms_folders = ["CurrentAI", "PowRaw", "VoltageAI"]
    # for folder in tdms_folders:
    #     print(f"\tExtracting TDMS folder {folder}...")
    #     extract.extract_modaq_folder(str(Path(raw_folder, folder)), one_to_one_folder)

    # # ROS Bagfile Extraction
    # extract.extract_remote_daq_bagfiles(
    #     str(Path(raw_folder, "BuoyBagfiles")), one_to_one_folder
    # )

    # # CSV Extraction
    # extract.extract_mooring_daq_folder(
    #     str(Path(raw_folder, "MooringDAQ")), one_to_one_folder
    # )

    # # NC Extraction
    # extract.extract_weather(str(Path(raw_folder, "Weather")), one_to_one_folder)

    # print("Step 2: Partitioning...")
    # partition.partition_by_time(
    #     one_to_one_folder, partition_folder, "HERO-WEC-Electric-Deployment"
    # )
    # partition.cleanup_partitions(one_to_one_folder, partition_folder)

    # print("Step 3: Filtering...")
    # filter.filter_modaq_by_file(partition_folder, filter_folder)

    # print("Step 4: Creating Standardized Paritions")
    # partition.create_standardized_partitions(
    #     filter_folder, standardized_partition_folder
    # )

    # print("\tStep 4a: Performing QC on Standardized Paritions")
    # qc.qc_standardized_partitions(standardized_partition_folder)

    # print("\tStep 4b: Resampling WEC Subsystem on Standardized Paritions")
    # partition.resample_wec_subsystem_partition(standardized_partition_folder)

    # print("\tStep 4c: Resampling RO Subsystem on Standardized Paritions")
    # partition.resample_ro_subsystem_partition(standardized_partition_folder)

    # print("\tStep 4d: Concatenating Standardized Paritions")
    # partition.concat_standardized_partitions(standardized_partition_folder)

    # print("Step 5: Calculating VAP...")
    # vap.vap_standardized_partitions(standardized_partition_folder, vap_partition_folder)

    print("Step 6: Summarizing...")
    summary.summarize_vap_partition(vap_partition_folder, summary_folder)

    print("Complete!")
