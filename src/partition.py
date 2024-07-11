import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .ModaqPartitionDataframe import ModaqPartitionDataframe
from .extract import get_dirs_in_folder, get_files_in_folder


def partition_single_file(partitioner, file, name, group_name, output_folder):
    print(f"\t\tPartitioning file: {file}")
    partitioner.partition_df(
        str(file),
        name,
        group_name,
        output_folder,
    )


def partition_by_time(input_folder, output_folder, name):
    partitioner = ModaqPartitionDataframe()
    one_to_one_folders = get_dirs_in_folder(input_folder)

    for folder in one_to_one_folders:
        group_name = Path(folder).name
        input_parquet_files = get_files_in_folder(folder, "parquet")
        print(f"\tPartitioning {len(input_parquet_files)} parquet files in {folder}...")

        # Use ProcessPoolExecutor to parallelize the partitioning process
        num_workers = os.cpu_count()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    partition_single_file,
                    partitioner,
                    file,
                    name,
                    group_name,
                    output_folder,
                )
                for file in input_parquet_files
            ]
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()  # Retrieve the result to catch any exceptions
                    print(
                        f"\t\tSuccessfully partitioned file {count} of {len(input_parquet_files)}"
                    )
                except Exception as e:
                    print(
                        f"\t\tFailed to partition file {count} of {len(input_parquet_files)}: {e}"
                    )

    return True


def cleanup_single_partition(folder):
    ModaqPartitionDataframe().cleanup_single_partition(folder)


def cleanup_partitions(input_path, partition_folder):
    input_path = str(Path(partition_folder).resolve())
    folders = get_dirs_in_folder(input_path)

    # Use ProcessPoolExecutor to parallelize the cleanup process
    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(cleanup_single_partition, folder) for folder in folders
        ]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()  # Retrieve the result to catch any exceptions
                print(
                    f"\t\tSuccessfully cleaned up partition {count} of {len(folders)}"
                )
            except Exception as e:
                print(
                    f"\t\tFailed to clean up partition {count} of {len(folders)}: {e}"
                )


def create_standardized_partitions(input_folder, output_folder):
    ModaqPartitionDataframe().create_standardized_partitions(
        input_folder, output_folder
    )


def resample_wec_subsystem_partition(input_folder):
    ModaqPartitionDataframe().resample_wec_subsystem(input_folder)


def resample_ro_subsystem_partition(input_folder):
    ModaqPartitionDataframe().align_ro_subsystem(input_folder)


def concat_standardized_partitions(input_folder):
    ModaqPartitionDataframe().concat_standardized_partitions(input_folder)
