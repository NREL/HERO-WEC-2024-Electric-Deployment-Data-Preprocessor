import os
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

from .ModaqHeroFilter import ModaqHeroFilter

from .extract import get_files_in_folder


def filter_individual_file(input_file, output_folder):
    groups_to_skip = ["GPS", "MAG", "AIN", "HDG", "IMU", "ODOM", "IMF"]
    hero_filter = ModaqHeroFilter()
    file_path = input_file.parent
    file_parts = file_path.parts
    partition_file_parts = [part for part in file_parts if "=" in part]
    group_name = [part for part in partition_file_parts if "group" in part][0].split(
        "="
    )[-1]

    if group_name in groups_to_skip:
        return

    if "BAG" in group_name:
        string_to_remove = "-BAG"
        group_name = group_name.replace(string_to_remove, "")
        partition_file_parts = [
            part.replace(string_to_remove, "") for part in partition_file_parts
        ]

    full_output_folder = Path(output_folder, *partition_file_parts)
    print(f"\t\tFiltering {input_file}...")
    hero_filter.filter(input_file, full_output_folder, group_name)


def filter_modaq_by_file(input_folder: str, output_folder: str):
    input_files = get_files_in_folder(input_folder, "parquet")
    print(f"\tFiltering {len(input_files)} parquet_files...")

    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(filter_individual_file, input_file, output_folder)
            for input_file in input_files
        ]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()  # Retrieve the result to catch any exceptions
                print(f"\t\tSuccessfully filtered {count} of {len(input_files)}")
            except Exception as e:
                print(f"\t\tFailed to filter {count} of {len(input_files)}: {e}")
