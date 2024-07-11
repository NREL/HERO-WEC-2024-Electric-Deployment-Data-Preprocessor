from typing import List, Optional, Union
from pathlib import Path

import duckdb

import pandas as pd


def extract_group(
    partition_folder: str,
    group: str,
    query_columns: Union[List[str], str] = "*",
    selected_columns: Optional[List[str]] = None,
):
    def build_partition_folder_name(partition_name, parameter):
        return f"{partition_name}={parameter}"

    query_month = build_partition_folder_name("month", "*")
    query_day = build_partition_folder_name("day", "*")
    query_hour = build_partition_folder_name("hour", "*")
    query_minute = build_partition_folder_name("minute", "*")
    query_group = build_partition_folder_name("group", group)

    partition_parquet_path = Path(partition_folder)

    # Collect all matching parquet file paths
    parquet_files = list(
        partition_parquet_path.glob(
            f"year=*/{query_month}/{query_day}/{query_hour}/{query_minute}/{query_group}/*.parquet"
        )
    )

    if query_columns != "*":
        if isinstance(query_columns, str):
            query_columns = [query_columns]

        query_columns.extend(["Datetime"])

        query_columns = [f'"{col}"' for col in query_columns]

        query_columns = ", ".join(query_columns)

    # Process files in smaller batches
    batch_size = 128  # Adjust batch size as needed
    result_dfs = []

    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i : i + batch_size]

        query = f"""
        SELECT {query_columns} FROM read_parquet([{", ".join([f"'{file}'" for file in batch_files])}], hive_partitioning=1, union_by_name=true);
        """

        print("\tExecuting query on batch:", i // batch_size + 1)
        this_result_df = duckdb.sql(query).df()

        if query_columns == "*":
            this_result_df = this_result_df.drop(
                columns=["year", "month", "day", "hour", "minute", "group"]
            )

        result_dfs.append(this_result_df)

    print(f"\tConcatenating {query_columns}...")
    result_df = pd.concat(result_dfs)

    print(f"\tSetting Index for {query_columns}...")
    result_df = result_df.set_index(["Datetime"])

    print(f"\tSorting {query_columns}...")
    result_df = result_df.sort_index()

    if selected_columns is not None:
        result_df = result_df[selected_columns]

    return result_df
