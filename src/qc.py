from pathlib import Path

import pandas as pd

from .extract import get_files_in_folder


def qc_standardized_partitions(partition_folder):
    qc_df = pd.read_excel(
        Path(".", "qc", "hero_wec_electric_deploy_qc_values.xlsx").resolve()
    )
    qc_channels = qc_df["Channel"].to_list()

    parquet_files = get_files_in_folder(partition_folder, "parquet")

    for file in parquet_files:
        df = pd.read_parquet(file)
        for col in df.columns:
            if col in qc_channels:
                if col == "Timestamp_NS":
                    continue

                qc_col = qc_df[qc_df["Channel"] == col]
                this_series = df[col]

                qc_max = qc_col["QC Range Max"].values[0]
                qc_min = qc_col["QC Range Min"].values[0]

                fill_value = qc_col["Mean"].values[0]

                this_gt = this_series.gt(qc_max, fill_value=fill_value).astype(
                    int
                )  # QC flag is 1

                this_lt = (
                    this_series.lt(qc_min, fill_value=fill_value)
                    .astype(int)
                    .replace(1, 2)
                )  # QC flag is 2

                this_qc = this_gt + this_lt
                qc_col_name = f"qc_{col}"
                df[qc_col_name] = this_qc
                df[qc_col_name] = df[qc_col_name].astype("uint8")

                insert_index = list(df.columns).index(col)

                df.insert(insert_index, qc_col_name, df.pop(qc_col_name))

        df.to_parquet(file)

    return True
