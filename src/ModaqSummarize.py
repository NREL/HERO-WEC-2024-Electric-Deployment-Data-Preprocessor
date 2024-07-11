from pathlib import Path
import pandas as pd
import numpy as np

from scipy import integrate

from dataclasses import dataclass


@dataclass
class TimeIndexUnits:
    year: str
    month: str
    day: str
    hour: str
    minute: str
    second: str


@dataclass
class ModaqSummarize:
    partition_folder: str
    summary_output_folder: str
    summary_column_prefix: str = "avg"

    def __post_init__(self):
        self.result_dfs = []

    def filter_output_columns(self, input_df: pd.DataFrame) -> pd.DataFrame:
        columns_to_remove = [
            "calc-Acc-Flow",
            "POS-OS-1001",
            "vap-POS-OS-1001-UW",
        ]

        output_df = input_df.drop(columns_to_remove, axis="columns", errors="ignore")
        return output_df

    def rename_output_columns(self, input_df: pd.DataFrame) -> pd.DataFrame:
        column_rename_map = {
            "FLOW-ON-1001": "calc_Q-Feed",
            "FLOW-ON-1002": "calc_Q-Clark-Pump",
            "FLOW-ON-1003": "calc_Q-Brine",
            "FLOW-ON-1004": "calc_Q-Permeate",
            "calc_Eta_CC_I": "calc_Eta_CC",
        }
        output_df = input_df.rename(columns=column_rename_map, errors="ignore")
        return output_df

    def standardize_output_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = self.filter_output_columns(input_df)
        output_df = self.rename_output_columns(output_df)
        return output_df

    def finalize_output_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df.columns = input_df.columns.str.replace("-", "_")
        return input_df

    def get_index_time_components(
        self, input_index: pd.DatetimeIndex, index_loc=0
    ) -> TimeIndexUnits:
        i = input_index[index_loc]
        return TimeIndexUnits(
            year=f"{i.year:04}",  # type: ignore
            month=f"{i.month:02}",  # type: ignore
            day=f"{i.day:02}",  # type: ignore
            hour=f"{i.hour:02}",  # type: ignore
            minute=f"{i.minute:02}",  # type: ignore
            second=f"{i.second:02}",  # type: ignore
        )

    def create_output_filename(self, output_df, group):
        start_tc = self.get_index_time_components(output_df.index)  # type: ignore
        end_tc = self.get_index_time_components(output_df.index, -1)  # type: ignore
        start_ts_str = f"{start_tc.year}-{start_tc.month}-{start_tc.day}_{start_tc.hour}-{start_tc.minute}"
        end_ts_str = (
            f"{end_tc.year}-{end_tc.month}-{end_tc.day}_{end_tc.hour}-{end_tc.minute}"
        )
        group = group.replace("group=", "")
        if "Weather" not in group:
            group = f"NREL_MODAQ_{group}"
        output_filename = f"{start_ts_str}_to_{end_ts_str}__{group.replace('group=', '')}__30_min_summary__Nags_Head_NC_USA__HERO_WEC_Electric_Deployment.parquet"
        return output_filename

    def summarize_partition_folder(self):
        folder = Path(self.partition_folder)
        prefix = self.summary_column_prefix

        group_folders = [entry for entry in folder.iterdir() if entry.is_dir()]

        output_folder = Path(self.summary_output_folder).resolve()
        output_folder.mkdir(exist_ok=True, parents=True)

        self.output_folder = output_folder

        groups = []
        for folder in group_folders:
            parts = folder.parts
            partition_parts = [part for part in parts if "=" in part]
            group = [part for part in partition_parts if "group" in part][0]
            groups.append(group)

        groups = [g.replace("group=", "") for g in groups]

        for group in groups:
            self.summarize_group(group)

        final_output_df = pd.concat(self.result_dfs, axis=1)
        final_output_df["Timestamp_NS"] = final_output_df.index.astype("int64")
        final_output_df.insert(0, "Timestamp_NS", final_output_df.pop("Timestamp_NS"))

        final_output_df = self.finalize_output_df(final_output_df)

        # Averages across groups
        final_output_df[f"{prefix}_calc_Eta_Rectifier"] = (
            final_output_df[f"{prefix}_calc_P_Electric_1"]
            / final_output_df[f"{prefix}_calc_P_WEC_Electric_Out"]
        )
        final_output_df[f"{prefix}_calc_Eta_WEC"] = (
            final_output_df[f"{prefix}_calc_P_WEC_Electric_Out"]
            / final_output_df[f"{prefix}_calc_P_abs"]
        )
        final_output_df[f"{prefix}_calc_Eta_Overall"] = (
            final_output_df[f"{prefix}_calc_P_Electric_3"]
            / final_output_df[f"{prefix}_calc_P_abs"]
        )

        final_output_df = self.finalize_output_df(final_output_df)

        output_filename = self.create_output_filename(final_output_df, "All")
        final_output_df.to_parquet(Path(output_folder, output_filename))

        # ValueError: Excel does not support datetimes with timezones. Please ensure that datetimes are timezone unaware before writing to Excel.
        final_output_df.index = final_output_df.index.tz_localize(None)  # type: ignore
        final_output_df.index.name = "Datetime_UTC"
        final_output_df.to_excel(
            Path(self.output_folder, output_filename.replace(".parquet", ".xlsx"))
        )

    def nan_mask_df_per_qc(self, df):
        qc_prefix = "qc_"
        qc_columns = [col for col in df.columns if col.startswith(qc_prefix)]

        for qc_col in qc_columns:
            data_col = qc_col.replace(qc_prefix, "")
            if data_col in df.columns:
                mask = df[qc_col] > 0
                df.loc[mask, data_col] = np.nan

        df = df.drop(columns=qc_columns)
        return df

    def summarize_group(self, group):
        print("\tSummarizing group", group)
        folder = Path(self.partition_folder, f"group={group}")
        parquet_files = list(folder.rglob("**/*.parquet"))
        parquet_files.sort()

        this_dfs = []

        count = 1
        for file in parquet_files:
            print(f"\t\tSummarizing file {count} of {len(parquet_files)}:", file.name)
            this_df = pd.read_parquet(file)

            this_df = self.nan_mask_df_per_qc(this_df)

            if group == "Gen":
                gen_group_sample_rate_hz = 5000
                this_df["calc_P_WEC_Electric_Out"] = self.calculate_average_power(
                    this_df,
                    ["PT-ON-1001", "PT-ON-1002", "PT-ON-1003"],
                    ["CT-ON-1001", "CT-ON-1002", "CT-ON-1003"],
                    gen_group_sample_rate_hz,
                )
                this_df["calc_Work_Electrical"] = integrate.trapezoid(
                    this_df["calc_P_WEC_Electric_Out"]
                )

            if group == "PE":
                this_df = this_df.drop(["calc_Eta_CC_I"], axis="columns")
                this_df["calc_Eta_CC"] = (
                    this_df["calc_P_Electric_2"] / this_df["calc_P_Electric_3"]
                )

            if group == "RO":
                ns_index = this_df.index.astype("int64")
                ns_elapsed = ns_index - ns_index[0]
                seconds_elapsed = ns_elapsed / 1_000_000_000
                calc_Volume_RO = integrate.trapezoid(
                    this_df["FLOW-ON-1001"] / 60, seconds_elapsed
                )
                calc_Volume_Clark_Pump = integrate.trapezoid(
                    this_df["FLOW-ON-1002"] / 60, seconds_elapsed
                )
                calc_Volume_Permeate = integrate.trapezoid(
                    this_df["FLOW-ON-1004"] / 60, seconds_elapsed
                )

            if "Weather" not in group:
                this_df = self.standardize_output_df(this_df)
                this_df = this_df.add_prefix(f"{self.summary_column_prefix}_")
                this_df = this_df.rename(
                    columns={
                        f"{self.summary_column_prefix}_Timestamp_NS": "Timestamp_NS"
                    }
                )

            if "NDBC" in group:
                this_df = this_df.apply(pd.to_numeric)

            thirty_min_mean_df = this_df.resample("30T").mean()

            if group == "WEC":
                thirty_min_mean_df["max_LC-MOOR-1001"] = this_df[
                    f"{self.summary_column_prefix}_LC-MOOR-1001"
                ].max()
                thirty_min_mean_df["min_LC-MOOR-1001"] = this_df[
                    f"{self.summary_column_prefix}_LC-MOOR-1001"
                ].min()

            if group == "RO":
                ns_index = this_df.index.astype("int64")
                ns_elapsed = ns_index - ns_index[0]
                seconds_elapsed = ns_elapsed / 1_000_000_000
                thirty_min_mean_df["calc_Volume_RO"] = calc_Volume_RO  # type: ignore
                thirty_min_mean_df["calc_Volume_Clark_Pump"] = calc_Volume_Clark_Pump  # type: ignore
                thirty_min_mean_df["calc_Volume_Permeate"] = calc_Volume_Permeate  # type: ignore

            thirty_min_mean_df["Timestamp_NS"] = thirty_min_mean_df.index.astype(
                "int64"
            )
            this_dfs.append(thirty_min_mean_df)

            count += 1

        output_df = pd.concat(this_dfs)

        if "Wave_Variables" in group:
            this_group = group.split("=")[-1]
            prefix = "_".join(this_group.split("_")[1:3]) + "_"
            timestamp_ns = output_df["Timestamp_NS"]
            output_df = output_df.drop(["Timestamp_NS"], axis="columns")
            output_df = output_df.add_prefix(prefix)
            output_df["Timestamp_NS"] = timestamp_ns

        output_df.index = pd.to_datetime(output_df.index, utc=True)
        output_df.index.name = "Datetime"
        output_df = output_df.sort_index()
        output_df.insert(0, "Timestamp_NS", output_df.pop("Timestamp_NS"))

        output_df = self.finalize_output_df(output_df)

        for_all_summary_df = output_df.drop(["Timestamp_NS"], axis="columns")
        self.result_dfs.append(for_all_summary_df)

    def calculate_average_power(
        self, df, voltage_columns, current_columns, sampling_rate
    ):
        """
        Calculate average power output for all three phases from voltage and current data.

        Parameters:
        - df: pandas DataFrame containing voltage and current data with datetime index.
        - voltage_columns: list of column names for voltage data [V] (e.g., ['Voltage_Phase1', 'Voltage_Phase2', 'Voltage_Phase3']).
        - current_columns: list of column names for current data [A] corresponding to voltage_columns.
        - sampling_rate: sampling rate of the data in Hz.

        Returns:
        - average_power: summary statistic of average power [W] for all three phases.
        """
        total_time = len(df) / sampling_rate  # total time duration in seconds

        average_power = 0.0

        for i in range(3):  # iterate over each phase
            voltage = df[voltage_columns[i]]
            current = df[current_columns[i]]

            # Calculate instantaneous power
            instantaneous_power = voltage * current

            # Calculate average power using numerical integration (Trapezoidal rule)
            avg_power_phase_i = (
                np.trapz(instantaneous_power, dx=1 / sampling_rate) / total_time
            )

            # Accumulate average power of all phases
            average_power += avg_power_phase_i

        return average_power
