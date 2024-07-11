import math

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class TimeIndexUnits:
    year: str
    month: str
    day: str
    hour: str
    minute: str
    second: str


@dataclass
class ModaqPartitionDataframe:
    pd_frequency_string: str = "30min"  # Partition in 30 minute intervals
    ns_timestamp_column: str = "Timestamp_NS"
    timestamp_ns_col_name: str = "Timestamp_NS"
    datetime_col_name: str = "Datetime"
    partition_separator: str = "="

    def create_datetime_index(
        self, input_df: pd.DataFrame, timeshift_seconds: float = 0
    ) -> pd.DataFrame:
        # If we already have a datetime index, skip the conversion
        if isinstance(input_df.index, pd.DatetimeIndex):
            input_df["Timestamp_NS"] = input_df.index.astype("int64")

            return input_df

        # Figure out which columns we are operating on and set the expected units

        if "Time_DAQmx" in input_df.columns:
            timestamp_col = "Time_DAQmx"
            timestamp_cols_to_delete = ["Time_PTP"]
            timestamp_units = "ns"
        elif "UTC2" in input_df.columns:
            timestamp_col = "UTC2"
            timestamp_cols_to_delete = ["UTC1"]
            timestamp_units = "s"
        elif "MOORING-DAQ_Timestamp" in input_df.columns:
            timestamp_col = "MOORING-DAQ_Timestamp"
            timestamp_cols_to_delete = []
            timestamp_units = "s"
        elif "Timestamp_NS" in input_df.columns:
            timestamp_col = "Timestamp_NS"
            timestamp_cols_to_delete = ["Timestamp"]
            timestamp_units = "ns"
        else:
            raise ValueError(
                "Unexpected timestamp column in columns: ", input_df.columns
            )

        # Convert the timestamp number to a datetime
        input_df[self.datetime_col_name] = pd.to_datetime(
            input_df[timestamp_col], unit=timestamp_units, utc=True, origin="unix"
        )

        if timeshift_seconds != 0:
            print(f"\tShifting time by {timeshift_seconds}...")
            input_df[self.datetime_col_name] = input_df[
                self.datetime_col_name
            ] + pd.to_timedelta(timeshift_seconds, unit="s")

        # Convert the datetime to nanoseconds
        input_df[self.timestamp_ns_col_name] = input_df[self.datetime_col_name].astype(
            "int64"
        )

        # Delete unused columns
        if len(timestamp_cols_to_delete) > 0:
            if timestamp_col != self.ns_timestamp_column:
                timestamp_cols_to_delete.append(timestamp_col)
            input_df = input_df.drop(timestamp_cols_to_delete, axis="columns")

        # Set the index to a datetime
        input_df = input_df.set_index([self.datetime_col_name])

        # Sort the dataframe by the new datetime index
        input_df = input_df.sort_index()

        # Is this dataframe valid
        index_is_increasing = input_df.index.is_monotonic_increasing
        ns_is_increasing = input_df[self.timestamp_ns_col_name].is_monotonic_increasing

        if index_is_increasing is False or ns_is_increasing is False:
            raise ValueError("Non monotonic timestamp")

        # Does the output dataframe have unique timestamps
        index_is_unique = input_df.index.duplicated().sum() == 0

        if not index_is_unique:
            print(
                f"Warning: Duplicate timestamp index found, removing {input_df.index.duplicated().sum()} duplicate timestamps"
            )
            print(f"Input df length: {len(input_df)}")
            input_df = input_df[~input_df.index.duplicated(keep="first")]
            print(f"Input removed duplicates df length: {len(input_df)}")

        return input_df

    def get_index_time_components(
        self, input_index: pd.DatetimeIndex
    ) -> TimeIndexUnits:
        i = input_index[0]
        return TimeIndexUnits(
            year=f"{i.year:04}",  # type: ignore
            month=f"{i.month:02}",  # type: ignore
            day=f"{i.day:02}",  # type: ignore
            hour=f"{i.hour:02}",  # type: ignore
            minute=f"{i.minute:02}",  # type: ignore
            second=f"{i.second:02}",  # type: ignore
        )

    def create_partition_path(
        self, output_folder, time_components: TimeIndexUnits, group
    ) -> Path:
        partition_sep = self.partition_separator
        # TODO: This is dependent on the grouper frequency string

        formatted_minute = str(math.floor(int(time_components.minute) / 30) * 30)

        return Path(
            output_folder,
            f"year{partition_sep}{time_components.year}",
            f"month{partition_sep}{time_components.month}",
            f"day{partition_sep}{time_components.day}",
            f"hour{partition_sep}{time_components.hour}",
            f"minute{partition_sep}{formatted_minute}",
            f"group{partition_sep}{group}",
        )

    def partition_df(
        self,
        input_file_location: str,
        project: str,
        group_name: str,
        output_folder: str,
    ):
        file_path = Path(input_file_location)
        input_df = pd.read_parquet(file_path)
        if group_name == "MOORING-DAQ":
            input_df = self.create_datetime_index(input_df, -15.152)
        else:
            input_df = self.create_datetime_index(input_df)

        grouped_dfs = input_df.groupby(pd.Grouper(freq=self.pd_frequency_string))

        for _, df in grouped_dfs:
            if len(df) == 0:
                continue

            if "Weather" in group_name:
                this_group_name = "_".join(file_path.name.split("_")[0:2])
                if "spectra" in file_path.name:
                    this_group_name = this_group_name + "_Wave_Spectral_Density"
                if "qoi" in file_path.name:
                    this_group_name = this_group_name + "_Wave_Variables"
                this_group_name = "Weather_" + this_group_name
                this_group_name = this_group_name.replace("-", "_")
            else:
                this_group_name = group_name

            time_components = self.get_index_time_components(df.index)  # type: ignore
            t = time_components
            partition_path = self.create_partition_path(
                output_folder, time_components, this_group_name
            )

            duration = pd.Timedelta(df.index[-1] - df.index[0]).seconds  # type: ignore
            # Change the minute to be the low

            num_samples = len(df)

            time_delta_ns_avg = df["Timestamp_NS"].diff().mean()
            sample_rate_hz = 1e9 / time_delta_ns_avg

            # Don't give weather group files the project name
            # if group_name == "Weather":
            if "Weather" in group_name:
                parquet_filename = f"{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}_{file_path.name.split('.')[0]}_{this_group_name}.parquet"
            else:
                parquet_filename = f"{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}_{duration}s_{num_samples}_samples_{sample_rate_hz:.2f}_hz_{project}_{group_name}.parquet"

            partition_path.mkdir(exist_ok=True, parents=True)

            if isinstance(df.index, pd.DatetimeIndex) is False:
                print(input_df.info())
                print(type(input_df.index))
                raise ValueError(
                    "Cannot partition dataframe that does not have a datetime index!"
                )

            df = df.sort_index()

            df.to_parquet(Path(partition_path, parquet_filename))

        return True

    def cleanup_single_partition(self, location):
        location = Path(location)
        files = list(location.rglob("*.parquet"))

        file_map = {}

        for file in files:
            filename = file.name
            # Get the path of the folder that the files are located in
            path = str(file.parent)

            if path not in file_map:
                file_map[path] = []

            file_map[path].append(filename)

        for location in file_map:
            num_files = len(file_map[location])
            if num_files > 1:
                original_files = file_map[location]
                first_file_wo_ext = original_files[0].replace(".parquet", "")
                group_name = first_file_wo_ext.split("_")[-1]
                if "Weather" in group_name:
                    continue

                project = first_file_wo_ext.split("_")[-2]

                print(f"\tCleaning up partition for {location}...")
                df = pd.read_parquet(location)

                if isinstance(df.index, pd.DatetimeIndex) is False:
                    raise ValueError(
                        "Cannot cleanup a partition dataframe that does not have a datetime index!"
                    )

                df = df.sort_index()

                t = self.get_index_time_components(df.index)
                duration = pd.Timedelta(df.index[-1] - df.index[0]).seconds

                duration_threshold = 598

                num_samples = len(df)

                time_delta_ns_avg = df["Timestamp_NS"].diff().mean()
                sample_rate_hz = 1e9 / time_delta_ns_avg

                parquet_filename = f"{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}_{duration}s_{num_samples}_samples_{sample_rate_hz:.2f}_hz_{project}_{group_name}.parquet"

                if duration < duration_threshold:
                    print(
                        f"Warning: Partition {parquet_filename} only contains {duration} seconds of data"
                    )

                for file in original_files:
                    file_to_delete = Path(location, file)
                    print(f"\t\tDeleting: {file}...")
                    file_to_delete.unlink()

                print(f"\t\tCreating {parquet_filename}...")
                df.to_parquet(Path(location, parquet_filename))

    def create_standardized_partition_path(self, output_folder, group) -> Path:
        partition_sep = self.partition_separator

        return Path(
            output_folder,
            f"group{partition_sep}{group}",
        )

    # Assume files are already partitioned by time
    # Create logical output groups by project subsystems
    def create_standardized_partitions(self, input_folder, output_folder):
        input_path = Path(input_folder).resolve()
        files = list(input_path.rglob("*.parquet"))

        for file in files:
            print(f"\tCreating standardized partition for {file.name}")
            file_parts = file.parts
            partition_parts = [part for part in file_parts if "=" in part]
            input_group_name = [part for part in partition_parts if "group" in part][
                0
            ].split("=")[-1]
            this_df = pd.read_parquet(file)

            if "Weather" in input_group_name:
                output_path = self.create_standardized_partition_path(
                    output_folder, input_group_name
                )
            elif input_group_name == "PowRaw":
                output_path = self.create_standardized_partition_path(
                    output_folder, "Gen"
                )
            elif input_group_name == "CurrentAI":
                output_path = self.create_standardized_partition_path(
                    output_folder, "RO"
                )
            elif input_group_name == "MOORING-DAQ":
                this_df = this_df[["Timestamp_NS", "MOORING-DAQ_LC-MOOR-001"]]
                this_df = this_df.rename(
                    {"MOORING-DAQ_LC-MOOR-001": "LC-MOOR-1001"}, axis="columns"
                )
                output_path = self.create_standardized_partition_path(
                    output_folder, "WEC"
                )
            elif input_group_name == "AIN":
                this_df = this_df[["Timestamp_NS", "POS-OS-1001", "PRESS-OS-2002"]]
                output_path = self.create_standardized_partition_path(
                    output_folder, "WEC"
                )
            elif input_group_name == "VoltageAI":
                pe_subsystem_columns = [
                    "PT-ON-2001",
                    "PT-ON-2002",
                    "PT-ON-2003",
                    "PT-ON-2004",
                    "PT-ON-2005",
                    "PT-ON-2006",
                    "PT-ON-2007",
                    "CT-ON-2001",
                    "CT-ON-2002",
                    "CT-ON-2003",
                    "CT-ON-2004",
                    "CT-ON-2005",
                    "CT-ON-2006",
                    "CT-ON-2007",
                    "Timestamp_NS",
                ]
                pe_df = this_df[pe_subsystem_columns]
                output_path = self.create_standardized_partition_path(
                    output_folder, "PE"
                )
                output_path.mkdir(parents=True, exist_ok=True)
                output_filename = file.name.replace("filtered", "standardized")

                pe_df.to_parquet(Path(output_path, output_filename))

                ro_df = this_df[["Timestamp_NS", "PRESS-ON-1001", "PRESS-ON-1002"]]
                output_path = self.create_standardized_partition_path(
                    output_folder, "RO"
                )
                output_path.mkdir(parents=True, exist_ok=True)
                output_filename = file.name.replace("filtered", "standardized")
                ro_df.to_parquet(Path(output_path, output_filename))

                continue
            else:
                continue
                # raise ValueError("Unexpected input group name")

            output_filename = file.name.replace("filtered", "standardized")
            output_path.mkdir(parents=True, exist_ok=True)
            this_df.to_parquet(Path(output_path, output_filename))

    def perfectly_resample(self, input_df, sample_rate_hz):
        input_df = input_df.drop(["Timestamp_NS"], axis="columns", errors="ignore")
        first_timestamp = input_df.index[0].floor("s")
        last_timestamp = input_df.index[-1].ceil("s") - pd.Timedelta("1ns")

        sample_rate_ms = (1 / sample_rate_hz) * 1_000

        perfect_sample_range_index = pd.date_range(
            first_timestamp, last_timestamp, freq=f"{sample_rate_ms}ms"
        )

        input_df = input_df.reindex(input_df.index.union(perfect_sample_range_index))
        input_df = input_df.interpolate(method="time", limit_direction="both")
        input_df = input_df.reindex(perfect_sample_range_index)

        return input_df

    def align_to_df(self, align_source_df, align_target_df):
        align_target_df = align_target_df.drop(
            ["Timestamp_NS"], axis="columns", errors="ignore"
        )

        # Reindexing
        print("\t\tReindexing...")
        align_target_df = align_target_df.reindex(
            align_target_df.index.union(align_source_df.index)
        )
        print("\t\tInterpolating...")
        aligned_df = align_target_df.interpolate(method="time", limit_direction="both")

        aligned_df = aligned_df.reindex(align_source_df.index)

        print("\t\tMerging...")

        result_df = pd.merge(
            align_source_df, aligned_df, left_index=True, right_index=True
        )

        # Not sure if this is a valid technique, but the interpolation leaves a
        # floating point for qc values from the target, this converts them to
        # the original type
        for col in result_df:
            if col.startswith("qc_"):
                result_df[col] = result_df[col].astype("uint8")

        return result_df

    def resample_wec_subsystem(self, input_folder):
        wec_subsystem_sample_rate_hz = 10
        wec_subsystem_path = Path(input_folder, "group=WEC")
        ain_parquet_files = list(wec_subsystem_path.rglob("**/*AIN*.parquet"))
        ain_parquet_files.sort()
        lc_parquet_files = list(wec_subsystem_path.rglob("**/*MOORING*.parquet"))

        zero_cross_parquet_file = Path(
            input_folder,
            "..",
            "z99_custom_extracted_parquet",
            "2024-HERO-Extracted-Partitioned-AIN-BAG-Zero-Crossing-Fix-Matlab.parquet",
        )
        zero_cross_df = pd.read_parquet(zero_cross_parquet_file)
        zero_cross_df = zero_cross_df.set_index(["Datetime"])
        zero_cross_df["vap_POS-OS-1001_UW"] = zero_cross_df["POS_OS_1001"]
        zero_cross_df = pd.DataFrame(zero_cross_df["vap_POS-OS-1001_UW"])

        zero_cross_df_resampled = self.perfectly_resample(
            zero_cross_df, wec_subsystem_sample_rate_hz
        )

        ain_df = pd.concat([pd.read_parquet(file) for file in ain_parquet_files])
        ain_df = ain_df.sort_index()

        ain_df_resampled = self.perfectly_resample(ain_df, wec_subsystem_sample_rate_hz)

        lc_df = pd.concat([pd.read_parquet(file) for file in lc_parquet_files])
        lc_df = lc_df.sort_index()

        lc_df_resampled = self.perfectly_resample(lc_df, wec_subsystem_sample_rate_hz)

        wec_subsystem_df = pd.merge(
            ain_df_resampled,
            lc_df_resampled,
            left_index=True,
            right_index=True,
            sort=True,
        )

        wec_subsystem_df = pd.merge(
            wec_subsystem_df,
            zero_cross_df_resampled,
            left_index=True,
            right_index=True,
            sort=True,
        )

        # Remove QC columns
        for col in wec_subsystem_df:
            if col.startswith("qc_"):
                wec_subsystem_df = wec_subsystem_df.drop([col], axis="columns")

        wec_subsystem_df.index.name = "Datetime"
        wec_subsystem_df["Timestamp_NS"] = wec_subsystem_df.index.astype("int64")
        wec_subsystem_df.insert(0, "Timestamp_NS", wec_subsystem_df.pop("Timestamp_NS"))

        wec_subsystem_df.insert(
            2, "vap_POS-OS-1001_UW", wec_subsystem_df.pop("vap_POS-OS-1001_UW")
        )

        grouped_dfs = wec_subsystem_df.groupby(
            pd.Grouper(freq=self.pd_frequency_string)
        )

        for _, this_df in grouped_dfs:
            time_components = self.get_index_time_components(this_df.index)  # type: ignore
            t = time_components
            output_path = self.create_standardized_partition_path(input_folder, "WEC")
            duration = pd.Timedelta(this_df.index[-1] - this_df.index[0]).seconds  # type: ignore
            # Change the minute to be the low

            num_samples = len(this_df)

            # time_delta_ns_avg = np.array(this_df.index.astype("int64").diff())
            time_delta_ns_avg = np.diff(np.array(this_df.index.astype("int64")))
            time_delta_ns_avg = np.nanmean(time_delta_ns_avg)
            sample_rate_hz = 1e9 / time_delta_ns_avg
            project = "HERO-WEC-Electric-Deployment"
            this_group_name = "WEC-Resampled"
            parquet_filename = f"{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}_{duration}s_{num_samples}_samples_{sample_rate_hz:.2f}_hz_{project}_{this_group_name}.parquet"
            print("\t\tWriting: ", parquet_filename)
            try:
                this_df.to_parquet(Path(output_path, parquet_filename))
            except OSError:
                print(
                    "Grouped dataframe does not have an existing path:",
                    parquet_filename,
                )
                continue

        # Delete source files
        [ain_file.unlink() for ain_file in ain_parquet_files]
        [lc_file.unlink() for lc_file in lc_parquet_files]

        return True

    def align_ro_subsystem(self, input_folder):
        ro_subsystem_path = Path(input_folder, "group=RO")

        current_ai_parquet_files = list(
            ro_subsystem_path.rglob("**/*CurrentAI*.parquet")
        )
        current_ai_parquet_files.sort()

        voltage_ai_parquet_files = list(
            ro_subsystem_path.rglob("**/*VoltageAI*.parquet")
        )
        voltage_ai_parquet_files.sort()

        if len(current_ai_parquet_files) != len(voltage_ai_parquet_files):
            raise ValueError(
                "RO subsystem Measurements do not have the same number of partition files"
            )

        count = 1

        files = zip(current_ai_parquet_files, voltage_ai_parquet_files)

        for file_set in files:
            current_ai_file = file_set[0]
            voltage_ai_file = file_set[1]

            print(
                f"\tWorking {count} of {len(current_ai_parquet_files)}:",
                current_ai_file.name,
            )

            current_ai_df = pd.read_parquet(current_ai_file)
            voltage_ai_df = pd.read_parquet(voltage_ai_file)

            aligned_df = self.align_to_df(current_ai_df, voltage_ai_df)
            aligned_df.insert(0, "Timestamp_NS", aligned_df.pop("Timestamp_NS"))

            time_components = self.get_index_time_components(aligned_df.index)  # type: ignore
            t = time_components
            output_path = self.create_standardized_partition_path(input_folder, "RO")
            duration = pd.Timedelta(aligned_df.index[-1] - aligned_df.index[0]).seconds  # type: ignore

            num_samples = len(aligned_df)

            time_delta_ns_avg = np.diff(np.array(aligned_df.index.astype("int64")))
            time_delta_ns_avg = np.nanmean(time_delta_ns_avg)
            sample_rate_hz = 1e9 / time_delta_ns_avg
            project = "HERO-WEC-Electric-Deployment"
            aligned_group_name = "RO-Aligned"
            parquet_filename = f"{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}_{duration}s_{num_samples}_samples_{sample_rate_hz:.2f}_hz_{project}_{aligned_group_name}.parquet"

            print("\t\tWriting: ", parquet_filename)

            aligned_df.to_parquet(Path(output_path, parquet_filename))

            current_ai_file.unlink()
            voltage_ai_file.unlink()

            count += 1

        return True

    def concat_standardized_partitions(self, input_folder):
        groups = ["Weather"]
        input_path = Path(input_folder).resolve()
        folders = [entry for entry in input_path.iterdir() if entry.is_dir()]

        for folder in folders:
            for group in groups:
                if group in folder.name:
                    files = [entry for entry in folder.iterdir() if entry.is_file()]
                    file_name = files[0].name
                    relevant_file_name = "_".join(file_name.split("_")[2:])

                    this_df = pd.read_parquet(folder)
                    this_df = this_df.sort_index()
                    time_components = self.get_index_time_components(this_df.index)  # type: ignore
                    t = time_components
                    duration = pd.Timedelta(this_df.index[-1] - this_df.index[0]).seconds  # type: ignore

                    num_samples = len(this_df)

                    time_delta_ns_avg = np.diff(np.array(this_df.index.astype("int64")))
                    time_delta_ns_avg = np.nanmean(time_delta_ns_avg)
                    sample_rate_hz = 1e9 / time_delta_ns_avg
                    parquet_filename = f"{t.year}-{t.month}-{t.day}_{t.hour}-{t.minute}_{duration}s_{num_samples}_samples_{sample_rate_hz:.4f}_hz_{relevant_file_name}"

                    # Delete all the source parquet files
                    [entry.unlink() for entry in folder.iterdir() if entry.is_file()]

                    this_df.to_parquet(Path(folder, parquet_filename))
