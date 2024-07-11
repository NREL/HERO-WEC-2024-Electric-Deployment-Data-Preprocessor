from dataclasses import dataclass, field

from pathlib import Path

from typing import List

import numpy as np
import pandas as pd


@dataclass
class ModaqHeroFilter:
    default_columns_to_drop: List[str] = field(
        default_factory=lambda: [
            "PRESS-ON-1003",
            "LVL-ON-1001",
            "AIN_1",  # PRESS-OS-1001 - Hydro Water Pressure
            "AIN_3",  # FLOW-OS-1001 - Hydro Water Flow
            "connection_timestamp",  # Bagfiles timestamp
        ]
    )
    timestamp_ns_col_name: str = "Timestamp_NS"
    datetime_col_name: str = "Datetime"
    # Start Time: 1713448800000000000 (4-18-24 10:00am EDT)
    # End Time: 1713531600000000000 (4-19-24 5:00am EDT)
    electric_deploy_start_ns_timestamp: int = 1713448800000000000
    electric_deploy_stop_ns_timestamp: int = 1713531600000000000

    def filter_unused_columns(self, input_df):
        input_df = input_df.drop(
            self.default_columns_to_drop, axis="columns", errors="ignore"
        )
        return input_df

    def filter_nan(self, input_df):
        # We can't operate on dataframes with na data, so we drop them here
        input_df.dropna(inplace=True)
        return input_df

    # https://s3.amazonaws.com/files.microstrain.com/GQ7+User+Manual/external_content/dcp/Data/filter_data/data/mip_field_filter_status.htm
    def add_gx5_status_flags(self, input_df, status_flag_column, filter_state_column):
        # Define status flag positions in a dictionary

        # Per https://www.microstrain.com/sites/default/files/applications/files/3dm-gx5-45_dcp_manual_8500-0064_rev_m_002.pdf
        # Page 145. Filter state determines flags 12 and 13. Filter state 1 is initializing, 2 and 3 are run. Not sure what to do about filter_state == 0

        # filter_state is 1
        filter_state_initialized = {
            "gx5_init_no_attitude": 12,
            "gx5_init_no_position_velocity": 13,
        }

        # Filter state is either 2 or 3
        filter_state_running = {
            "gx5_run_mag_bias_est_high_warning": 12,
            "gx5_run_ant_offset_correction_est_high_warning": 13,
        }

        flag_mapping = {
            "gx5_run_imu_unavailable": 0,
            "gx5_run_gps_unavailable": 1,
            "gx5_run_matrix_singularity": 3,
            "gx5_run_position_covariance_warning": 4,
            "gx5_run_velocity_covariance_warning": 5,
            "gx5_run_attitude_covariance_warning": 6,
            "gx5_run_nan_in_solution_warning": 7,
            "gx5_run_gyro_bias_est_high_warning": 8,
            "gx5_run_accel_bias_est_high_warning": 9,
            "gx5_run_gyro_scale_factor_est_high_warning": 10,
            "gx5_run_accel_scale_factor_est_high_warning": 11,
            "gx5_run_mag_hard_iron_est_high_warning": 14,
            "gx5_run_mag_soft_iron_est_high_warning": 15,
        }

        # Extract the status flag from the specified column
        status_flag = input_df[status_flag_column].values.astype(np.uint16)
        filter_state = input_df[filter_state_column].values.astype(np.uint16)

        # Extract bitflags using bitwise operations based on flag_mapping
        for flag, position in flag_mapping.items():
            input_df[flag] = (status_flag >> position) & 1
            # This is an inefficent hack that lets us set the status flags here, but interpolate between them later on
            input_df[flag] = input_df[flag].astype("float")

        # Process the filter state dependent flags
        for _, filter_state_dict in zip(
            ["filter_state_initialized", "filter_state_running"],
            [filter_state_initialized, filter_state_running],
        ):
            for flag, position in filter_state_dict.items():
                # Initialize all columns to 0
                input_df[flag] = 0

        # Set the filter state dependent flags based on the filter_state values
        for flag, position in filter_state_initialized.items():
            input_df.loc[filter_state == 1, flag] = (
                status_flag[filter_state == 1] >> position
            ) & 1

        for flag, position in filter_state_running.items():
            input_df.loc[filter_state == 2, flag] = (
                status_flag[filter_state == 2] >> position
            ) & 1
            input_df.loc[filter_state == 3, flag] = (
                status_flag[filter_state == 3] >> position
            ) & 1

        return input_df

    def setup_wec_ain(self, input_df):
        press_os_2002_slope = 15.625
        press_os_2002_offset = -62.5
        pos_os_1001_slope = 22.71164413
        pos_os_1001_offset = -89.92221262

        input_df["PRESS-OS-2002"] = (
            input_df["AIN_0"] * press_os_2002_slope
        ) + press_os_2002_offset
        input_df["POS-OS-1001"] = (
            input_df["AIN_2"] * pos_os_1001_slope
        ) + pos_os_1001_offset
        input_df = input_df.drop(["AIN_0", "AIN_2"], axis="columns")

        return input_df

    def setup_mooring_daq(self, input_df):
        lc_moor_001_slope = 0.002117658601513155
        lc_moor_001_offset = -17687.31280850665
        lc_col = "MOORING-DAQ_LC-MOOR-001"
        input_df[lc_col] = (input_df[lc_col] * lc_moor_001_slope) + lc_moor_001_offset

        return input_df

    def filter(self, input_parquet_file_path, output_file_folder, group_name):
        df = pd.read_parquet(input_parquet_file_path)

        input_filename = str(Path(input_parquet_file_path).name).replace(".parquet", "")

        df = self.filter_unused_columns(df)
        if "Weather" not in group_name:
            df = self.filter_nan(df)

        if group_name == "FSTAT":
            df = self.add_gx5_status_flags(df, "status_flags", "filter_state")

        if group_name == "AIN":
            df = self.setup_wec_ain(df)

        if group_name == "MOORING-DAQ":
            df = self.setup_mooring_daq(df)

        if len(df) == 0:
            return False

        # Don't output dataframes that are outside of the desired time frame
        first_ns_timestamp = df[self.timestamp_ns_col_name].iloc[0]
        last_ns_timestamp = df[self.timestamp_ns_col_name].iloc[-1]

        is_before_area_of_interest = (
            last_ns_timestamp < self.electric_deploy_start_ns_timestamp
        )
        is_after_area_of_interest = (
            first_ns_timestamp > self.electric_deploy_stop_ns_timestamp
        )

        if is_before_area_of_interest or is_after_area_of_interest:
            area_of_interest_start = pd.to_datetime(
                self.electric_deploy_start_ns_timestamp,
                unit="ns",
                origin="unix",
                utc=True,
            )
            area_of_interest_end = pd.to_datetime(
                self.electric_deploy_stop_ns_timestamp,
                unit="ns",
                origin="unix",
                utc=True,
            )
            print(
                f"Skipping {input_filename} because it is outside of area of interest {area_of_interest_start} to {area_of_interest_end}"
            )
            return

        if len(df.columns) > 1:
            output_parquet_folder = Path(output_file_folder).resolve()
            output_parquet_folder.mkdir(parents=True, exist_ok=True)

            output_parquet_path = str(
                Path(output_file_folder, f"{input_filename}_filtered.parquet")
            )

            return df.to_parquet(output_parquet_path)
