import shutil

from dataclasses import dataclass
from pathlib import Path

import mhkit
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter  # Equivalent to smoothdata



@dataclass
class HeroVAP:
    partition_folder: str
    vap_output_folder: str
    vap_column_prefix: str = "vap_"
    calc_column_prefix: str = "calc_"

    def vap_partition_folder(self):
        folder = Path(self.partition_folder)

        group_folders = [entry for entry in folder.iterdir() if entry.is_dir()]

        output_folder = Path(self.vap_output_folder).resolve()
        output_folder.mkdir(exist_ok=True, parents=True)

        self.output_folder = output_folder

        groups = []
        for folder in group_folders:
            parts = folder.parts
            partition_parts = [part for part in parts if "=" in part]
            group = [part for part in partition_parts if "group" in part][0]
            groups.append(group)

        for group in groups:
            self.vap_group(group)

    def vap_group(self, group):
        folder = Path(self.partition_folder, group)
        parquet_files = folder.rglob("**/*.parquet")
        for file in parquet_files:
            self.calculate_vap(file, group)

    def calculate_vap(self, this_file, this_group):
        sanitized_group = this_group.split("=")[-1]
        print(f"\tCalculating {this_file.name} vap for {sanitized_group}")
        output_df = None

        if "WEC" in this_group:
            if "Resampled" in this_file.name:
                this_df = pd.read_parquet(this_file)
                output_df = self.calculate_wec_vap(this_df)
        if "RO" in this_group:
            if "Aligned" in this_file.name:
                this_df = pd.read_parquet(this_file)
                output_df = self.calculate_ro_vap(this_df)
        if "Gen" in this_group:
            this_df = pd.read_parquet(this_file)
            output_df = self.calculate_gen_vap(this_df)

        if "PE" in this_group:
            this_df = pd.read_parquet(this_file)
            output_df = self.calculate_pe_vap(this_df)
        if "Weather" in this_group:
            if "Spectral" in this_group:
                this_df = pd.read_parquet(this_file)
                station_id = sanitized_group.replace("Weather_", "").replace(
                    "_Wave_Spectral_Density", ""
                )
                if "243" in station_id:
                    station_depth = 21.0  # Meters
                elif "433" in station_id:
                    station_depth = 18.13999939  # Meters
                else:
                    raise ValueError("Unexpected Wave Resource Data")

                output_df = self.calculate_wave_resource_vap(
                    this_df, station_id, station_depth
                )
            else:
                # Needed to manipulate the Timestamp_NS column in the writer below
                output_df = pd.read_parquet(this_file)

        if output_df is not None:
            if list(output_df.columns)[0] != "Timestamp_NS":
                output_df.insert(0, "Timestamp_NS", output_df.pop("Timestamp_NS"))

            output_path = Path(self.output_folder, this_group)
            output_path.mkdir(exist_ok=True, parents=True)
            output_df.to_parquet(
                Path(
                    output_path,
                    f"{this_file.name.replace('.parquet', '')}_vap.parquet",
                )
            )
        else:
            print(this_group)
            print(f"Copying {this_file} to {self.output_folder}")
            this_output_folder = Path(self.output_folder, this_group)
            this_output_folder.mkdir(exist_ok=True, parents=True)

            shutil.copy(this_file, this_output_folder)

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

    def calculate_wec_vap(self, df) -> pd.DataFrame:
        # WEC Geometry
        SR_Gear_Ratio = 4.5  # Gear ratio of spring return single stage
        GB_Gear_Ratio = 11.27  # Gear ratio of gearbox
        D_Winch = 8.5

        df["Time_Elapsed_NS"] = df["Timestamp_NS"] - df["Timestamp_NS"].iloc[0]
        df["Time_Elapsed"] = df["Time_Elapsed_NS"] * (10**-9)

        enc = "vap_POS-OS-1001_UW"
        moor = "LC-MOOR-1001"
        smooth = "S"
        velocity = "V"

        enc_smooth = f"{enc}_{smooth}"
        enc_vel = f"{self.calc_column_prefix}{enc.replace(self.vap_column_prefix, '')}_{smooth}_{velocity}"
        winch_vel = f"{self.calc_column_prefix}Winch_{velocity}"
        gearbox_vel = f"{self.calc_column_prefix}Gearbox_{velocity}"
        absorbed_power = f"{self.calc_column_prefix}P_abs"

        # 1. Smooth Encoder Position
        # Smoothed encoder position data used to calculate encoder velocity - VERIFY INPUTS, this is ChatGPT's approximation of Matlab smoothdata inputs from LAMP
        window_length = 51
        polyorder = 3
        df[enc_smooth] = savgol_filter(df[enc], window_length, polyorder)

        # Calc_V_Encoder [rpm]
        df[enc_vel] = np.gradient(df[enc_smooth], df["Time_Elapsed"]) * (60 / 360)

        # Calc_V_Winch [rpm]
        df[winch_vel] = df[enc_vel] * SR_Gear_Ratio

        # Calc_V_Gearbox [rpm] - if statement used to represent oneway clutch
        # Negative encoder velocity corresponds to the winch unspooling
        df[gearbox_vel] = (
            np.where(df[winch_vel] < 0, df[winch_vel] * GB_Gear_Ratio, 0)
        ) * -1.0

        # Calc_P_abs [kW]
        if moor in df.columns:
            df[absorbed_power] = np.where(
                df[winch_vel] < 0,
                (df[moor] * 0.5 * D_Winch) * abs(df[winch_vel] / 5252) * 0.7457,
                0,
            )

        df = df.drop(["Time_Elapsed_NS", "Time_Elapsed"], axis="columns")

        return df

    def calculate_ro_vap(self, df) -> pd.DataFrame:
        nan_mask_df = self.nan_mask_df_per_qc(df)

        # Calc_P_Hydraulic_2 - Hydraulic power at RO system inlet
        # [kW] Instantaneous hydraulic power at RO inlet
        df[f"{self.calc_column_prefix}P_Hydraulic_2"] = (
            (nan_mask_df["PRESS-ON-1001"] * nan_mask_df["FLOW-ON-1001"]) / 1714
        ) * 0.7457

        # Calc_P_Hydraulic_3 -Hydraulic power at Clark pump inlet
        df[f"{self.calc_column_prefix}P_Hydraulic_3"] = (
            (nan_mask_df["PRESS-ON-1002"] * nan_mask_df["FLOW-ON-1002"]) / 1714
        ) * 0.7457  # [kW] Instantaneous hydraulic power at Clark pump inlet

        # Calc_Acc_Flow - Calculated flow rate into accumulators
        df[f"{self.calc_column_prefix}Acc_Flow"] = (
            nan_mask_df["FLOW-ON-1001"] - nan_mask_df["FLOW-ON-1002"]
        )  # [gpm] Instantaneous flowrate into onshore accumulators

        # Calc_Filter_DP - Calculated pressure loss across prefilters
        df[f"{self.calc_column_prefix}Filter_DP"] = (
            nan_mask_df["PRESS-ON-1001"] - nan_mask_df["PRESS-ON-1002"]
        )  # [psi] Instantaneous pressure drop across RO prefilters

        # Calc_Recovery_Ratio - Calculated recovery ratio of RO system
        df[f"{self.calc_column_prefix}Recovery_Ratio"] = (
            nan_mask_df["FLOW-ON-1004"] / nan_mask_df["FLOW-ON-1003"]
        )  # [-] Instantaneous RO recovery ratio

        return df

    def calculate_gen_vap(self, df) -> pd.DataFrame:
        nan_mask_df = self.nan_mask_df_per_qc(df)

        watts_to_kw = 1000
        df[f"{self.calc_column_prefix}P_WEC_Electric_Out"] = (
            (nan_mask_df["PT-ON-1001"] * (nan_mask_df["CT-ON-1001"] / watts_to_kw))
            + (nan_mask_df["PT-ON-1002"] * (nan_mask_df["CT-ON-1001"] / watts_to_kw))
            + (nan_mask_df["PT-ON-1003"] * (nan_mask_df["CT-ON-1003"] / watts_to_kw))
        )  # [kW] Three phase output power of generator. Setup for 1 kHz data - change 1000 in equation if sample rate changes
        return df

    def calculate_pe_vap(self, df):
        nan_mask_df = self.nan_mask_df_per_qc(df)
        watts_to_kw = 1000

        # Calc_P_Electric_1 - Calculated instantaneous DC power at rectifier output
        df[f"{self.calc_column_prefix}P_Electric_1"] = (
            nan_mask_df["PT-ON-2001"] * nan_mask_df["CT-ON-2001"]
        ) / watts_to_kw  # [kW]

        # Calc_P_Electric_2 - Calculated instantaneous DC power at charge controller input
        df[f"{self.calc_column_prefix}P_Electric_2"] = (
            nan_mask_df["PT-ON-2002"] * nan_mask_df["CT-ON-2002"]
        ) / watts_to_kw  # [kW]

        # Calc_P_Electric_3 - Calculated instantaneous DC power at charge controller output
        df[f"{self.calc_column_prefix}P_Electric_3"] = (
            nan_mask_df["PT-ON-2003"] * nan_mask_df["CT-ON-2003"]
        ) / watts_to_kw  # [kW]

        # Calc_Eta_CC_I - Calculated instantaneous charge controller efficiency
        try:
            df[f"{self.calc_column_prefix}Eta_CC_I"] = (
                nan_mask_df[f"{self.calc_column_prefix}P_Electric_3"]
                / nan_mask_df[f"{self.calc_column_prefix}P_Electric_2"]
            ) * 100  # [%]
        except KeyError:
            df[f"{self.calc_column_prefix}Eta_CC_I"] = np.nan

        # Calc_P_Electric_4 - Calculated instantaneous DC power input into batteries - Note: negative value expected when batteries are discharged to operate pumps
        df[f"{self.calc_column_prefix}P_Electric_4"] = (
            nan_mask_df["PT-ON-2004"] * nan_mask_df["CT-ON-2004"]
        ) / watts_to_kw  # [kW]

        # Calc_P_Electric_5 - Calculated instantaneous DC power supplied to pump controllers
        df[f"{self.calc_column_prefix}P_Electric_5"] = (
            nan_mask_df["PT-ON-2005"] * nan_mask_df["CT-ON-2005"]
        ) / watts_to_kw  # [kW]

        # Calc_P_Electric_P1 - Calculated instantaneous DC power supplied to pump 1
        df[f"{self.calc_column_prefix}P_Electric_P1"] = (
            nan_mask_df["PT-ON-2006"] * nan_mask_df["CT-ON-2006"]
        ) / watts_to_kw  # [kW]

        # Calc_P_Electric_P2 - Calculated instantaneous DC power supplied to pump 1
        df[f"{self.calc_column_prefix}P_Electric_P2"] = (
            nan_mask_df["PT-ON-2007"] * nan_mask_df["CT-ON-2007"]
        ) / watts_to_kw  # [kW]

        return df

    def calculate_wave_resource_vap(self, df, station_id, station_depth):
        df = df.drop(["Timestamp_NS"], axis="columns")
        df.columns = [float(col) for col in df.columns]
        df = df.T

        Hm_0 = mhkit.wave.resource.significant_wave_height(df)
        T_z = mhkit.wave.resource.average_zero_crossing_period(df)
        T_avg = mhkit.wave.resource.average_crest_period(df)
        T_m = mhkit.wave.resource.average_wave_period(df)
        T_p = mhkit.wave.resource.peak_period(df)
        T_e = mhkit.wave.resource.energy_period(df)
        J = mhkit.wave.resource.energy_flux(df, station_depth)

        # Use join all columns in "other" with Hm_0 using the index
        result_df = Hm_0.join([T_z, T_avg, T_m, T_p, T_e, J], how="left")  # type: ignore

        result_df = result_df.add_prefix(
            f"{self.vap_column_prefix}{station_id}_Spectral_"
        )

        result_df["Timestamp_NS"] = pd.to_datetime(
            result_df.index, utc=True, origin="unix"
        ).astype("int64")

        result_df.insert(0, "Timestamp_NS", result_df.pop("Timestamp_NS"))

        return result_df
