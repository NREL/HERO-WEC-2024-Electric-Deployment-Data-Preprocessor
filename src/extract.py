import os
import sqlite3

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import mhkit
import xarray as xr

from .NativeRosMessageReader import NativeRosMessageReader
from .CustomRosMessageReader import CustomRosMessageReader
from .ModaqTdmsExtractor import ModaqTdmsExtractor


def get_dirs_in_folder(input_folder: str) -> List[str]:
    """Return a list of directory names in the specified folder.

    Parameters:
        input_folder (str): The path to the folder.

    Returns:
        List[str]: A list of directory names.
    """
    input_path = Path(input_folder).resolve()
    return [str(entry) for entry in input_path.iterdir() if entry.is_dir()]


def get_files_in_folder(input_folder: str, file_extension: str) -> List[Path]:
    """Return a list of file names in the specified folder with the given file extension.

    Parameters:
        input_folder (str): The path to the folder.
        file_extension (str): The file extension to filter.

    Returns:
        List[str]: A list of file names.
    """
    input_path = Path(input_folder).resolve()
    return list(input_path.rglob(f"*.{file_extension}"))


def extract_tdms_file(
    tdms_file: str, output_folder: str, output_file_prefix: str
) -> bool:
    """Extract data from a TDMS file and save it in the specified output folder with the given prefix.

    Parameters:
        tdms_file (str): The path to the TDMS file.
        output_folder (str): The path to the output folder.
        output_file_prefix (str): The prefix for the output file.

    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    extractor = ModaqTdmsExtractor(
        output_filetypes=["parquet"], output_file_prefix=output_file_prefix
    )

    extractor.extract(tdms_file, output_folder)
    return True


def extract_modaq_folder(input_folder: str, output_folder: str) -> bool:
    """Extract data from all MODAQ files in the input folder and save them in the output folder.

    Parameters:
        input_folder (str): The path to the input folder.
        output_folder (str): The path to the output folder.

    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    valid_extensions = ["done", "tdms"]
    input_path = Path(input_folder).resolve()
    tdms_files = []

    for ext in valid_extensions:
        this_tdms_files = list(input_path.rglob(f"*.{ext}"))
        tdms_files.extend(this_tdms_files)

    total = len(tdms_files)
    print(f"\tExtracting {total} tdms files from {input_folder}")

    # Use ProcessPoolExecutor to parallelize the extraction process
    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(extract_tdms_file, str(file), output_folder, "")
            for file in tdms_files
        ]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()  # Retrieve the result to catch any exceptions
                print(f"\t\tSuccessfully extracted file {count} of {total}")
            except Exception as e:
                print(f"\t\tFailed to extract file {count} of {total}: {e}")

    return True


def extract_mooring_daq_folder(input_folder: str, output_folder: str) -> bool:
    """Extract data from all HERO WEC Mooring DAQ CSV files in the input folder and save them in the output folder.

    Parameters:
        input_folder (str): The path to the input folder.
        output_folder (str): The path to the output folder.
        metadata (Dict[str, Any]): Metadata dictionary.

    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    valid_extensions = ["csv"]
    input_path = Path(input_folder).resolve()
    csv_files = []

    for ext in valid_extensions:
        this_csv_files = list(input_path.rglob(f"*.{ext}"))
        csv_files.extend(this_csv_files)

    count = 1
    total = len(csv_files)

    hero_mooring_daq_channel_names = [
        "Timestamp",
        "Temperature",
        "Acc_X",
        "Acc_Y",
        "Acc_Z",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z",
        "Angle_X",
        "Angle_Y",
        "Angle_Z",
        "LC-MOOR-001",
    ]
    channel_names = [f"MOORING-DAQ_{ch}" for ch in hero_mooring_daq_channel_names]

    print(f"\tExtracting {total} csv files from {input_folder}")
    for file in csv_files:
        print(f"\tExtracting {file} - {count} of {total}")

        df = pd.read_csv(
            file,
            sep=",",
            header=0,
            usecols=list(range(len(channel_names))),
            names=channel_names,
        )

        input_file_name_without_extension = file.name.split(".")[0]
        group_output_folder = Path(output_folder, "MOORING-DAQ")
        group_output_folder.mkdir(exist_ok=True, parents=True)

        df = df.dropna(axis=0, how="any")
        df.to_parquet(
            Path(group_output_folder, f"{input_file_name_without_extension}.parquet")
        )

        count += 1

    return True


def save_ros_df(ros_df, output_fname, group, output_folder):
    output_parquet_path = Path(output_folder).resolve()
    this_output_folder = Path(output_parquet_path, f"{group}-BAG")
    this_output_folder.mkdir(exist_ok=True, parents=True)
    sanitized_output_fname = output_fname.replace(".db3", "")
    this_output_path = Path(
        this_output_folder, f"{sanitized_output_fname}-{group}.parquet"
    )
    ros_df.to_parquet(this_output_path)


def process_bagfile(bagfile, output_folder):
    # Re-instantiate CustomRosMessageReader and NativeRosMessageReader inside the function
    HERO_AIN_DEF = """
    std_msgs/Header header
    float64 AIN_0
    float64 AIN_1
    float64 AIN_2
    float64 AIN_3
    """
    hero_ain = CustomRosMessageReader(HERO_AIN_DEF, "/ain", "ain/msg/remote_ain")

    hero_gps = NativeRosMessageReader(
        "/gps",
        "sensor_msgs/msg/NavSatFix",
        [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "latitude",
            "longitude",
            "altitude",
            "status.status",
            "status.service",
        ],
    )

    HERO_FILTER_STATUS_LORD_GX5 = """
    uint16 filter_state
    uint16 dynamics_mode
    uint16 status_flags
    """
    hero_fstat = CustomRosMessageReader(
        HERO_FILTER_STATUS_LORD_GX5,
        "/fstat",
        "microstrain_inertial_msgs/msg/MipFilterStatus",
    )

    HERO_HEADING_LORD_GX5 = """
    float32 heading_deg
    float32 heading_rat
    uint16 status_flags
    """
    hero_heading = CustomRosMessageReader(
        HERO_HEADING_LORD_GX5,
        "/hdg",
        "microstrain_inertial_msgs/msg/FilterHeading",
    )

    hero_odom = NativeRosMessageReader(
        "/odom",
        "nav_msgs/msg/Odometry",
        [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "pose.pose.position.x",
            "pose.pose.position.y",
            "pose.pose.position.z",
            "pose.pose.orientation.x",
            "pose.pose.orientation.y",
            "pose.pose.orientation.z",
            "pose.pose.orientation.w",
            "twist.twist.linear.x",
            "twist.twist.linear.y",
            "twist.twist.linear.z",
            "twist.twist.angular.x",
            "twist.twist.angular.y",
            "twist.twist.angular.z",
        ],
    )

    hero_imf = NativeRosMessageReader(
        "/imf",
        "sensor_msgs/msg/Imu",
        [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "orientation.x",
            "orientation.y",
            "orientation.z",
            "orientation.w",
            "angular_velocity.x",
            "angular_velocity.y",
            "angular_velocity.z",
            "linear_acceleration.x",
            "linear_acceleration.y",
            "linear_acceleration.z",
        ],
    )

    hero_imu = NativeRosMessageReader(
        "/imu",
        "sensor_msgs/msg/Imu",
        [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "orientation.x",
            "orientation.y",
            "orientation.z",
            "orientation.w",
            "angular_velocity.x",
            "angular_velocity.y",
            "angular_velocity.z",
            "linear_acceleration.x",
            "linear_acceleration.y",
            "linear_acceleration.z",
        ],
    )

    hero_mag = NativeRosMessageReader(
        "/mag",
        "sensor_msgs/msg/MagneticField",
        [
            "header.stamp.sec",
            "header.stamp.nanosec",
            "magnetic_field.x",
            "magnetic_field.y",
            "magnetic_field.z",
        ],
    )

    try:
        conn = sqlite3.connect(bagfile)
        cursor = conn.cursor()

        topics_data = cursor.execute("SELECT id, name, type FROM topics").fetchall()
        topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
        topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}

        for this_topic_id in topic_id.keys():
            topic_enum = topic_id[this_topic_id]
            rows = cursor.execute(
                f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_enum}"
            ).fetchall()

            for row in rows:
                row_timestamp = row[0]
                row_data = row[1]
                if this_topic_id == "/ain":
                    hero_ain.deserialize(row_data, row_timestamp)
                elif this_topic_id == "/fstat":
                    hero_fstat.deserialize(row_data, row_timestamp)
                elif this_topic_id == "/gps":
                    hero_gps.deserialize(row_data)
                elif this_topic_id == "/hdg":
                    hero_heading.deserialize(row_data, row_timestamp)
                elif this_topic_id == "/imf":
                    hero_imf.deserialize(row_data)
                elif this_topic_id == "/imu":
                    hero_imu.deserialize(row_data)
                elif this_topic_id == "/mag":
                    hero_mag.deserialize(row_data)
                elif this_topic_id == "/odom":
                    hero_odom.deserialize(row_data)

        save_ros_df(hero_ain.to_df(), Path(bagfile).name, "AIN", output_folder)
        save_ros_df(hero_gps.to_df(), Path(bagfile).name, "GPS", output_folder)
        save_ros_df(hero_odom.to_df(), Path(bagfile).name, "ODOM", output_folder)
        save_ros_df(hero_imf.to_df(), Path(bagfile).name, "IMF", output_folder)
        save_ros_df(hero_imu.to_df(), Path(bagfile).name, "IMU", output_folder)
        save_ros_df(hero_fstat.to_df(), Path(bagfile).name, "FSTAT", output_folder)
        save_ros_df(hero_heading.to_df(), Path(bagfile).name, "HEADING", output_folder)
        save_ros_df(hero_mag.to_df(), Path(bagfile).name, "MAG", output_folder)

        hero_ain.reset()
        hero_gps.reset()
        hero_odom.reset()
        hero_imf.reset()
        hero_imu.reset()
        hero_fstat.reset()
        hero_heading.reset()
        hero_mag.reset()
    except Exception as e:
        raise e
    finally:
        conn.close()


def extract_remote_daq_bagfiles(input_folder: str, output_folder: str) -> bool:
    """Extract data from all HERO WEC Mooring DAQ CSV files in the input folder and save them in the output folder.

    Parameters:
        input_folder (str): The path to the input folder.
        output_folder (str): The path to the output folder.

    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    print(f"\tDeserializing bagfiles in {input_folder}...")

    rosbag2_files = get_files_in_folder(input_folder, "db3")
    rosbag2_files.sort()

    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_bagfile, bagfile, output_folder)
            for bagfile in rosbag2_files
        ]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                print(
                    f"\t\tSuccessfully processed bagfile {count} of {len(rosbag2_files)}"
                )
            except Exception as e:
                print(
                    f"\t\tFailed to process bagfile {count} of {len(rosbag2_files)}: {e}"
                )

    return True


def extract_ndbc_met_text_file(file: Path) -> pd.DataFrame:
    file_name = file.name
    ndbc_station_id = file_name.split("-")[0]
    df, meta = mhkit.wave.io.ndbc.read_file(str(file))

    if meta is not None:
        column_rename_map = {}
        for column in df.columns:
            column_rename_map[column] = f"{column}_{meta[column]}"

        df = df.rename(columns=column_rename_map)

    df = df.add_prefix(f"NDBC_{ndbc_station_id}_")

    # Drop columns where everything is NaN
    df = df.dropna(axis=1, how="all")

    df = df.sort_index()
    return df


def extract_cdip_nc_spectra(file: Path):
    ds = xr.open_dataset(file)
    station_id = ds.attrs["cdip_station_id"]
    platform_name = ds.attrs["platform_name"].split(",")[0].replace(" ", "_")
    fname_prefix = f"CDIP_{station_id}_{platform_name}"

    time_index = pd.to_datetime(
        ds["waveTime"].values, unit="ns", utc=True, origin="unix"
    )
    frequency_index = ds["waveFrequency"].values
    # Parquet files cant store a column name that is a numeric type
    frequency_index = [str(val) for val in frequency_index]
    wave_energy_density = ds["waveEnergyDensity"].values

    spectra_df = pd.DataFrame(
        wave_energy_density, index=time_index, columns=frequency_index
    )

    spectra_df.index.name = "waveFrequencyTime"

    wave_qoi_df = ds[
        [
            "waveHs",
            "waveTp",
            "waveTa",
            "waveDp",
            "wavePeakPSD",
            "waveTz",
            "waveFlagPrimary",
            "waveFlagSecondary",
        ]
    ].to_pandas()
    station_depth = ds["metaWaterDepth"].values  # Singular value
    station_depth = np.full(len(wave_qoi_df), station_depth)

    wave_qoi_df["metaWaterDepth"] = station_depth
    wave_qoi_df.index = pd.to_datetime(
        wave_qoi_df.index, unit="ns", utc=True, origin="unix"
    )

    return {
        "spectra_df": spectra_df,
        "wave_qoi_df": wave_qoi_df,
        "station_id": station_id,
        "fname_prefix": fname_prefix,
    }


def extract_weather(input_folder: str, output_folder: str):
    ndbc_txt_files = get_files_in_folder(input_folder, "txt")
    cdip_nc_files = get_files_in_folder(input_folder, "nc")

    ndbc_met_files = [file for file in ndbc_txt_files if "Meterological" in file.name]

    start_datetime = "2024-04-18 00:00:00"
    end_datetime = "2024-04-20 23:59:00"

    for file in ndbc_met_files:
        df = extract_ndbc_met_text_file(file)
        df = df.loc[start_datetime:end_datetime]
        output_location = Path(output_folder, "Weather")
        output_location.mkdir(exist_ok=True, parents=True)

        df.to_parquet(
            Path(
                output_folder,
                "Weather",
                f"NDBC_{file.name.replace('.txt', '.parquet').replace(' ', '_')}",
            )
        )

    for nc_file in cdip_nc_files:
        result = extract_cdip_nc_spectra(nc_file)
        output_location = Path(output_folder, "Weather")
        output_location.mkdir(exist_ok=True, parents=True)
        file_name = nc_file.name.replace(".nc", ".parquet")

        spectra_df = result["spectra_df"]
        spectra_df = spectra_df.loc[start_datetime:end_datetime]

        spectra_df.to_parquet(
            Path(
                output_folder,
                "Weather",
                f"{result['fname_prefix']}_spectra_{file_name}",
            )
        )

        wave_qoi_df = result["wave_qoi_df"]
        wave_qoi_df = wave_qoi_df.loc[start_datetime:end_datetime]

        wave_qoi_df.to_parquet(
            Path(
                output_folder,
                "Weather",
                f"{result['fname_prefix']}_wave_qoi_{file_name}",
            )
        )
