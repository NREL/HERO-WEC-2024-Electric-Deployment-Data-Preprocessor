import pandas as pd
from rosbags.typesys import Stores, get_typestore


class RosReader:
    """
    A class for reading ROS messages and converting them into a columnar format.

    Attributes:
        topic_id (str): The ID of the ROS topic.
        topic_message_name (str): The name of the ROS message associated with the topic.
        typestore: The typestore object for ROS message types.
        data (dict): Dictionary containing parsed message data.
    """

    def __init__(self, topic_id: str, topic_message_name: str):
        """
        Initialize the RosReader object.

        Args:
            topic_id (str): The ID of the ROS topic.
            topic_message_name (str): The name of the ROS message associated with the topic.
        """
        self.topic_id = topic_id
        self.topic_message_name = topic_message_name
        self.typestore = get_typestore(Stores.ROS2_IRON)
        self.data = self._setup_data_dict()

    def _setup_data_dict(self):
        """
        Parse the message definition into a dictionary.

        Returns:
            dict: A dictionary containing parsed message data.
        """
        raise NotImplementedError("Subclasses must implement _setup_data_dict method")

    def to_df(self):
        """
        Convert the message data into a Pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the message data.
        """
        df = pd.DataFrame(self.data)
        return self.setup_timestamps(df)

    def setup_timestamps(
        self, input_df, drop_original_columns=True, setup_timestamp_index=False
    ):
        """
        Set up timestamps in the DataFrame.

        Args:
            input_df (pandas.DataFrame): The DataFrame to set up timestamps in.
            drop_original_columns (bool): Whether to drop original timestamp columns.
            setup_timestamp_index (bool): Whether to set up timestamp index.

        Returns:
            pandas.DataFrame: The DataFrame with timestamps set up.
        """
        columns = input_df.columns
        if "sec" not in columns or "nanosec" not in columns:
            input_df["Timestamp"] = pd.to_datetime(
                self.data["connection_timestamp"], unit="ns", origin="unix", utc=True
            )
        else:
            input_df["Timestamp"] = pd.to_datetime(
                input_df["sec"], unit="s"
            ) + pd.to_timedelta(input_df["nanosec"], unit="ns")

        input_df["Timestamp_NS"] = input_df["Timestamp"].astype("int64")

        if drop_original_columns is True:
            if "sec" in columns and "nanosec" in columns:
                input_df = input_df.drop(columns=["sec", "nanosec"])
            else:
                input_df = input_df.drop(columns=["connection_timestamp"])

        input_df.insert(0, "Timestamp", input_df.pop("Timestamp"))
        input_df.insert(0, "Timestamp_NS", input_df.pop("Timestamp_NS"))

        if setup_timestamp_index:
            input_df.set_index(["Timestamp"], inplace=True)
            input_df.index = input_df.index.tz_localize("UTC")

        return input_df
