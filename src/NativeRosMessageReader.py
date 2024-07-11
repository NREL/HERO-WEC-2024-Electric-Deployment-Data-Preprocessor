import functools
from dataclasses import dataclass
from typing import List
import pandas as pd
from .RosReader import RosReader


@dataclass
class NativeRosMessageReader(RosReader):
    """
    A class for reading native ROS messages and converting them into Pandas DataFrame.

    Attributes:
        topic_id (str): The ID of the ROS topic.
        message_name (str): The name of the ROS message associated with the topic.
        attrs (List[str]): A list of attributes to extract from the ROS message.
    """

    topic_id: str
    message_name: str
    attrs: List[str]

    def __post_init__(self):
        """
        Initialize the NativeRosMessageReader object after its creation.

        """
        super().__init__(self.topic_id, self.message_name)

    def reset(self):
        super().__init__(self.topic_id, self.message_name)

    def _setup_data_dict(self):
        """
        Initialize a dictionary to store parsed message data.

        Returns:
            dict: An empty dictionary to store parsed message data.

        """
        result = {}
        for col in self.attrs:
            result[col] = []
        return result

    def rgetattr(self, obj, attr, *args):
        """
        Recursively get attribute from nested objects.

        Args:
            obj: The object to retrieve the attribute from.
            attr (str): The name of the attribute.
            *args: Additional arguments.

        Returns:
            object: The value of the attribute.

        """

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    def deserialize(self, raw_message):
        """
        Deserialize a raw message and store its data.

        Args:
            raw_message: The raw message to deserialize.

        """
        data = self.typestore.deserialize_cdr(raw_message, self.topic_message_name)

        for col in self.attrs:
            col_data = self.rgetattr(data, col)
            self.data[col].append(col_data)

    def to_df(self):
        """
        Convert the message data into a Pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the message data.

        """
        df = pd.DataFrame(self.data)
        df = df.rename(
            columns={"header.stamp.sec": "sec", "header.stamp.nanosec": "nanosec"}
        )
        return self.setup_timestamps(df)
