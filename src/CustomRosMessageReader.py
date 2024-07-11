from dataclasses import dataclass
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from .RosReader import RosReader


@dataclass
class CustomRosMessageReader(RosReader):
    """
    A class for reading custom ROS messages and converting them into Pandas DataFrame.

    Attributes:
        message_definition (str): The definition of the ROS message.
        topic_id (str): The ID of the ROS topic.
        message_name (str): The name of the ROS message associated with the topic.
        columns (list): A list of column names parsed from the message definition.
    """

    message_definition: str
    topic_id: str
    message_name: str

    def __post_init__(self):
        """
        Initialize the CustomRosMessageReader object after its creation.

        This method sets up the typestore and adds the message type to ROS.

        """
        self.columns = self._parse_data_columns()

        super().__init__(self.topic_id, self.message_name)

        self.typestore = get_typestore(Stores.ROS2_IRON)
        self._add_type_to_ros()

    def _add_type_to_ros(self):
        """
        Add the message type to the ROS typestore.

        """
        self.typestore.register(
            get_types_from_msg(self.message_definition, self.message_name)
        )

    def reset(self):
        self.__post_init__()

    def _parse_data_columns(self):
        """
        Parse the columns from the message definition.

        Returns:
            list: A list of column names parsed from the message definition.

        """
        lines = self.message_definition.split("\n")
        columns = [line.split(" ")[-1] for line in lines if len(line) > 0]
        return [col for col in columns if len(col) > 0]

    def _setup_data_dict(self):
        """
        Initialize a dictionary to store parsed message data.

        Returns:
            dict: An empty dictionary to store parsed message data.

        """
        result = {}
        for col in self.columns:
            if col == "header":
                result["sec"] = []
                result["nanosec"] = []
            else:
                result[col] = []
        result["connection_timestamp"] = []
        return result

    def deserialize(self, raw_message, connection_timestamp):
        """
        Deserialize a raw message and store its data.

        Args:
            raw_message: The raw message to deserialize.
            connection_timestamp: The timestamp of the message.

        """
        data = self.typestore.deserialize_cdr(raw_message, self.topic_message_name)

        self.data["connection_timestamp"].append(connection_timestamp)

        for col in self.columns:
            if col == "header":
                ros_timestamp = data.header.stamp  # type: ignore
                self.data["sec"].append(ros_timestamp.sec)
                self.data["nanosec"].append(ros_timestamp.nanosec)
            else:
                col_data = getattr(data, col)
                self.data[col].append(col_data)
