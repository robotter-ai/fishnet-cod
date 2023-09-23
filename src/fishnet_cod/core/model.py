from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from aars import Record
from decimal import Decimal

from pydantic import BaseModel


class UserInfo(Record):
    username: str
    address: str
    bio: Optional[str]
    email: Optional[str]
    link: Optional[str]


class Timeseries(Record):
    name: str
    owner: str
    desc: Optional[str]
    available: bool = True
    min: Optional[float]
    max: Optional[float]
    avg: Optional[float]
    std: Optional[float]
    median: Optional[float]


class Granularity(str, Enum):
    DAY = "DAY"  # 1 value every five minutes
    WEEK = "WEEK"  # 1 value every 15 minutes
    MONTH = "MONTH"  # 1 value every hour
    THREE_MONTHS = "THREE_MONTHS"  # 1 value every 3 hours
    YEAR = "YEAR"  # 1 value every day


class View(Record):
    startTime: int
    endTime: int
    granularity: Granularity
    columns: List[str]
    values: Dict[str, List[Tuple[int, float]]]  # timeseriesID -> data


class Dataset(Record):
    name: str
    owner: str
    ownsAllTimeseries: bool
    available: bool = True
    timeseriesIDs: List[str]
    desc: Optional[str]
    viewIDs: Optional[List[str]]
    price: Optional[str]


class PermissionStatus(str, Enum):
    REQUESTED = "REQUESTED"
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class Permission(Record):
    """
    A permission request for a dataset.
    """

    authorizer: str
    requestor: str
    datasetID: str
    """
    The datasetID that the permission was requested for.
    Can be a different dataset than the one the timeseries belongs to.
    If the dataset is a composite dataset, this is the composite dataset and timeseriesID CANNOT be None.
    This is because multiple permissions of multiple users are potentially required for a composite dataset.
    """
    timeseriesID: Optional[str]
    """
    The timeseriesID that the permission was requested for.
    """
    status: PermissionStatus


class TimeseriesSliceStats(BaseModel):
    min: float
    max: float
    avg: float
    std: float
    median: float


class DatasetSlice(Record):
    """
    Metadata and descriptive statistics about a time slice of a dataset.
    The actual data is stored in a file on the location specified.
    """
    datasetID: str
    """
    The datasetID that the slice belongs to.
    """
    locationUrl: str
    """
    The URL where the slice's timeseries data is stored.
    """
    timeseriesStats: Dict[str, TimeseriesSliceStats]
    """
    The stats for each timeseries in the slice.
    """
    startTime: int
    """
    The start time of the slice.
    """
    endTime: int
    """
    The end time of the slice.
    """


class DataNodeConfig(BaseModel):
    url: str
    """
    The URL of the data node.
    """
    startTime: int
    """
    The earliest timestamp that the data node has data for.
    """
    endTime: int
    """
    The latest timestamp that the data node has data for.
    """


class FishnetConfig(BaseModel):
    nodes: Dict[str, DataNodeConfig]
    """
    The data nodes that the Fishnet API can use to store data. The keys are the item hashes of the PROGRAM
    messages that created the data nodes.
    """
