from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from aars import Record


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
    data: List[Tuple[int, float]]


# Check coinmarketcap.com for the exact granularity/aggregation timeframes
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
    values: Dict[str, List[Tuple[int, float]]]  # timeseriesID -> data


class Dataset(Record):
    name: str
    owner: str
    ownsAllTimeseries: bool
    available: bool = True
    timeseriesIDs: List[str]
    desc: Optional[str]
    viewIDs: Optional[List[str]]


class Algorithm(Record):
    name: str
    desc: str
    owner: str
    code: str


class ExecutionStatus(str, Enum):
    REQUESTED = "REQUESTED"
    PENDING = "PENDING"
    DENIED = "DENIED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class Execution(Record):
    algorithmID: str
    datasetID: str
    owner: str
    status: ExecutionStatus = ExecutionStatus.REQUESTED
    params: Optional[dict]


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
    algorithmID: Optional[str]
    status: PermissionStatus
    executionCount: int
    maxExecutionCount: Optional[int]


class Result(Record):
    executionID: str
    owner: str
    executor_vm: str
    data: Any
