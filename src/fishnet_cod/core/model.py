from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

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
    desc: Optional[str]
    available: bool = True
    ownsAllTimeseries: bool
    timeseriesIDs: List[str]
    views: Optional[List[str]]


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
    resultID: Optional[str]
    params: Optional[dict]


class PermissionStatus(str, Enum):
    REQUESTED = "REQUESTED"
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class DatasetPermissionStatus(str, Enum):
    NOT_REQUESTED = "NOT REQUESTED"
    REQUESTED = "REQUESTED"
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class Permission(Record):
    timeseriesID: str
    algorithmID: Optional[str]
    authorizer: Optional[str]
    status: PermissionStatus
    executionCount: int
    maxExecutionCount: Optional[int]
    requestor: str


class Result(Record):
    executionID: str
    data: Any
    owner: str
    executor_vm: str
