from typing import List, Optional, Tuple

from aars import Index
from pydantic import BaseModel

from ..core.model import (
    Dataset,
    Algorithm,
    Execution,
    Permission,
    Timeseries,
    Result,
    UserInfo,
    View,
    Granularity,
)

# indexes to fetch by timeseries
Index(Timeseries, "owner")

# indexes to fetch by algorithm
Index(Algorithm, "owner")
Index(Algorithm, "name")

# indexes to fetch by dataset
Index(Dataset, "owner")
Index(Dataset, "timeseriesIDs")

# indexes to fetch by execution
Index(Execution, "datasetID")
Index(Execution, "owner")
Index(Execution, "status")

# index to fetch permissions by timeseriesID and requestor
Index(Permission, "id_hash")
Index(Permission, "status")
Index(Permission, "timeseriesID")
Index(Permission, "requestor")
Index(Permission, "authorizer")
Index(Permission, ["timeseriesID", "requestor"])
Index(Permission, ["requestor", "timeseriesID", "status"])

# index to fetch results with owner
Index(Result, "owner")

Index(UserInfo, "address")


class TimeseriesItem(BaseModel):
    id_hash: Optional[str]
    name: str
    owner: str
    desc: Optional[str]
    data: List[Tuple[int, float]]


class UploadTimeseriesRequest(BaseModel):
    timeseries: List[TimeseriesItem]


class UploadDatasetRequest(BaseModel):
    id_hash: Optional[str]
    name: str
    desc: Optional[str]
    owner: str
    ownsAllTimeseries: bool
    timeseriesIDs: List[str]


class PutUserInfo(BaseModel):
    username: str
    address: str
    bio: Optional[str]
    email: Optional[str]
    link: Optional[str]


class UploadAlgorithmRequest(BaseModel):
    id_hash: Optional[str]
    name: str
    desc: str
    owner: str
    code: str


class RequestExecutionRequest(BaseModel):
    algorithmID: str
    datasetID: str
    owner: str
    status: Optional[str]


class ExecutionStatusHistory(BaseModel):
    revision_hash: str
    status: str
    timestamp: float


class ExecutionResponse(BaseModel):
    execution: Execution
    statusHistory: List[ExecutionStatusHistory]


class RequestExecutionResponse(BaseModel):
    execution: Execution
    permissionRequests: Optional[List[Permission]]
    unavailableTimeseries: Optional[List[Timeseries]]


class PutViewRequest(BaseModel):
    id_hash: Optional[str]
    timeseriesIDs: List[str]
    granularity: Granularity
    startTime: int
    endTime: int


class PutViewResponse(BaseModel):
    dataset: Dataset
    views: List[View]
