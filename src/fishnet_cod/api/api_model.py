from typing import List, Optional, Tuple

from pydantic import BaseModel

from ..core.model import Execution, Permission, Timeseries


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
    datasetIDs: List[str]
    executionIDs: List[str]
    algorithmIDs: List[str]
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
