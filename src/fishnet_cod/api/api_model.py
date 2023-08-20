from enum import Enum
from typing import List, Optional, Tuple, Union

from aars import Index
from pydantic import BaseModel

from ..core.model import (
    Algorithm,
    Dataset,
    Execution,
    Granularity,
    Permission,
    PermissionStatus,
    Result,
    Timeseries,
    UserInfo,
    View, ExecutionStatus,
)

# indexes to fetch by timeseries
Index(Timeseries, "owner")

# indexes to fetch by algorithm
Index(Algorithm, "owner")
Index(Algorithm, "name")

# indexes to fetch by dataset
Index(Dataset, "owner")
Index(Dataset, "name")

# indexes to fetch by execution
Index(Execution, "owner")
Index(Execution, "datasetID")
Index(Execution, "status")
Index(Execution, ["datasetID", "status"])

# index to fetch permissions by timeseriesID and requestor
Index(Permission, "requestor")
Index(Permission, "authorizer")
Index(Permission, ["timeseriesID", "requestor"])
Index(Permission, ["timeseriesID", "authorizer"])
Index(Permission, ["timeseriesID", "status"])
Index(Permission, ["datasetID", "status"])
Index(Permission, ["authorizer", "status"])

# index to fetch results with owner
Index(Result, "owner")

Index(UserInfo, "address")
Index(UserInfo, "username")


class TimeseriesItem(BaseModel):
    item_hash: Optional[str]
    name: str
    owner: str
    desc: Optional[str]
    data: List[Tuple[int, float]]


class UploadTimeseriesRequest(BaseModel):
    timeseries: List[TimeseriesItem]


class PostPermission(BaseModel):
    timeseriesID: str
    algorithmID: Optional[str]
    authorizer: str
    status: PermissionStatus
    executionCount: int
    maxExecutionCount: Optional[int]
    requestor: str


class UploadDatasetRequest(BaseModel):
    item_hash: Optional[str]
    name: str
    desc: Optional[str]
    owner: Optional[str]
    ownsAllTimeseries: Optional[bool]
    timeseriesIDs: List[str]


class DatasetPermissionStatus(str, Enum):
    NOT_REQUESTED = "NOT REQUESTED"
    REQUESTED = "REQUESTED"
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class DatasetResponse(Dataset):
    permission_status: Optional[DatasetPermissionStatus]


class RequestDatasetPermissionsRequest(BaseModel):
    requestor: str
    algorithmID: Optional[str]
    timeseriesIDs: Optional[List[str]]
    requestedExecutionCount: Optional[int]


class GrantDatasetPermissionsRequest(BaseModel):
    authorizer: str
    requestor: str
    algorithmID: Optional[str]
    timeseriesIDs: Optional[List[str]]
    maxExecutionCount: Optional[int]


class UploadPermissionRecords:
    timeseriesID: str
    algorithmID: Optional[str]
    authorizer: str
    status: PermissionStatus
    executionCount: int
    maxExecutionCount: Optional[int]
    requestor: str


class UploadDatasetTimeseriesRequest(BaseModel):
    dataset: UploadDatasetRequest
    timeseries: List[TimeseriesItem]


class UploadDatasetTimeseriesResponse(BaseModel):
    dataset: Dataset
    timeseries: List[Timeseries]


class PutUserInfo(BaseModel):
    username: str
    address: str
    bio: Optional[str]
    email: Optional[str]
    link: Optional[str]


class UploadAlgorithmRequest(BaseModel):
    item_hash: Optional[str]
    name: str
    desc: str
    owner: str
    code: str


class RequestExecutionRequest(BaseModel):
    algorithmID: str
    datasetID: str
    owner: str
    status: Optional[str] = ExecutionStatus.REQUESTED


class ExecutionStatusHistory(BaseModel):
    revision_hash: str
    status: str
    timestamp: float


class NotificationType(str, Enum):
    PermissionRequest = "PermissionRequest"
    ExecutionTriggered = "ExecutionTriggered"


class Notification(BaseModel):
    type: NotificationType
    message_text: str


class PermissionRequestNotification(Notification):
    type: NotificationType = NotificationType.PermissionRequest
    requestor: str
    datasetID: str
    uses: Optional[int]
    algorithmIDs: Optional[List[str]]


class ExecutionTriggeredNotification(Notification):
    type: NotificationType = NotificationType.ExecutionTriggered
    executionID: str
    datasetID: str
    algorithmID: str
    status: ExecutionStatus


class ExecutionResponse(BaseModel):
    execution: Execution
    statusHistory: List[ExecutionStatusHistory]


class RequestExecutionResponse(BaseModel):
    execution: Execution
    permissionRequests: Optional[List[Permission]]
    unavailableTimeseries: Optional[List[Timeseries]]


class ApprovePermissionsResponse(BaseModel):
    updatedPermissions: List[Permission]
    triggeredExecutions: List[Execution]


class DenyPermissionsResponse(BaseModel):
    updatedPermissions: List[Permission]
    deniedExecutions: List[Execution]


class PutViewRequest(BaseModel):
    item_hash: Optional[str]
    timeseriesIDs: List[str]
    granularity: Granularity = Granularity.YEAR
    startTime: Optional[int] = None
    endTime: Optional[int] = None


class PutViewResponse(BaseModel):
    dataset: Dataset
    views: List[View]


class Attribute(BaseModel):
    trait_type: str
    value: Optional[Union[str, int, float]]


class FungibleAssetStandard(BaseModel):
    name: str
    symbol: str
    description: Optional[str]
    image: Optional[str]
    animation_url: Optional[str]
    external_url: Optional[str]
    attributes: List[Attribute]


class MessageResponse(BaseModel):
    response: str


class ColumnNameType(Enum):
    item_hash = "item_hash"
    name = "name"
