from enum import Enum
from typing import List, Optional, Tuple, Union

from aars import Index
from pydantic import BaseModel

from ..core.model import (
    Dataset,
    Granularity,
    Permission,
    PermissionStatus,
    Timeseries,
    UserInfo,
    View,
)

# indexes to fetch by timeseries
Index(Timeseries, "owner")

# indexes to fetch by dataset
Index(Dataset, "owner")
Index(Dataset, "name")

# index to fetch permissions by timeseriesID and requestor
Index(Permission, "requestor")
Index(Permission, "authorizer")
Index(Permission, ["timeseriesID", "requestor"])
Index(Permission, ["timeseriesID", "authorizer"])
Index(Permission, ["timeseriesID", "status"])
Index(Permission, ["datasetID", "status"])
Index(Permission, ["authorizer", "status"])

Index(UserInfo, "address")
Index(UserInfo, "username")


class TimeseriesItem(BaseModel):
    item_hash: Optional[str]
    name: str
    owner: Optional[str]  # TODO: to remove
    desc: Optional[str]
    data: List[Tuple[int, float]]


class UploadTimeseriesRequest(BaseModel):
    timeseries: List[TimeseriesItem]


class PostPermission(BaseModel):
    timeseriesID: str
    algorithmID: Optional[str]
    authorizer: str  # TODO: to remove
    status: PermissionStatus
    executionCount: int
    maxExecutionCount: Optional[int]
    requestor: str


class UploadDatasetRequest(BaseModel):
    item_hash: Optional[str]
    name: str
    desc: Optional[str]
    owner: Optional[str]  # TODO: to remove
    ownsAllTimeseries: Optional[bool]  # TODO: to remove
    timeseriesIDs: List[str]
    price: Optional[str] = None


class DatasetPermissionStatus(str, Enum):
    NOT_REQUESTED = "NOT REQUESTED"
    REQUESTED = "REQUESTED"
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class DatasetResponse(Dataset):
    permission_status: Optional[DatasetPermissionStatus]


class RequestDatasetPermissionsRequest(BaseModel):
    timeseriesIDs: Optional[List[str]]
    requestedExecutionCount: Optional[int]


class GrantDatasetPermissionsRequest(BaseModel):
    requestor: str
    timeseriesIDs: Optional[List[str]]


class UploadDatasetTimeseriesRequest(BaseModel):
    dataset: UploadDatasetRequest
    timeseries: List[TimeseriesItem]


class UploadDatasetTimeseriesResponse(BaseModel):
    dataset: Dataset
    timeseries: List[Timeseries]


class PutUserInfo(BaseModel):
    username: str
    bio: Optional[str]
    email: Optional[str]
    link: Optional[str]


class NotificationType(str, Enum):
    PermissionRequest = "PermissionRequest"


class Notification(BaseModel):
    type: NotificationType
    message_text: str


class PermissionRequestNotification(Notification):
    type: NotificationType = NotificationType.PermissionRequest
    requestor: str
    datasetID: str
    uses: Optional[int]


class ApprovePermissionsResponse(BaseModel):
    updatedPermissions: List[Permission]


class DenyPermissionsResponse(BaseModel):
    updatedPermissions: List[Permission]


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


class ColumnNameType(Enum):
    item_hash = "item_hash"
    name = "name"


class DataFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"
