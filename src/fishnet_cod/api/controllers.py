import asyncio
import io
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from fastapi import HTTPException, UploadFile
from fastapi_walletauth.middleware import JWTWalletAuthDep

from ..core.model import Dataset, Permission, PermissionStatus, Timeseries, View
from .api_model import (
    DatasetPermissionStatus,
    DatasetResponse,
    PutViewRequest,
    PutViewResponse,
)
from .utils import granularity_to_interval


async def request_permissions(
    dataset: Dataset,
    user: JWTWalletAuthDep,
) -> Tuple[List[Permission], List[Permission], List[Timeseries]]:
    """
    Request permissions for a dataset given an execution.

    Args:
        dataset: The dataset to request permissions for.

    Returns:
        A tuple of lists of permissions to create, permissions to update, and timeseries that are unavailable.
    """
    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    permissions = await Permission.filter(requestor=user.address).all()
    permissions = [
        permission
        for permission in permissions
        if permission.timeseriesID in dataset.timeseriesIDs
    ]

    ts_permission_map: Dict[str, Permission] = {
        permission.timeseriesID: permission for permission in permissions if permission  # type: ignore
    }
    create_permissions_requests = []
    update_permissions_requests = []
    unavailable_timeseries = []
    for ts in timeseries:
        if ts.owner == user.address:
            continue
        if not ts.available:
            unavailable_timeseries.append(ts)
        if ts.item_hash not in ts_permission_map:
            create_permissions_requests.append(
                Permission(
                    datasetID=str(dataset.item_hash),
                    timeseriesID=str(ts.item_hash),
                    authorizer=ts.owner,
                    requestor=user.address,
                    status=PermissionStatus.REQUESTED,
                ).save()
            )
        else:
            permission = ts_permission_map[ts.item_hash]
            if permission.status == PermissionStatus.GRANTED:
                continue
            if permission.status == PermissionStatus.DENIED:
                permission.status = PermissionStatus.REQUESTED
                update_permissions_requests.append(permission.save())
    created_permissions: List[Permission] = list(
        await asyncio.gather(*create_permissions_requests)
    )
    updated_permissions: List[Permission] = list(
        await asyncio.gather(*update_permissions_requests)
    )
    return created_permissions, updated_permissions, unavailable_timeseries


def get_dataset_df(dataset_id: str) -> pd.DataFrame:
    file_path = f"./files/{dataset_id}.parquet"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail="Dataset slice not found on this node"
        )
    df = pd.read_parquet(file_path)
    return df


def load_data_df(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please provide a filename")
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith(".parquet"):
        df = pd.read_parquet(file.file)
    elif file.filename.endswith(".feather"):
        df = pd.read_feather(file.file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format (only CSV, parquet and feather are supported)",
        )

    # find the first column with a timestamp or ISO8601 date and use it as the index
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=col)
            continue
        if "date" in col.lower() or "time" in col.lower():
            df.index = pd.to_datetime(df[col])
            df = df.drop(columns=col)
    return df


"""
async def update_timeseries(
    df: pd.DataFrame,
    metadata: Union[List[CreateTimeseriesRequest], List[UpdateTimeseriesRequest]],
    timeseries: Optional[List[Timeseries]],
    user: JWTWalletAuthDep,
) -> List[Timeseries]:
    timeseries_by_name: Dict[str, Timeseries] = (
        {ts.name: ts for ts in timeseries} if timeseries else {}
    )
    metadata_by_item_hash: Dict[str, UpdateTimeseriesRequest] = (
        {
            ts.item_hash: ts
            for ts in metadata
            if isinstance(ts, UpdateTimeseriesRequest) and ts.item_hash in metadata
        }
        if metadata
        else {}
    )
    metadata_by_name: Dict[
        str, Union[CreateTimeseriesRequest, UpdateTimeseriesRequest]
    ] = ({ts.name: ts for ts in metadata if ts.name in metadata} if metadata else {})
    timeseries_requests = []
    for col in df.columns:
        try:
            ts = timeseries_by_name.get(col)
            if not ts:
                md = metadata_by_name.get(col)
                ts = Timeseries(
                    item_hash=None,
                    name=col,
                    desc=md.desc if md else None,
                    owner=user.address,
                    min=df[col].min(),
                    max=df[col].max(),
                    avg=df[col].mean(),
                    std=df[col].std(),
                    median=df[col].median(),
                )
            else:
                md = metadata_by_item_hash.get(str(ts.item_hash))
                if md:
                    ts.name = md.name if md.name else ts.name
                    ts.desc = md.desc if md.desc else ts.desc
                ts.min = df[col].min()
                ts.max = df[col].max()
                ts.avg = df[col].mean()
                ts.std = df[col].std()
                ts.median = df[col].median()
            assert ts is not None
            ts.name = col
            timeseries_requests.append(ts.save())
        except (ValidationError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {e}",
            )
    return await asyncio.gather(*timeseries_requests)
"""


class DataFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"


async def get_data_stream(df: pd.DataFrame, data_format: DataFormat = DataFormat.CSV):
    stream: Union[io.StringIO, io.BytesIO]
    if data_format == DataFormat.CSV:
        stream = io.StringIO()
        df.to_csv(stream)
        media_type = "text/csv"
    elif data_format == DataFormat.FEATHER:
        stream = io.BytesIO()
        df.to_feather(stream)
        media_type = "application/octet-stream"
    elif data_format == DataFormat.PARQUET:
        stream = io.BytesIO()
        df.to_parquet(stream)
        media_type = "application/octet-stream"
    else:
        raise HTTPException(status_code=400, detail="Unsupported data format")
    stream.seek(0)
    return media_type, stream


async def merge_data(df: pd.DataFrame, file_path: Path, update_values: bool = False):
    existing_df = (
        pd.read_parquet(file_path) if os.path.exists(file_path) else pd.DataFrame()
    )
    df = pd.concat([existing_df, df])

    df = df[~df.index.duplicated(keep="last" if update_values else "first")]
    df = df.sort_index()
    return df


async def get_views_by_dataset(dataset_id):
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await View.fetch(dataset.viewIDs).all()


async def generate_views(dataset_id, view_params):
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    view_ids = [view.item_hash for view in view_params if view.item_hash is not None]
    views_map = {view.item_hash: view for view in await View.fetch(view_ids).all()}

    full_df = get_dataset_df(dataset_id)
    view_requests = []
    for view_req in view_params:
        df = full_df[view_req.columns].copy()
        end_time, start_time, view_values = await calculate_view(df, view_req)

        if views_map.get(view_req.item_hash):
            old_view = views_map[view_req.item_hash]
            old_view.startTime = start_time
            old_view.endTime = end_time
            old_view.granularity = view_req.granularity
            old_view.values = view_values
            view_requests.append(old_view.save())
        else:
            view_requests.append(
                View(
                    startTime=start_time,
                    endTime=end_time,
                    granularity=view_req.granularity,
                    columns=df.columns,
                    values=view_values,
                ).save()
            )

    views = await asyncio.gather(*view_requests)
    dataset.viewIDs = [view.item_hash for view in views]
    await dataset.save()

    return PutViewResponse(dataset=dataset, views=views)


async def update_views(dataset_id):
    views = await get_views_by_dataset(dataset_id)
    views_params = [
        PutViewRequest(
            item_hash=view.item_hash,
            startTime=view.startTime,
            endTime=view.endTime,
            granularity=view.granularity,
            columns=view.columns,
        )
        for view in views
    ]
    await generate_views(dataset_id, views_params)


async def calculate_view(df: pd.DataFrame, view_req: PutViewRequest):
    # filter by time window
    if view_req.startTime is not None:
        start_date = pd.to_datetime(view_req.startTime, unit="s")
        df = df[df.index >= start_date]
    if view_req.endTime is not None:
        end_date = pd.to_datetime(view_req.endTime, unit="s")
        df = df[df.index <= end_date]
    # normalize and round values
    df = (df - df.min()) / (df.max() - df.min())
    # drop points according to granularity
    df = df.resample(granularity_to_interval(view_req.granularity)).mean().dropna()
    df.round(2)

    view_values = {
        str(col): [
            (int(index.timestamp()), float(value)) for index, value in df[col].items()
        ]
        for col in df.columns
    }

    start_time = (
        int(df.index.min().timestamp())
        if view_req.startTime is None
        else view_req.startTime
    )
    end_time = (
        int(df.index.max().timestamp())
        if view_req.endTime is None
        else view_req.endTime
    )
    return end_time, start_time, view_values


async def view_datasets_as(datasets: List[Dataset], view_as: str):
    ts_ids = [ts_id for dataset in datasets for ts_id in dataset.timeseriesIDs]
    ts_ids_unique = list(set(ts_ids))
    dataset_ids = [dataset.item_hash for dataset in datasets]
    resp = await asyncio.gather(
        Permission.filter(timeseriesID__in=ts_ids_unique, requestor=view_as).all(),
        Permission.filter(
            datasetID__in=dataset_ids, requestor=view_as, timeseriesID=None
        ).all(),
    )
    permissions = [item for sublist in resp for item in sublist]
    returned_datasets: List[DatasetResponse] = []
    for dataset in datasets:
        returned_datasets.append(
            DatasetResponse(
                **dataset.dict(),
                permission_status=get_dataset_permission_status(dataset, permissions),
            )
        )
    return returned_datasets


def get_dataset_permission_status(
    dataset: Dataset, permissions: List[Permission]
) -> DatasetPermissionStatus:
    """
    Get the permission status for a given dataset and a list of timeseries ids and their permissions.
    """
    permissions = [
        p
        for p in permissions
        if p.datasetID == dataset.item_hash or p.timeseriesID in dataset.timeseriesIDs
    ]

    if not permissions:
        return DatasetPermissionStatus.NOT_REQUESTED

    for permission in permissions:
        if permission.timeseriesID is None:
            if permission.status == PermissionStatus.GRANTED:
                return DatasetPermissionStatus.GRANTED
            elif permission.status == PermissionStatus.DENIED:
                return DatasetPermissionStatus.DENIED
            elif permission.status == PermissionStatus.REQUESTED:
                return DatasetPermissionStatus.REQUESTED

    permissions_status = [p.status for p in permissions]

    if all(status == PermissionStatus.GRANTED for status in permissions_status):
        return DatasetPermissionStatus.GRANTED
    elif PermissionStatus.DENIED in permissions_status:
        return DatasetPermissionStatus.DENIED
    elif PermissionStatus.REQUESTED in permissions_status:
        return DatasetPermissionStatus.REQUESTED

    raise Exception("Should not reach here")
