import asyncio
import os
from typing import Dict, List, Tuple, Union, Optional

from fastapi import Depends, HTTPException
from fastapi_walletauth import JWTWalletAuthDep
from fastapi_walletauth.middleware import BearerWalletAuth, jwt_credentials_manager, JWTWalletAuthDep

import pandas as pd
from pydantic import ValidationError

from .api_model import ColumnNameType, CreateTimeseriesRequest, UpdateTimeseriesRequest
from ..core.model import (
    Dataset,
    Granularity,
    Permission,
    PermissionStatus,
    Timeseries,
)


def granularity_to_interval(granularity: Granularity) -> str:
    """
    Get pandas-compatible interval from Granularity

    Args:
        start: start timestamp
        end: end timestamp
        granularity: granularity (frequency) of timestamps

    Returns:
        List of timestamps
    """
    if granularity == Granularity.DAY:
        return "5min"
    elif granularity == Granularity.WEEK:
        return "15min"
    elif granularity == Granularity.MONTH:
        return "H"
    elif granularity == Granularity.THREE_MONTHS:
        return "3H"
    else:  # granularity == Granularity.YEAR:
        return "D"


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
    permissions = await Permission.filter(
        requestor=user.address
    ).all()
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


AuthorizedRouterDep = Depends(BearerWalletAuth(jwt_credentials_manager))


def get_dataset_df(dataset_id: str) -> pd.DataFrame:
    file_path = f"./files/{dataset_id}.parquet"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset slice not found on this node")
    df = pd.read_parquet(file_path)
    return df


def load_data_file(file):
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith(".parquet"):
        df = pd.read_parquet(file.file)
    elif file.filename.endswith(".feather"):
        df = pd.read_feather(file.file)
    else:
        raise HTTPException(status_code=400,
                            detail="Unsupported file format (only CSV, parquet and feather are supported)")

    # find the first column with a timestamp or ISO8601 date and use it as the index
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=col)
            continue
        if "date" in col.lower() or "time" in col.lower():
            df.index = pd.to_datetime(df[col])
            df = df.drop(columns=col)
    return df


async def update_timeseries(
    df: pd.DataFrame,
    metadata: Union[List[CreateTimeseriesRequest], List[UpdateTimeseriesRequest]],
    timeseries: Optional[List[Timeseries]],
    user: JWTWalletAuthDep,
) -> List[Timeseries]:

    timeseries_by_name: Dict[str, Timeseries] = {ts.name: ts for ts in timeseries} if timeseries else {}
    metadata_by_item_hash: Dict[str, UpdateTimeseriesRequest] = {
        ts.item_hash: ts for ts in metadata
        if isinstance(ts, UpdateTimeseriesRequest) and ts.item_hash in metadata
    } if metadata else {}
    metadata_by_name: Dict[str, Union[CreateTimeseriesRequest, UpdateTimeseriesRequest]] = {
        ts.name: ts for ts in metadata
        if ts.name in metadata
    } if metadata else {}
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
