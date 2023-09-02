import asyncio
from typing import Dict, List, Tuple

from fastapi import Depends
from fastapi_walletauth.middleware import BearerWalletAuth, jwt_credentials_manager, JWTWalletAuthDep

import pandas as pd

from .api_model import ColumnNameType
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


async def get_harmonized_timeseries_df(
    timeseries: List[Timeseries],
    column_names: ColumnNameType = ColumnNameType.item_hash,
) -> pd.DataFrame:

    # parse all as series
    if column_names == ColumnNameType.item_hash:
        series = [pd.Series(dict(ts.data), name=ts.item_hash) for ts in timeseries]
    else:
        series = [pd.Series(dict(ts.data), name=ts.name) for ts in timeseries]
    # merge all series into one dataframe and pad missing values
    df = pd.concat(series, axis=1).fillna(method="pad")
    # set the index to the python timestamp
    df.index = pd.to_datetime(df.index, unit="s")

    return df


AuthorizedRouterDep = Depends(BearerWalletAuth(jwt_credentials_manager))
