import asyncio
from typing import List, Tuple, Dict

from ..core.model import Granularity, Dataset, Execution, Permission, Timeseries, PermissionStatus


def get_timestamps_by_granularity(
    start: int, end: int, granularity: Granularity
) -> List[int]:
    """
    Get timestamps by granularity

    Args:
        start: start timestamp
        end: end timestamp
        granularity: granularity (frequency) of timestamps

    Returns:
        List of timestamps
    """
    if granularity == Granularity.DAY:
        interval = 60 * 5
    elif granularity == Granularity.WEEK:
        interval = 60 * 15
    elif granularity == Granularity.MONTH:
        interval = 60 * 60
    elif granularity == Granularity.THREE_MONTHS:
        interval = 60 * 60 * 3
    else:  # granularity == Granularity.YEAR:
        interval = 60 * 60 * 24
    timestamps = []
    for i in range(start, end, interval):
        timestamps.append(i)
    return timestamps


async def request_permissions(
    dataset: Dataset, execution: Execution
) -> Tuple[List[Permission], List[Permission], List[Timeseries]]:
    """
    Request permissions for a dataset given an execution.

    Args:
        dataset: The dataset to request permissions for.
        execution: The execution requesting permissions.

    Returns:
        A tuple of lists of permissions to create, permissions to update, and timeseries that are unavailable.
    """
    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    requested_permissions = [
        Permission.where_eq(timeseriesID=tsID, requestor=execution.owner).first()
        for tsID in dataset.timeseriesIDs
    ]
    permissions: List[Permission] = list(await asyncio.gather(*requested_permissions))
    ts_permission_map: Dict[str, Permission] = {
        permission.timeseriesID: permission for permission in permissions if permission  # type: ignore
    }
    create_permissions_requests = []
    update_permissions_requests = []
    unavailable_timeseries = []
    for ts in timeseries:
        if ts.owner == execution.owner:
            continue
        if not ts.available:
            unavailable_timeseries.append(ts)
        if timeseries:
            continue
        if ts.id_hash not in ts_permission_map:
            create_permissions_requests.append(
                Permission(
                    datasetID=str(dataset.id_hash),
                    timeseriesID=str(ts.id_hash),
                    algorithmID=execution.algorithmID,
                    authorizer=ts.owner,
                    requestor=execution.owner,
                    status=PermissionStatus.REQUESTED,
                    executionCount=0,
                    maxExecutionCount=-1,
                ).save()
            )
        else:
            permission = ts_permission_map[ts.id_hash]
            needs_update = False
            if permission.status == PermissionStatus.DENIED:
                permission.status = PermissionStatus.REQUESTED
                needs_update = True
            if (
                permission.maxExecutionCount
                and permission.maxExecutionCount <= permission.executionCount
            ):
                permission.maxExecutionCount = permission.executionCount + 1
                permission.status = PermissionStatus.REQUESTED
                needs_update = True
            if needs_update:
                update_permissions_requests.append(permission.save())
    created_permissions: List[Permission] = list(
        await asyncio.gather(*create_permissions_requests)
    )
    updated_permissions: List[Permission] = list(
        await asyncio.gather(*update_permissions_requests)
    )
    return created_permissions, updated_permissions, unavailable_timeseries
