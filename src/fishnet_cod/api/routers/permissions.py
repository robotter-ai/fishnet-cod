import asyncio
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException
from fastapi_walletauth import JWTWalletAuthDep

from ...core.model import (
    Dataset,
    Permission,
    PermissionStatus,
    Timeseries,
)
from ..api_model import (
    ApprovePermissionsResponse,
    DenyPermissionsResponse,
    RequestDatasetPermissionsRequest,
    GrantDatasetPermissionsRequest,
)
from ..common import request_permissions, AuthorizedRouterDep

router = APIRouter(
    prefix="/permissions",
    tags=["permissions"],
    responses={404: {"description": "Not found"}},
    dependencies=[AuthorizedRouterDep],
)


@router.put("/approve")
async def approve_permissions(
    permission_hashes: List[str],
    user: JWTWalletAuthDep,
) -> ApprovePermissionsResponse:
    """
    Approve permission.
    This EndPoint will approve a list of permissions by their item hashes
    If an 'item_hashes' is provided, it will change all the Permission status
    to 'Granted'.
    """
    permissions = await Permission.fetch(permission_hashes).all()
    if not permissions:
        return ApprovePermissionsResponse(
            updatedPermissions=[],
        )

    if any([permission.authorizer != user.address for permission in permissions]):
        raise HTTPException(
            status_code=403,
            detail="Cannot approve permissions for other users",
        )

    # grant permissions
    permission_requests = []
    for permission in permissions:
        permission.status = PermissionStatus.GRANTED
        permission_requests.append(permission.save())
    permissions = await asyncio.gather(*permission_requests)

    return ApprovePermissionsResponse(
        updatedPermissions=permissions,
    )


@router.put("/deny")
async def deny_permissions(
    permission_hashes: List[str],
    user: JWTWalletAuthDep,
) -> DenyPermissionsResponse:
    """
    Deny permission.
    This EndPoint will deny a list of permissions by their item hashes
    If an `item_hashes` is provided, it will change all the Permission status
    to 'Denied'.
    """
    permissions = await Permission.fetch(permission_hashes).all()
    if not permissions:
        return DenyPermissionsResponse(
            updatedPermissions=[],
        )

    if any([permission.authorizer != user.address for permission in permissions]):
        raise HTTPException(
            status_code=403,
            detail="Cannot deny permissions for other users",
        )

    # deny permissions and get dataset ids
    dataset_ids = []
    permission_requests = []
    for permission in permissions:
        permission.status = PermissionStatus.DENIED
        dataset_ids.append(permission.datasetID)
        permission_requests.append(permission.save())
    await asyncio.gather(*permission_requests)

    return DenyPermissionsResponse(
        updatedPermissions=permissions
    )


@router.put("/datasets/{dataset_id}/request")
async def request_dataset_permissions(
    dataset_id: str,
    request: RequestDatasetPermissionsRequest,
    user: JWTWalletAuthDep
) -> List[Permission]:
    """
    Request permissions for a given dataset. If the dataset is non-mixed, a general
    permission may be created, if no timeseriesIDs are provided. If the dataset is
    mixed, permissions will be requested for all timeseries in the dataset, so that
    potentially multiple users can be asked for permission. Already existing and
    denied permissions will be returned too. Denied permissions cannot be requested
    again and must be manually granted by the owner.
    """
    # get dataset
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset does not exist")

    # check if dataset is non-mixed
    if dataset.ownsAllTimeseries:
        if dataset.owner == user.address:
            raise HTTPException(
                status_code=400,
                detail="Cannot request permissions for dataset you own",
            )
        else:
            # get general dataset permissions
            permissions = await Permission.filter(
                datasetID=dataset_id, requestor=user.address, timeseriesID=None
            ).all()
            # check if general permission exists
            for permission in permissions:
                return [permission]
            # create general permission if requested
            if not request.timeseriesIDs:
                return [
                    await Permission(
                        authorizer=dataset.owner,
                        datasetID=dataset_id,
                        timeseriesID=None,
                        requestor=user.address,
                        status=PermissionStatus.REQUESTED,
                    ).save()
                ]

    # in case of potentially mixed dataset, request permissions for all timeseries to check for
    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    timeseries_by_owner: Dict[str, List[Timeseries]] = {}
    for ts in timeseries:
        try:
            timeseries_by_owner[ts.owner].append(ts)
        except KeyError:
            timeseries_by_owner[ts.owner] = [ts]
    del timeseries_by_owner[user.address]

    # get requested timeseries permissions
    if request.timeseriesIDs:
        requested_timeseries_ids = set(request.timeseriesIDs)
    else:
        requested_timeseries_ids = set(dataset.timeseriesIDs)
    permissions = await Permission.filter(
        timeseriesID__in=requested_timeseries_ids, requestor=user.address
    ).all()

    # check if all requested timeseries have permissions
    permission_requests = []
    denied_permissions = []
    timeseries_permission_map = {p.timeseriesID: p for p in permissions}
    for timeseries_id in requested_timeseries_ids:
        # check if permission exists
        if timeseries_id not in timeseries_permission_map.keys():
            permission_requests.append(
                Permission(
                    datasetID=dataset_id,
                    timeseriesID=timeseries_id,
                    authorizer=dataset.owner,
                    requestor=user.address,
                    status=PermissionStatus.REQUESTED,
                ).save()
            )
            continue
        # check if permission is sufficient
        permission = timeseries_permission_map[timeseries_id]
        if permission.status == PermissionStatus.DENIED:
            denied_permissions.append(permission)
            continue

    # execute any permission requests
    requested_permissions = []
    if permission_requests:
        requested_permissions = await asyncio.gather(*permission_requests)
    all_permissions = permissions + requested_permissions + denied_permissions
    # remove double entries
    all_permissions = list({p.item_hash: p for p in all_permissions}.values())
    return all_permissions


@router.put("/datasets/{dataset_id}/grant")
async def grant_dataset_permissions(
    dataset_id: str,
    request: GrantDatasetPermissionsRequest,
    user: JWTWalletAuthDep
) -> List[Permission]:
    """
    Grant permissions for a given dataset. This will grant permissions for all
    timeseries in the dataset, unless `timeseriesIDs` are provided. Previously
    denied permissions can be granted too.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset does not exist")

    # check if dataset is non-mixed and general permission is requested
    if dataset.ownsAllTimeseries and not request.timeseriesIDs:
        if dataset.owner != user.address:
            raise HTTPException(
                status_code=403,
                detail="Cannot grant permissions for dataset you do not own",
            )
        # check if general permission exists
        permissions = await Permission.filter(
            datasetID=dataset_id, requestor=request.requestor, timeseriesID=None
        ).all()
        for permission in permissions:
            if permission.status in [
                PermissionStatus.REQUESTED,
                PermissionStatus.DENIED,
            ]:
                permission.status = PermissionStatus.GRANTED
                return [await permission.save()]
            return [permission]
        return [
            await Permission(
                datasetID=dataset_id,
                timeseriesID=None,
                authorizer=user.address,
                requestor=request.requestor,
                status=PermissionStatus.GRANTED,
            ).save()
        ]

    # otherwise, create permissions for all timeseries
    timeseries = await Timeseries.fetch(request.timeseriesIDs).all()
    # check if all timeseries belong to dataset and user is owner
    bad_timeseries = [
        ts.item_hash
        for ts in timeseries
        if ts.item_hash not in dataset.timeseriesIDs and ts.owner != user.address
    ]
    if bad_timeseries:
        raise HTTPException(
            status_code=400,
            detail=f"User {user.address} cannot set permission for {bad_timeseries} or they do not belong to the given dataset",
        )

    permission_requests = []
    for ts in timeseries:
        permission_requests.append(
            Permission(
                datasetID=dataset_id,
                timeseriesID=ts.item_hash,
                authorizer=user.address,
                requestor=request.requestor,
                status=PermissionStatus.GRANTED,
            ).save()
        )
    return list(await asyncio.gather(*permission_requests))
