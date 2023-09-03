import asyncio
import pandas as pd
from typing import Awaitable, List, Optional, Union, Dict, Annotated

from aars.utils import PageableRequest, PageableResponse
from fastapi import APIRouter, HTTPException, Depends
from fastapi_walletauth import WalletAuth

from ...core.model import (
    Dataset,
    Execution,
    ExecutionStatus,
    Permission,
    PermissionStatus,
    Timeseries,
    View,
)
from ..api_model import (
    Attribute,
    DatasetResponse,
    FungibleAssetStandard,
    PutViewRequest,
    PutViewResponse,
    UploadDatasetRequest,
    UploadDatasetTimeseriesRequest,
    UploadDatasetTimeseriesResponse,
    UploadTimeseriesRequest,
    DatasetPermissionStatus,
)
from ..common import granularity_to_interval, OptionalWalletAuthDep, get_harmonized_timeseries_df, OptionalWalletAuth
from .timeseries import upload_timeseries

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
    dependencies=[OptionalWalletAuthDep],
)


@router.get("")
async def get_datasets(
    view_as: Optional[str] = None,
    by: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[DatasetResponse]:
    """
    Get all datasets. Returns a list of tuples of datasets and their permission status for the given `view_as` user.
    If `view_as` is not given, the permission status will be `none` for all datasets.
    If `by` is given, it will return all datasets owned by that user.
    """
    dataset_resp: Union[PageableRequest[Dataset], PageableResponse[Dataset]]
    if by:
        dataset_resp = Dataset.filter(owner=by)
    else:
        dataset_resp = Dataset.fetch_objects()
    datasets = await dataset_resp.page(page=page, page_size=page_size)

    if view_as:
        return await view_datasets_as(datasets, view_as)
    else:
        return [
            DatasetResponse(**dataset.dict(), permission_status=None)
            for dataset in datasets
        ]


@router.post("/getByIDs")
async def get_datasets_by_ids(
    dataset_ids: List[str], view_as: Optional[str] = None
) -> List[DatasetResponse]:
    """
    Get all datasets by their ids. Returns a list of tuples of datasets and their permission status for the given `view_as` user.
    If `view_as` is not given, the permission status will be `none` for all datasets.
    """
    datasets = await Dataset.fetch(dataset_ids).all()
    if view_as:
        return await view_datasets_as(datasets, view_as)
    else:
        return [
            DatasetResponse(**dataset.dict(), permission_status=None)
            for dataset in datasets
        ]


async def view_datasets_as(datasets, view_as):
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
                permission_status=get_dataset_permission_status(
                    dataset, permissions
                ),
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


@router.put("")
async def upload_dataset(
    dataset: UploadDatasetRequest,
    user: Annotated[Optional[WalletAuth], Depends(OptionalWalletAuth)]
) -> Dataset:
    """
    Upload a dataset.
    If an `item_hash` is provided, it will update the dataset with that id.
    """
    if user.address:
        if dataset.owner != user.address:
            raise HTTPException(
                status_code=403, detail="Cannot upload dataset that is not owned by you"
            )
        if dataset.owner is None:
            dataset.owner = user.address
    else:
        if dataset.owner is None:
            raise HTTPException(
                status_code=403, detail="Cannot upload dataset without owner"
            )
    if dataset.ownsAllTimeseries or dataset.ownsAllTimeseries is None:
        timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
        dataset.ownsAllTimeseries = all(
            [ts.owner == dataset.owner for ts in timeseries]
        )
    if dataset.item_hash is not None:
        old_dataset = await Dataset.fetch(dataset.item_hash).first()
        if old_dataset is not None:
            if old_dataset.owner != dataset.owner:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot overwrite dataset that is not owned by you",
                )
            old_dataset.name = dataset.name
            old_dataset.desc = dataset.desc
            old_dataset.timeseriesIDs = dataset.timeseriesIDs
            old_dataset.ownsAllTimeseries = dataset.ownsAllTimeseries
            old_dataset.price = dataset.price
            if old_dataset.changed:
                await old_dataset.save()
            return old_dataset
    return await Dataset(**dataset.dict()).save()


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str, view_as: Optional[str] = None
) -> DatasetResponse:
    """
    Get a dataset by its id.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="No Dataset found")

    if view_as:
        ts_ids = [ts_id for ts_id in dataset.timeseriesIDs]
        resp = await asyncio.gather(
            Permission.filter(timeseriesID__in=ts_ids, requestor=view_as).all(),
            Permission.filter(datasetID=dataset_id, requestor=view_as).all(),
        )
        permissions = [item for sublist in resp for item in sublist]
        return DatasetResponse(
            **dataset.dict(),
            permission_status=get_dataset_permission_status(dataset, permissions),
        )

    return DatasetResponse(**dataset.dict(), permission_status=None)


@router.get("/{dataset_id}/permissions")
async def get_dataset_permissions(dataset_id: str) -> List[Permission]:
    """
    Get all granted permissions for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="No Dataset found")
    ts_ids = [ts_id for ts_id in dataset.timeseriesIDs]
    matched_permission_records = [
        Permission.filter(timeseriesID=ts_id, status=PermissionStatus.GRANTED).all()
        for ts_id in ts_ids
    ] + [Permission.filter(datasetID=dataset_id, status=PermissionStatus.GRANTED).all()]
    records = await asyncio.gather(*matched_permission_records)
    permission_records = [element for row in records for element in row if element]

    return permission_records


@router.get("/{dataset_id}/metaplex")
async def get_dataset_metaplex_dataset(dataset_id: str) -> FungibleAssetStandard:
    """
    Get the metaplex metadata for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert dataset.item_hash
    return FungibleAssetStandard(
        name=dataset.name,
        symbol=dataset.item_hash,
        description=dataset.desc,
        # TODO: Generate chart image
        image="https://ipfs.io/ipfs/Qma2eje8yY57pNuaUyo4dsjtB9xwPz5yV6pCbK2PxpjUzo",
        animation_url=None,
        external_url=f"http://localhost:5173/data/{dataset.item_hash}/details",
        attributes=[
            Attribute(trait_type="Owner", value=dataset.owner),
            Attribute(trait_type="Last Updated", value=dataset.timestamp),
            Attribute(trait_type="Columns", value=str(len(dataset.timeseriesIDs))),
        ],
    )


@router.post("/upload/timeseries")
async def upload_dataset_with_timeseries(
    upload_dataset_timeseries_request: UploadDatasetTimeseriesRequest,
    user: Annotated[Optional[WalletAuth], Depends(OptionalWalletAuth)]
) -> UploadDatasetTimeseriesResponse:
    """
    Upload a dataset and timeseries at the same time.
    """
    if upload_dataset_timeseries_request.dataset.item_hash is not None:
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update a dataset. Use PUT /datasets instead.",
        )
    if any(
        [
            ts.item_hash is not None
            for ts in upload_dataset_timeseries_request.timeseries
        ]
    ):
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update timeseries. Use PUT /timeseries instead.",
        )
    timeseries = await upload_timeseries(
        req=UploadTimeseriesRequest(
            timeseries=upload_dataset_timeseries_request.timeseries
        ),
        user=user
    )
    upload_dataset_timeseries_request.dataset.timeseriesIDs = [
        ts.item_hash for ts in timeseries
    ]
    dataset = await upload_dataset(dataset=upload_dataset_timeseries_request.dataset, user=user)
    return UploadDatasetTimeseriesResponse(
        dataset=dataset,
        timeseries=[ts for ts in timeseries if not isinstance(ts, BaseException)],
    )


@router.get("/{dataset_id}/timeseries")
async def get_dataset_timeseries(dataset_id: str) -> List[Timeseries]:
    """
    Get all timeseries for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await Timeseries.fetch(dataset.timeseriesIDs).all()


@router.get("/{dataset_id}/views")
async def get_views(dataset_id: str) -> List[View]:
    """
    Get all views for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await View.fetch(dataset.viewIDs).all()


@router.put("/{dataset_id}/views")
async def generate_view(
    dataset_id: str, view_params: List[PutViewRequest]
) -> PutViewResponse:
    # get the dataset
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # get the views
    view_ids = [view.item_hash for view in view_params if view.item_hash is not None]
    views_map = {view.item_hash: view for view in await View.fetch(view_ids).all()}
    view_requests = []
    for view_req in view_params:
        # get all the timeseries
        timeseries = await Timeseries.fetch(view_req.timeseriesIDs).all()
        timeseries_df = await get_harmonized_timeseries_df(timeseries)
        # filter by time window
        if view_req.startTime is not None:
            start_date = pd.to_datetime(view_req.startTime, unit="s")
            timeseries_df = timeseries_df[timeseries_df.index >= start_date]
        if view_req.endTime is not None:
            end_date = pd.to_datetime(view_req.endTime, unit="s")
            timeseries_df = timeseries_df[timeseries_df.index <= end_date]
        # normalize and round values
        timeseries_df = (timeseries_df - timeseries_df.min()) / (
            timeseries_df.max() - timeseries_df.min()
        )
        # drop points according to granularity
        timeseries_df = (
            timeseries_df.resample(granularity_to_interval(view_req.granularity))
            .mean()
            .dropna()
        )
        timeseries_df.round(2)
        # convert to dict of timeseries values
        view_values = {
            ts.item_hash: [
                [index.timestamp(), value]
                for index, value in timeseries_df[ts.item_hash].items()
            ]
            for ts in timeseries
        }
        column_names = [ts.name for ts in timeseries]
        # prepare view request
        start_time = int(timeseries_df.index.min().timestamp()) if view_req.startTime is None else view_req.startTime
        end_time = int(timeseries_df.index.max().timestamp()) if view_req.endTime is None else view_req.endTime
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
                    item_hash=view_req.item_hash,
                    startTime=start_time,
                    endTime=end_time,
                    granularity=view_req.granularity,
                    columns=column_names,
                    values=view_values,
                ).save()
            )

    # save all records
    views = await asyncio.gather(*view_requests)
    dataset.viewIDs = [view.item_hash for view in views]
    await dataset.save()

    return PutViewResponse(dataset=dataset, views=views)


@router.put("/{dataset_id}/available/{available}")
async def set_dataset_available(dataset_id: str, available: bool) -> Dataset:
    """
    Set a dataset to be available or not. This will also update the status of all
    executions that are waiting for permission on this dataset.
    param `dataset_id':put the dataset hash here
    param 'available':put the Boolean value
    """

    requests: List[Awaitable[Union[Dataset, Timeseries, Execution]]] = []
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="No Dataset found")
    dataset.available = available
    requests.append(dataset.save())

    ts_list = await Timeseries.fetch(dataset.timeseriesIDs).all()
    if not ts_list:
        raise HTTPException(status_code=424, detail="No Timeseries found")

    for timeseries in ts_list:
        if timeseries.available != available:
            timeseries.available = available
            requests.append(timeseries.save())

    # TODO: background task
    for execution in await Execution.filter(datasetID=dataset_id).all():
        if execution.status == ExecutionStatus.PENDING:
            execution.status = ExecutionStatus.DENIED
            requests.append(execution.save())

    await asyncio.gather(*requests)
    return dataset
