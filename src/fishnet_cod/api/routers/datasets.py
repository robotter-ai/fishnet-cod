import asyncio
from typing import Awaitable, List, Optional, Union

from aars.utils import PageableRequest, PageableResponse
from fastapi import APIRouter, HTTPException

from ...core.model import (
    Dataset,
    DatasetPermissionStatus,
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
)
from ..common import get_timestamps_by_granularity
from .timeseries import upload_timeseries

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def get_datasets(
    ids: Optional[Union[str, List[str]]] = None,
    view_as: Optional[str] = None,
    by: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[DatasetResponse]:
    """
    Get all datasets. Returns a list of tuples of datasets and their permission status for the given `view_as` user.
    If `view_as` is not given, the permission status will be `none` for all datasets.
    If `id` is given, it will return the dataset with that id.
    If `by` is given, it will return all datasets owned by that user.
    """
    dataset_resp: Union[PageableRequest[Dataset], PageableResponse[Dataset]]
    if ids:
        if isinstance(ids, str):
            ids = [id.strip() for id in ids.split(",")]
        dataset_resp = Dataset.fetch(ids)
    elif by:
        dataset_resp = Dataset.where_eq(owner=by)
    else:
        dataset_resp = Dataset.fetch_objects()
    datasets = await dataset_resp.page(page=page, page_size=page_size)

    if view_as:
        ts_ids = []
        for rec in datasets:
            ts_ids.extend(rec.timeseriesIDs)
        ts_ids_unique = list(set(ts_ids))

        req = [
            Permission.where_eq(timeseriesID=ts_id, authorizer=view_as).all()
            for ts_id in ts_ids_unique
        ]
        resp = await asyncio.gather(*req)
        permissions = [item for sublist in resp for item in sublist]

        returned_datasets: List[DatasetResponse] = []
        for rec in datasets:
            dataset_permissions = []
            for ts_id in rec.timeseriesIDs:
                dataset_permissions.extend(
                    list(filter(lambda x: x.timeseriesID == ts_id, permissions))
                )
            if not dataset_permissions:
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(),
                        permission_status=DatasetPermissionStatus.NOT_REQUESTED,
                    )
                )
                continue

            permission_status = [perm_rec for perm_rec in dataset_permissions]
            if all(status == PermissionStatus.GRANTED for status in permission_status):
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(), permission_status=DatasetPermissionStatus.GRANTED
                    )
                )
            elif PermissionStatus.DENIED in permission_status:
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(), permission_status=DatasetPermissionStatus.DENIED
                    )
                )
            elif PermissionStatus.REQUESTED in permission_status:
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(),
                        permission_status=DatasetPermissionStatus.REQUESTED,
                    )
                )
        return returned_datasets
    else:
        return [
            DatasetResponse(**rec.dict(), permission_status=None) for rec in datasets
        ]


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
        Permission.where_eq(timeseriesID=ts_id, status=PermissionStatus.GRANTED).all()
        for ts_id in ts_ids
    ]
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
    assert dataset.id_hash
    return FungibleAssetStandard(
        name=dataset.name,
        symbol=dataset.id_hash,
        description=dataset.desc,
        # TODO: Generate chart image
        image="https://ipfs.io/ipfs/Qma2eje8yY57pNuaUyo4dsjtB9xwPz5yV6pCbK2PxpjUzo",
        animation_url=None,
        external_url=f"http://localhost:5173/data/{dataset.id_hash}/details",
        attributes=[
            Attribute(trait_type="Owner", value=dataset.owner),
            Attribute(trait_type="Last Updated", value=dataset.timestamp),
            Attribute(trait_type="Columns", value=str(len(dataset.timeseriesIDs))),
        ],
    )


@router.post("/upload/timeseries")
async def upload_dataset_timeseries(
    upload_dataset_timeseries_request: UploadDatasetTimeseriesRequest,
) -> UploadDatasetTimeseriesResponse:
    """
    Upload a dataset and timeseries at the same time.
    """
    if upload_dataset_timeseries_request.dataset.id_hash is not None:
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update a dataset. Use PUT /datasets/upload instead.",
        )
    if any([ts.id_hash is None for ts in upload_dataset_timeseries_request.timeseries]):
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update timeseries. Use PUT /timeseries/upload instead.",
        )
    timeseries = await upload_timeseries(
        req=UploadTimeseriesRequest(
            timeseries=upload_dataset_timeseries_request.timeseries
        )
    )
    dataset = await upload_dataset(dataset=upload_dataset_timeseries_request.dataset)
    return UploadDatasetTimeseriesResponse(
        dataset=dataset,
        timeseries=[ts for ts in timeseries if not isinstance(ts, BaseException)],
    )


@router.put("/")
async def upload_dataset(dataset: UploadDatasetRequest) -> Dataset:
    """
    Upload a dataset.
    If an `id_hash` is provided, it will update the dataset with that id.
    """
    if dataset.ownsAllTimeseries:
        timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
        dataset.ownsAllTimeseries = all(
            [ts.owner == dataset.owner for ts in timeseries]
        )
    if dataset.id_hash is not None:
        old_dataset = await Dataset.fetch(dataset.id_hash).first()
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
            return await old_dataset.save()
    return await Dataset(**dataset.dict()).save()


@router.put("/{dataset_id}/views")
async def generate_view(
    dataset_id: str, view_params: List[PutViewRequest]
) -> PutViewResponse:
    # get the dataset
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    view_requests = []
    for view_req in view_params:
        # get all the timeseries
        timeseries = await Timeseries.fetch(view_req.timeseriesIDs).all()
        view_values = {}
        for ts in timeseries:
            # normalize and round values
            values = [p[1] for p in ts.data]
            minimum = min(values)
            maximum = max(values)
            normalized = [
                (p[0], round((p[1] - minimum) / (maximum - minimum), 2))
                for p in ts.data
            ]
            # drop points according to granularity
            thinned = []
            i = 0  # cursor for normalized entries
            timestamps = get_timestamps_by_granularity(
                view_req.startTime, view_req.endTime, view_req.granularity
            )
            # append each point that is closest to the timestamp
            for timestamp in timestamps:
                while i < len(normalized) and normalized[i][0] < timestamp:
                    i += 1
                if i == len(normalized):
                    break
                if i == 0:
                    thinned.append(normalized[i])
                else:
                    if abs(normalized[i][0] - timestamp) < abs(
                        normalized[i - 1][0] - timestamp
                    ):
                        thinned.append(normalized[i])
                    else:
                        thinned.append(normalized[i - 1])

            view_values[str(ts.id_hash)] = thinned

        # prepare view request
        view_requests.append(
            View(
                id_hash=view_req.id_hash,
                startTime=view_req.startTime,
                endTime=view_req.endTime,
                granularity=view_req.granularity,
                values=view_values,
            ).save()
        )

    # save all records
    views = await asyncio.gather(*view_requests)
    dataset.views = [view.id_hash for view in views]
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
    for execution in await Execution.fetch(dataset_id).all():
        if execution.status == ExecutionStatus.PENDING:
            execution.status = ExecutionStatus.DENIED
            requests.append(execution.save())

    await asyncio.gather(*requests)
    return dataset
