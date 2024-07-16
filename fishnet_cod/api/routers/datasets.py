import asyncio
from typing import Annotated, Awaitable, List, Optional, Union

from aars.utils import PageableRequest, PageableResponse
from fastapi import APIRouter, Depends, HTTPException
from fastapi_walletauth import JWTWalletAuthDep, JWTWalletCredentials
from fastapi_walletauth.middleware import jwt_credentials_manager
from starlette.responses import StreamingResponse

from ...core.model import Dataset, Permission, PermissionStatus, Timeseries, View
from ..api_model import (
    Attribute,
    ColumnNameType,
    DatasetResponse,
    FungibleAssetStandard,
    PutViewRequest,
    PutViewResponse,
    TimeseriesWithData,
    UploadDatasetRequest,
    UploadDatasetTimeseriesRequest,
    UploadDatasetTimeseriesResponse,
)
from ..controllers import (
    check_access,
    create_csv_streaming_response,
    generate_views,
    get_dataset_permission_status,
    get_harmonized_timeseries_df,
    increase_user_downloads,
    upsert_timeseries,
    view_datasets_as,
)
from ..utils import ConditionalJWTWalletAuth, fetch_permissions

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
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
    
    async def fetch_dataset_with_permissions(dataset):
        ts_ids = [ts_id for ts_id in dataset.timeseriesIDs]
        permissions = await fetch_permissions(dataset.item_hash, ts_ids, view_as)
        return DatasetResponse(
            **dataset.dict(),
            permission_status=get_dataset_permission_status(
                dataset, permissions) if view_as else None
        )

    if view_as:
        return await asyncio.gather(*[fetch_dataset_with_permissions(dataset) for dataset in datasets])
    else:
        return [DatasetResponse(**dataset.dict(), permission_status=None) for dataset in datasets]


@router.put("")
async def upload_dataset(
    dataset_req: UploadDatasetRequest,
    user: JWTWalletAuthDep,
) -> Dataset:
    """
    Upload a dataset.
    If an `item_hash` is provided, it will update the dataset with that id.
    """
    if not dataset_req.timeseriesIDs:
        raise HTTPException(status_code=400, detail="No timeseries provided")
    timeseries = await Timeseries.fetch(dataset_req.timeseriesIDs).all()
    owns_all_timeseries = all([ts.owner == user.address for ts in timeseries])
    if dataset_req.item_hash is None:
        return await Dataset(
            **dataset_req.dict(),
            owner=user.address,
            ownsAllTimeseries=owns_all_timeseries,
        ).save()
    else:
        dataset = await Dataset.fetch(dataset_req.item_hash).first()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.owner != user.address:
            raise HTTPException(
                status_code=403,
                detail="Cannot overwrite dataset that is not owned by you",
            )
        dataset.name = dataset_req.name
        dataset.desc = dataset_req.desc
        dataset.timeseriesIDs = dataset_req.timeseriesIDs
        dataset.ownsAllTimeseries = owns_all_timeseries
        dataset.price = dataset_req.price
        if dataset.changed:
            await dataset.save()
        return dataset


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, view_as: Optional[str] = None) -> DatasetResponse:
    """
    Get a dataset by its id.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="No Dataset found")

    ts_ids = [ts_id for ts_id in dataset.timeseriesIDs]
    permissions = await fetch_permissions(dataset_id, ts_ids, view_as)

    return DatasetResponse(
        **dataset.dict(),
        permission_status=get_dataset_permission_status(
            dataset, permissions) if view_as else None,
    )


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
        Permission.filter(timeseriesID=ts_id,
                          status=PermissionStatus.GRANTED).all()
        for ts_id in ts_ids
    ] + [Permission.filter(datasetID=dataset_id, status=PermissionStatus.GRANTED).all()]
    records = await asyncio.gather(*matched_permission_records)
    permission_records = [
        element for row in records for element in row if element]

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
            Attribute(trait_type="Columns", value=str(
                len(dataset.timeseriesIDs))),
        ],
    )


@router.post("/upload/timeseries")
async def upload_dataset_with_timeseries(
    req: UploadDatasetTimeseriesRequest,
    user: JWTWalletAuthDep,
) -> UploadDatasetTimeseriesResponse:
    """
    Upload a dataset and timeseries at the same time.
    """
    if req.dataset.item_hash is not None:
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update a dataset. Use PUT /datasets instead.",
        )
    if any([ts.item_hash is not None for ts in req.timeseries]):
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update timeseries. Use PUT /timeseries instead.",
        )
    updated_timeseries, created_timeseries = await upsert_timeseries(
        timeseries_requests=req.timeseries, user=user
    )
    timeseries = updated_timeseries + created_timeseries
    req.dataset.timeseriesIDs = [str(ts.item_hash) for ts in timeseries]
    dataset = await upload_dataset(dataset_req=req.dataset, user=user)
    return UploadDatasetTimeseriesResponse(
        dataset=dataset,
        timeseries=[ts for ts in timeseries if not isinstance(
            ts, BaseException)],
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


@router.get("/{dataset_id}/timeseries/json")
async def get_dataset_timeseries_with_data(
    dataset_id: str,
    user: JWTWalletAuthDep,
) -> List[TimeseriesWithData]:
    """
    Get all timeseries for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    await check_access(timeseries, user)

    df = get_harmonized_timeseries_df(timeseries)

    result = []
    for ts in timeseries:
        data = [
            (int(dt.timestamp()), value)
            for dt, value in df[ts.item_hash].items()
        ]
        result.append(
            TimeseriesWithData(
                **ts.dict(),
                data=data,
            )
        )

    return result


async def does_dataset_cost(dataset_id: str) -> bool:
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset.price is not None and float(dataset.price) > 0


OnlyIfDatasetCosts = Annotated[
    Optional[JWTWalletCredentials],
    Depends(
        ConditionalJWTWalletAuth(
            jwt_credentials_manager,
            lambda request: does_dataset_cost(
                request.path_params["dataset_id"]),
        )
    ),
]


@router.get("/{dataset_id}/timeseries/csv")
async def get_dataset_timeseries_csv(
    dataset_id: str,
    user: OnlyIfDatasetCosts,
) -> StreamingResponse:
    """
    Get all timeseries for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()

    df = get_harmonized_timeseries_df(
        timeseries, column_names=ColumnNameType.name)

    if user:
        await check_access(timeseries, user)

    owners = {ts.owner for ts in timeseries}
    dataset.downloads = dataset.downloads + 1 if dataset.downloads else 1
    await asyncio.gather(increase_user_downloads(list(owners)), dataset.save())

    stream = await create_csv_streaming_response(df, compression=False)
    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={dataset.name}.csv"

    return stream


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
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await generate_views(dataset, view_params)


@router.put("/{dataset_id}/available/{available}")
async def set_dataset_available(dataset_id: str, available: bool) -> Dataset:
    """
    Set a dataset to be available or not. This will also update the status of all
    executions that are waiting for permission on this dataset.
    param `dataset_id':put the dataset hash here
    param 'available':put the Boolean value
    """

    requests: List[Awaitable[Union[Dataset, Timeseries]]] = []
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

    await asyncio.gather(*requests)
    return dataset
