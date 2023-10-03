import asyncio
import logging
from typing import Awaitable, List, Optional, Union

from aars.utils import PageableRequest, PageableResponse
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi_walletauth import JWTWalletAuthDep
from starlette.responses import StreamingResponse

from ...core.model import Dataset, Permission, PermissionStatus, Timeseries, View
from ..api_model import (
    Attribute,
    CreateDatasetRequest,
    DataFormat,
    DatasetPermissionStatus,
    DatasetResponse,
    FungibleAssetStandard,
    PutViewRequest,
    PutViewResponse,
    UpdateDatasetRequest,
)
from ..controllers import (
    calculate_view,
    get_data_stream,
    get_dataset_df,
    load_data_df,
    merge_data,
    update_timeseries,
    update_views,
)
from ..utils import AuthorizedRouterDep, get_file_path

logger = logging.getLogger("uvicorn")

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
    dependencies=[AuthorizedRouterDep],
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


@router.get("/{dataset_id}/download")
async def download_data(
    dataset_id: str,
    user: JWTWalletAuthDep,
    dataFormat: DataFormat = DataFormat.CSV,
) -> StreamingResponse:
    """
    Download a dataset or timeseries as a file.
    """
    logger.info(f"Received download request for dataset {dataset_id} from {user}")

    dataset = await Dataset.fetch(dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset does not exist")

    if dataset.owner != user.address:
        permissions = await Permission.filter(
            datasetID=dataset_id, requestor=user.address
        ).all()
        if not permissions:
            raise HTTPException(
                status_code=403, detail="User does not have access to this dataset"
            )

    df = get_dataset_df(dataset_id)

    media_type, stream = await get_data_stream(df, dataFormat)

    response = StreamingResponse(stream, media_type=media_type)
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename={dataset_id}.{dataFormat}"

    return response


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
        external_url=f"http://127.0.0.1:5173/data/{dataset.item_hash}/details",
        attributes=[
            Attribute(trait_type="Owner", value=dataset.owner),
            Attribute(trait_type="Last Updated", value=dataset.timestamp),
            Attribute(trait_type="Columns", value=str(len(dataset.timeseriesIDs))),
        ],
    )


@router.post("")
async def upload_dataset(
    dataset_req: CreateDatasetRequest,
    user: JWTWalletAuthDep,
    file: UploadFile = File(...),
) -> Dataset:
    """
    Upload a new dataset.
    """
    df = load_data_df(file)
    timeseries = await update_timeseries(
        df, metadata=dataset_req.timeseries, timeseries=None, user=user
    )
    dataset = await Dataset(
        name=dataset_req.name,
        desc=dataset_req.desc,
        owner=user.address,
        ownsAllTimeseries=True,
        price=dataset_req.price,
        timeseriesIDs=[str(ts.item_hash) for ts in timeseries],
        viewIDs=None,
    ).save()
    return dataset


@router.put("/{dataset_id}")
async def update_dataset_info(
    dataset_id: str,
    user: JWTWalletAuthDep,
    name: Optional[str] = None,
    desc: Optional[str] = None,
    price: Optional[str] = None,
) -> Dataset:
    """
    Update a dataset's name, description or price.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset does not exist")
    if dataset.owner != user.address:
        raise HTTPException(
            status_code=403, detail="Only the dataset owner can update the dataset"
        )
    if name is not None:
        dataset.name = name
    if desc is not None:
        dataset.desc = desc
    if price is not None:
        dataset.price = price
    if dataset.changed:
        return await dataset.save()
    return dataset


@router.put("/{dataset_id}/data")
async def update_data(
    dataset_id: str,
    user: JWTWalletAuthDep,
    file: UploadFile = File(...),
    metadata: Optional[UpdateDatasetRequest] = None,
    update_values: bool = False,
):
    logger.info(f"Received upload request for {file.filename} from {user}")
    dataset = await Dataset.fetch(dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset does not exist")
    if dataset.owner != user.address:
        raise HTTPException(
            status_code=403, detail="Only the dataset owner can upload files"
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    df = load_data_df(file)
    file_path = await get_file_path(dataset_id)
    logger.info(f"Received {len(df)} rows, saving to {str(file_path)}")

    df = await merge_data(df, file_path, update_values)

    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    timeseries = await update_timeseries(
        df, metadata.timeseries if metadata else [], timeseries, user  # type: ignore
    )

    if update_values:
        await update_views(dataset_id)

    df.to_parquet(file_path)
    return timeseries


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
