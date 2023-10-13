import asyncio
import os
from typing import Dict, List, Tuple

import pandas as pd
from fastapi import HTTPException, UploadFile
from fastapi_walletauth.middleware import JWTWalletAuthDep
from pydantic import ValidationError

from ..core.model import Dataset, Permission, PermissionStatus, Timeseries, View
from .api_model import (
    ColumnNameType,
    DatasetPermissionStatus,
    DatasetResponse,
    PutViewRequest,
    PutViewResponse,
    PutTimeseriesRequest,
)
from .utils import get_file_path, granularity_to_interval, find_first_row_with_comma, is_timestamp_column


async def get_dataset_df(
    dataset: Dataset,
    column_names: ColumnNameType = ColumnNameType.item_hash,
) -> pd.DataFrame:
    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    return get_harmonized_timeseries_df(
        timeseries=timeseries,
        column_names=column_names,
    )


def get_harmonized_timeseries_df(
    timeseries: List[Timeseries],
    column_names: ColumnNameType = ColumnNameType.item_hash,
) -> pd.DataFrame:
    # parse all as series
    df = pd.DataFrame()
    for ts in timeseries:
        assert ts.item_hash
        ts_df = pd.read_parquet(get_file_path(ts.item_hash))
        if column_names == ColumnNameType.name:
            ts_df.columns = [ts.name]
        elif column_names == ColumnNameType.item_hash:
            ts_df.columns = [ts.item_hash]
        else:
            raise ValueError("Invalid column name type")
        df = pd.concat([df, ts_df], axis=1)

    # set the index to the python timestamp
    df.index = pd.to_datetime(df.index, unit="s")
    df = df.ffill()

    return df


def load_data_df(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please provide a filename")
    if file.filename.endswith(".csv"):
        row = find_first_row_with_comma(file)
        file.file.seek(0)
        df = pd.read_csv(file.file, skiprows=row)
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
        if is_timestamp_column(col):
            df.index = pd.to_datetime(df[col], infer_datetime_format=True)
            df = df.drop(columns=col)
        # non-numeric values are not supported and dropped
        elif df[col].dtype == object:
            df = df.drop(columns=col)
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid columns found in file",
        )
    return df


async def merge_and_store_data(full_df: pd.DataFrame, update_values: bool = False):
    for item_hash in full_df.columns:
        file_path = get_file_path(item_hash)
        existing_df = (
            pd.read_parquet(file_path) if os.path.exists(file_path) else pd.DataFrame()
        )
        df = full_df[[item_hash]]
        df = pd.concat([existing_df, df])
        df = df[~df.index.duplicated(keep="last" if update_values else "first")]
        df = df.sort_index()
        df.to_parquet(file_path)
        full_df[item_hash] = df[item_hash]
    return full_df


async def upsert_timeseries(
    timeseries_requests: List[PutTimeseriesRequest],
    user: JWTWalletAuthDep,
) -> Tuple[List[Timeseries], List[Timeseries]]:
    """
    Upsert timeseries. If the timeseries already exists, update the metadata. If the timeseries does not exist, create
    it.

    Args:
        timeseries_requests: A list of timeseries metadata inputs.
        user: The user requesting the timeseries.

    Returns:
        A list of updated and a list of new timeseries.
    """
    update_timeseries = []
    new_timeseries = []
    for ts in timeseries_requests:
        if ts.item_hash is None:
            ts.owner = user.address
            new_timeseries.append(ts)
        else:
            update_timeseries.append(ts)

    existing_timeseries = await Timeseries.fetch([ts.item_hash for ts in update_timeseries]).all()

    if len(existing_timeseries) != len(update_timeseries):
        raise HTTPException(
            status_code=400,
            detail=f"{len(update_timeseries) - len(existing_timeseries)} Timeseries do not exist, aborting",
        )

    update_df = pd.DataFrame()
    for ts in update_timeseries:
        update_df = concat_timeseries(ts, update_df, column_names=ColumnNameType.item_hash)
    update_df = update_df.sort_index()
    update_df.ffill(inplace=True)
    existing_df = await merge_and_store_data(update_df, update_values=True)

    new_df = pd.DataFrame()
    for ts in new_timeseries:
        new_df = concat_timeseries(ts, new_df, column_names=ColumnNameType.name)

    updated_timeseries, created_timeseries = await update_timeseries_metadata(
        existing_df=existing_df,
        new_df=new_df,
        existing_timeseries=existing_timeseries,
        new_timeseries=new_timeseries,
    )

    for ts in created_timeseries:
        new_df[ts.item_hash] = new_df[ts.name]
        file_path = get_file_path(ts.item_hash)
        new_df[[ts.item_hash]].to_parquet(file_path)

    return updated_timeseries, created_timeseries


def concat_timeseries(ts, update_df, column_names: ColumnNameType = ColumnNameType.item_hash):
    update_df = pd.concat(
        [
            update_df,
            pd.DataFrame(
                [value for timestamp, value in ts.data],
                index=[pd.to_datetime(timestamp, unit="s") for timestamp, value in ts.data],
                columns=[ts.item_hash if column_names == ColumnNameType.item_hash else ts.name],
            ),
        ],
        axis=1,
    )
    return update_df


async def update_timeseries_metadata(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    existing_timeseries: List[Timeseries],
    new_timeseries: List[PutTimeseriesRequest],
) -> Tuple[List[Timeseries], List[Timeseries]]:
    """
    Args:
        existing_df: A dataframe of existing timeseries by their item_hash.
        new_df: A dataframe of new timeseries by their name.
        existing_timeseries: A list of existing timeseries metadata objects.
        new_timeseries: A list of new timeseries metadata inputs.

    Returns:
        A list of updated and a list of new timeseries.
    """
    new_by_name: Dict[str, PutTimeseriesRequest] = (
        {ts.name: ts for ts in new_timeseries} if new_timeseries else {}
    )
    create_timeseries_requests = []
    for col in new_df.columns:
        try:
            metadata = new_by_name.get(col)
            ts = Timeseries(
                item_hash=None,
                name=col,
                desc=metadata.desc if metadata else None,
                owner=metadata.owner,
                min=new_df[col].min(),
                max=new_df[col].max(),
                avg=new_df[col].mean(),
                std=new_df[col].std(),
                median=new_df[col].median(),
                earliest=int(new_df[col].index.min().timestamp()),
                latest=int(new_df[col].index.max().timestamp()),
            )
            create_timeseries_requests.append(ts.save())
        except (ValidationError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {e}",
            )
    created_timeseries = await asyncio.gather(*create_timeseries_requests)

    existing_by_item_hash: Dict[str, Timeseries] = (
        {ts.item_hash: ts for ts in existing_timeseries} if existing_timeseries else {}
    )
    update_timeseries_requests = []
    for col in existing_df.columns:
        try:
            ts = existing_by_item_hash[col]
            ts.min = existing_df[col].min()
            ts.max = existing_df[col].max()
            ts.avg = existing_df[col].mean()
            ts.std = existing_df[col].std()
            ts.median = existing_df[col].median()
            ts.earliest = int(existing_df[col].index.min().timestamp())
            ts.latest = int(existing_df[col].index.max().timestamp())
            if ts.changed:
                update_timeseries_requests.append(ts.save())
            else:
                update_timeseries_requests.append(ts)
        except (ValidationError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {e}",
            )
    updated_timeseries = await asyncio.gather(*update_timeseries_requests)

    return updated_timeseries, created_timeseries


async def generate_views(dataset: Dataset, view_params: List[PutViewRequest]):
    view_ids = [view.item_hash for view in view_params if view.item_hash is not None]
    views_map = {view.item_hash: view for view in await View.fetch(view_ids).all()}

    full_df = await get_dataset_df(dataset)
    view_requests = []
    for view_req in view_params:
        df = full_df[view_req.timeseriesIDs].copy()
        timeseries = await Timeseries.fetch(view_req.timeseriesIDs).all()
        end_time, start_time, view_values = await calculate_view(df, view_req)
        columns = []
        for ts_item_hash in view_values.keys():
            timeseries_name = next(
                ts.name for ts in timeseries if ts.item_hash == ts_item_hash
            )
            columns.append(timeseries_name)

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
                    columns=list(columns),
                    values=view_values,
                ).save()
            )

    views = await asyncio.gather(*view_requests)
    dataset.viewIDs = [view.item_hash for view in views]
    await dataset.save()

    return PutViewResponse(dataset=dataset, views=views)


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
    df = df.round(3)

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


async def check_access(timeseries, user):
    timeseries_ids = [ts.item_hash for ts in timeseries]
    if any(ts.owner != user.address for ts in timeseries):
        datasets = [dataset for dataset_list in [
            await Dataset.fetch_objects().all()
        ] for dataset in dataset_list if any(ts.item_hash in dataset.timeseriesIDs for ts in timeseries)]

        permissions = await Permission.filter(
            timeseriesID__in=timeseries_ids, requestor=user.address
        ).all() + await Permission.filter(
            datasetID__in=[dataset.item_hash for dataset in datasets], requestor=user.address
        ).all()
        for ts in timeseries:
            if ts.owner != user.address:
                permitted = False
                dataset = next((dataset for dataset in datasets if ts.item_hash in dataset.timeseriesIDs), None)
                if dataset and dataset.price == "0":
                    permitted = True
                else:
                    for permission in permissions:
                        if permission.timeseriesID == ts.item_hash and permission.status == PermissionStatus.GRANTED:
                            permitted = True
                            break
                        elif permission.datasetID and permission.status == PermissionStatus.GRANTED:
                            dataset = [dataset for dataset in datasets if permission.datasetID == dataset.item_hash][0]
                            if ts.item_hash in dataset.timeseriesIDs:
                                permitted = True
                                break
                if not permitted:
                    raise HTTPException(status_code=403, detail="You do not own all timeseries.")
    return timeseries_ids
