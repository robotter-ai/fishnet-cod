import asyncio
from typing import List

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from ...core.model import Timeseries
from ..api_model import UploadTimeseriesRequest

router = APIRouter(
    prefix="/timeseries",
    tags=["timeseries"],
    responses={404: {"description": "Not found"}},
)


@router.put("")
async def upload_timeseries(req: UploadTimeseriesRequest) -> List[Timeseries]:
    """
    Upload a list of timeseries. If the passed timeseries has an `item_hash` and it already exists,
    it will be overwritten. If the timeseries does not exist, it will be created.
    A list of the created/updated timeseries is returned. If the list is shorter than the passed list, then
    it might be that a passed timeseries contained illegal data.
    """
    ids_to_fetch = [ts.item_hash for ts in req.timeseries if ts.item_hash is not None]
    requests = []
    old_time_series = (
        {ts.item_hash: ts for ts in await Timeseries.fetch(ids_to_fetch).all()}
        if ids_to_fetch
        else {}
    )
    for ts in req.timeseries:
        if old_time_series.get(ts.item_hash) is None:
            requests.append(Timeseries(**dict(ts)).save())
            continue
        old_ts: Timeseries = old_time_series[ts.item_hash]
        if ts.owner != old_ts.owner:
            raise HTTPException(
                status_code=403,
                detail="Cannot overwrite timeseries that is not owned by you",
            )
        old_ts.name = ts.name
        old_ts.data = ts.data
        old_ts.desc = ts.desc
        requests.append(old_ts.save())
    upserted_timeseries = await asyncio.gather(*requests)
    return [ts for ts in upserted_timeseries if not isinstance(ts, BaseException)]


@router.post("/csv")
async def upload_timeseries_csv(
    owner: str = Form(...), data_file: UploadFile = File(...)
) -> List[Timeseries]:
    """
    Upload a csv file with timeseries data. The csv file must have a header row with the following columns:
    `item_hash`, `name`, `desc`, `data`. The `data` column must contain a json string with the timeseries data.
    """
    if data_file.filename and not data_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a csv file")
    df = pd.read_csv(data_file.file)
    # find the first column with a timestamp or ISO8601 date and use it as the index
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=col)
            continue
        if "date" in col.lower() or "time" in col.lower():
            df.index = pd.to_datetime(df[col])
            df = df.drop(columns=col)
    # create a timeseries object for each column
    timestamps = [dt.timestamp() for dt in df.index.to_pydatetime().tolist()]
    timeseries = []
    for col in df.columns:
        try:
            data = [(timestamps[i], value) for i, value in enumerate(df[col].tolist())]
            timeseries.append(
                Timeseries(
                    item_hash=None,
                    name=col,
                    desc=None,
                    data=data,
                    owner=owner,
                    min=df[col].min(),
                    max=df[col].max(),
                    avg=df[col].mean(),
                    std=df[col].std(),
                    median=df[col].median(),
                )
            )
        except ValidationError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {data}",
            )
    return timeseries
