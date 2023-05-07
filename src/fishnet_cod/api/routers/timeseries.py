import asyncio
from typing import List

import pandas as pd
from fastapi import HTTPException, Form, UploadFile, File
from pydantic import ValidationError

from ..api_model import UploadTimeseriesRequest
from ..main import app
from ...core.model import Timeseries


@app.put("/timeseries/upload")
async def upload_timeseries(req: UploadTimeseriesRequest) -> List[Timeseries]:
    """
    Upload a list of timeseries. If the passed timeseries has an `id_hash` and it already exists,
    it will be overwritten. If the timeseries does not exist, it will be created.
    A list of the created/updated timeseries is returned. If the list is shorter than the passed list, then
    it might be that a passed timeseries contained illegal data.
    """
    ids_to_fetch = [ts.id_hash for ts in req.timeseries if ts.id_hash is not None]
    requests = []
    old_time_series = (
        {ts.id_hash: ts for ts in await Timeseries.fetch(ids_to_fetch).all()}
        if ids_to_fetch
        else {}
    )
    for ts in req.timeseries:
        if old_time_series.get(ts.id_hash) is None:
            requests.append(Timeseries(**dict(ts)).save())
            continue
        old_ts: Timeseries = old_time_series[ts.id_hash]
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


@app.post("/timeseries/csv/preprocess")
async def upload_timeseries_csv(
    owner: str = Form(...), data_file: UploadFile = File(...)
) -> List[Timeseries]:
    """
    Upload a csv file with timeseries data. The csv file must have a header row with the following columns:
    `id_hash`, `name`, `desc`, `data`. The `data` column must contain a json string with the timeseries data.
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
            print(f"Using column {col} as index")
            df = df.drop(columns=col)
    # create a timeseries object for each column
    timestamps = [dt.timestamp() for dt in df.index.to_pydatetime().tolist()]
    timeseries = []
    for col in df.columns:
        try:
            data = [(timestamps[i], value) for i, value in enumerate(df[col].tolist())]
            timeseries.append(
                Timeseries(
                    id_hash=None,
                    name=col,
                    desc=None,
                    data=data,
                    owner=owner,
                )
            )
        except ValidationError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {data}",
            )
    return timeseries
