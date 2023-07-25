import asyncio
import io
from typing import List

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from starlette.responses import StreamingResponse

from ..common import OptionalWalletAuthDep, get_harmonized_timeseries
from ...core.model import Timeseries
from ..api_model import UploadTimeseriesRequest, ColumnNameType

router = APIRouter(
    prefix="/timeseries",
    tags=["timeseries"],
    responses={404: {"description": "Not found"}},
    dependencies=[OptionalWalletAuthDep],
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


@router.post("/csv/download")
async def download_timeseries_csv(
    timeseriesIDs: List[str],
    column_names: ColumnNameType = ColumnNameType.item_hash,
    compression: bool = False,
) -> StreamingResponse:
    """
    Download a csv file with timeseries data. The csv file will have a `timestamp` column and a column for each
    timeseries. The column name of each timeseries is either the `item_hash` or the `name` of the timeseries,
    depending on the `column_names` parameter.

    If timeseries timestamps do not align perfectly, the missing values will be filled with the last known value.

    If `compression` is set to `True`, the csv file will be compressed with gzip.
    """

    df = await get_harmonized_timeseries(timeseriesIDs, column_names=column_names)

    # Create an in-memory text stream
    stream = io.StringIO()
    # Write the DataFrame to the stream as CSV
    if compression:
        df.to_csv(stream, compression="gzip")
    else:
        df.to_csv(stream)
    # Set the stream position to the start
    stream.seek(0)
    # Create a streaming response with the stream and appropriate headers
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")

    # Generate a hash from the timeseries IDs and use it as the filename
    filename = hash("".join(timeseriesIDs))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}.csv"

    return response
