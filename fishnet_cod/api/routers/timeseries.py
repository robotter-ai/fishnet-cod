import asyncio
import io
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi_walletauth import JWTWalletAuthDep
from pydantic import ValidationError
from starlette.responses import StreamingResponse

from ...core.model import Timeseries, UserInfo
from ..api_model import ColumnNameType, TimeseriesWithData, UploadTimeseriesRequest
from ..controllers import get_harmonized_timeseries_df, upsert_timeseries
from ..utils import AuthorizedRouterDep

router = APIRouter(
    prefix="/timeseries",
    tags=["timeseries"],
    responses={404: {"description": "Not found"}},
    dependencies=[AuthorizedRouterDep],
)


@router.put("")
async def upload_timeseries(
    req: UploadTimeseriesRequest,
    user: JWTWalletAuthDep,
) -> List[Timeseries]:
    """
    Upload a list of timeseries. If the passed timeseries has an `item_hash` and it already exists,
    it will be overwritten. If the timeseries does not exist, it will be created.
    A list of the created/updated timeseries is returned. If the list is shorter than the passed list, then
    it might be that a passed timeseries contained illegal data.
    """
    return await upsert_timeseries(req.timeseries, user.address)


@router.post("/csv")
async def preprocess_timeseries_csv(
    user: JWTWalletAuthDep,
    data_file: UploadFile = File(...),
) -> List[TimeseriesWithData]:
    """
    Preprocess a csv file with timeseries data. The csv file must have a header row with the following columns:
    `item_hash`, `name`, `desc`, `data`. The `data` column must contain a json string with the timeseries data.
    The returned list of timeseries will not be persisted yet.
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
                TimeseriesWithData(
                    item_hash=None,
                    name=col,
                    desc=None,
                    data=data,
                    owner=user.address,
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
    column_names: ColumnNameType = ColumnNameType.name,
    compression: bool = False,
) -> StreamingResponse:
    """
    Download a csv file with timeseries data. The csv file will have a `timestamp` column and a column for each
    timeseries. The column name of each timeseries is either the `item_hash` or the `name` of the timeseries,
    depending on the `column_names` parameter.

    If timeseries timestamps do not align perfectly, the missing values will be filled with the last known value.

    If `compression` is set to `True`, the csv file will be compressed with gzip.
    """
    # fetch required permissions
    timeseries = await Timeseries.fetch(timeseriesIDs).all()
    df = get_harmonized_timeseries_df(timeseries, column_names=column_names)

    # increase download count
    owners = {ts.owner for ts in timeseries}
    user_infos = await UserInfo.filter(address__in=owners).all()
    requests = []
    for user_info in user_infos:
        user_info.downloads = user_info.downloads + 1 if user_info.downloads else 1
        requests.append(user_info.save())
    await asyncio.gather(*requests)

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
