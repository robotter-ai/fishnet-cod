# A FastAPI app accepting CSV, parquet and feather file uploads. Uses wallet authentication of the Aleph API.
# It allows to download the CSVs to users that have access to a particular dataset.
# This code runs on multiple machines, where each machine has a different slice of the dataset.
# It retrieves info about the datasets, permissions and network configuration from the Aleph API.
import asyncio
import io
import logging
import os
from enum import Enum
from typing import Union

import pandas as pd
from aars import AARS
from fastapi import FastAPI, File, UploadFile, HTTPException

from aleph.sdk.client import AlephClient
from aleph.sdk.conf import settings
from aleph.sdk.vm.app import AlephApp
from fastapi_walletauth import JWTWalletAuthDep
from pydantic import ValidationError
from starlette.responses import StreamingResponse, RedirectResponse

from ..core.model import Dataset, Timeseries, TimeseriesSliceStats, DatasetSlice, FishnetConfig, Permission
from ..core.session import initialize_aars

logger = logging.getLogger("uvicorn")
logger.debug("imports done")

http_app = FastAPI()
aleph_client = AlephClient(settings.API_HOST)
aars_client = initialize_aars()
app = AlephApp(http_app=http_app)
fishnet_config: FishnetConfig


class DataFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"


async def re_index():
    logger.info("API re-indexing")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@app.on_event("startup")
async def startup():
    global aars_client, fishnet_config
    aars_client = await initialize_aars()
    fishnet_config = FishnetConfig.from_dict(await aleph_client.fetch_aggregate("fishnet", "config"))
    await re_index()


@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=301)


@app.post("/upload")
async def upload(
    datasetID: str,
    user: JWTWalletAuthDep,
    file: UploadFile = File(...),
):
    logger.info(f"Received upload request for {file.filename} from {user}")
    dataset = await Dataset.fetch(datasetID).first()
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset does not exist")
    if dataset.owner != user.address:
        raise HTTPException(status_code=403, detail="Only the dataset owner can upload files")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith(".parquet"):
        df = pd.read_parquet(file.file)
    elif file.filename.endswith(".feather"):
        df = pd.read_feather(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format (only CSV, parquet and feather are supported)")

    logger.info(f"Calculating slice stats for {datasetID}")
    # find the first column with a timestamp or ISO8601 date and use it as the index
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=col)
            continue
        if "date" in col.lower() or "time" in col.lower():
            df.index = pd.to_datetime(df[col])
            df = df.drop(columns=col)
    timeseries = await Timeseries.filter(datasetID=datasetID).all()
    timeseries_stats = {}
    for col in df.columns:
        try:
            timeseries_id = next(ts.item_hash for ts in timeseries if ts.name == col)
            assert timeseries_id is not None
            timeseries_stats[timeseries_id] = (
                TimeseriesSliceStats(
                    min=df[col].min(),
                    max=df[col].max(),
                    avg=df[col].mean(),
                    std=df[col].std(),
                    median=df[col].median(),
                )
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {e}",
            )
    slice = await DatasetSlice(
        datasetID=datasetID,
        locationUrl=f"{fishnet_config.nodes[app.vm_hash].url}/download?datasetID={datasetID}",
        timeseriesStats=timeseries_stats,
        startTime=int(df.index.min().timestamp()),
        endTime=int(df.index.max().timestamp()),
    ).save()

    file_path = f"./files/{datasetID}.parquet"
    logger.info(f"Received {len(df)} rows, saving to {file_path}")
    existing_df = pd.read_parquet(file_path) if os.path.exists(file_path) else pd.DataFrame()
    # columns stay the same, but it might be that the new data has more rows
    df = pd.concat([existing_df, df])
    # sort by index
    df = df.sort_index()
    df.to_parquet(file_path)
    return slice


@app.get("/download")
async def download(
    datasetID: str,
    user: JWTWalletAuthDep,
    dataFormat: DataFormat = DataFormat.CSV,
) -> StreamingResponse:
    """
    Download a dataset or timeseries as a file.
    """
    logger.info(f"Received download request for dataset {datasetID} from {user}")

    dataset = await Dataset.fetch(datasetID).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset does not exist")

    if dataset.owner != user.address:
        permissions = await Permission.filter(
            datasetID=datasetID, requestor=user.address
        ).all()
        if not permissions:
            raise HTTPException(status_code=403, detail="User does not have access to this dataset")

    file_path = f"./files/{datasetID}.parquet"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset slice not found on this node")
    df = pd.read_parquet(file_path)

    stream: Union[io.StringIO, io.BytesIO]
    if dataFormat == DataFormat.CSV:
        stream = io.StringIO()
        df.to_csv(stream)
        media_type = "text/csv"
    elif dataFormat == DataFormat.FEATHER:
        stream = io.BytesIO()
        df.to_feather(stream)
        media_type = "application/octet-stream"
    elif dataFormat == DataFormat.PARQUET:
        stream = io.BytesIO()
        df.to_parquet(stream)
        media_type = "application/octet-stream"
    else:
        raise HTTPException(status_code=400, detail="Unsupported data format")
    stream.seek(0)

    def stream_generator():
        yield stream.getvalue()

    response = StreamingResponse(stream_generator(), media_type=media_type)
    response.headers["Content-Disposition"] = f"attachment; filename={datasetID}.{dataFormat}"

    return response
