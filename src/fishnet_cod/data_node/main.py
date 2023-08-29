# A FastAPI app accepting CSV, parquet and feather file uploads. Uses wallet authentication of the Aleph API.
# It allows to download the CSVs to users that have access to a particular dataset.
# This code runs on multiple machines, where each machine has a different slice of the dataset.
# It retrieves info about the datasets, permissions and network configuration from the Aleph API.
import asyncio
import logging

import pandas as pd
from aars import AARS
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends

from aleph.sdk.client import AlephClient
from aleph.sdk.conf import settings
from aleph.sdk.vm.app import AlephApp
from pydantic import ValidationError

from ..core.model import Dataset, Permission, Timeseries, UserInfo, TimeseriesSliceStats, Slice
from ..core.session import initialize_aars

logger = logging.getLogger("uvicorn")
logger.debug("imports done")

http_app = FastAPI()
aleph_client = AlephClient(settings.API_HOST)
aars_client = initialize_aars()
fishnet_config = aleph_client.fetch_aggregate("fishnet", "config").json()
app = AlephApp(http_app=http_app)


async def re_index():
    logger.info("API re-indexing")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@app.on_event("startup")
async def startup():
    global aars_client
    aars_client = initialize_aars()
    await re_index()


@app.get("/")
async def index():
    return {"status": "ok"}


@app.post("/upload")
async def upload(
    datasetID: str,
    file: UploadFile = File(...),
    user: UserInfo = Depends(app.user_info),
):
    logger.info(f"Received upload request for {file.filename} from {user}")
    dataset = await Dataset.fetch(datasetID).first()
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset does not exist")
    if dataset.owner != user.address:
        raise HTTPException(status_code=403, detail="Only the dataset owner can upload files")
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
    slice = await Slice(
        datasetID=datasetID,
        timeseriesStats=timeseries_stats,
    ).save()

    file_path = f"./files/{datasetID}.parquet"
    logger.info(f"Received {len(df)} rows, saving to {file_path}")
    df.to_parquet(file_path)
    return slice


@app.get("/download")
async def download(
    datasetID: str,
    user: UserInfo = Depends(app.user_info),
):
    logger.info(f"Received download request for dataset {datasetID} from {user}")

    timeseries = await Timeseries.filter(datasetID=datasetID).all()
        if timeseries is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if not await Permission.has_permission(user, timeseries):
            raise HTTPException(status_code=403, detail="No permission to download this dataset")
        return {"url": timeseries.url}
    else:
        raise HTTPException(status_code=400, detail="Unsupported dataset type (only timeseries are supported)")
    #