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
from aleph_message.models import PostMessage

from ..core.model import Dataset, Permission, Timeseries, UserInfo
from ..core.session import initialize_aars

logger = logging.getLogger("uvicorn")
logger.debug("imports done")

http_app = FastAPI()
aleph_client = AlephClient(settings.API_HOST)
aars_client = initialize_aars()
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
    file: UploadFile = File(...),
    user: UserInfo = Depends(app.user_info),
):
    logger.info(f"Received upload request for {file.filename} from {user}")
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    elif file.filename.endswith(".parquet"):
        df = pd.read_parquet(file.file)
    elif file.filename.endswith(".feather"):
        df = pd.read_feather(file.file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format (only CSV, parquet and feather are supported)")
    logger.info(f"Received {len(df)} rows")
    #
