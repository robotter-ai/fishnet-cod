import asyncio
import os
from os import listdir
from typing import Optional

import pandas as pd  # type: ignore
from aars import AARS, Record
from aleph.sdk.exceptions import BadSignatureError  # type: ignore
from aleph.sdk.vm.app import AlephApp  # type: ignore
from aleph_message.models import PostMessage  # type: ignore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

from ..core.constants import API_MESSAGE_FILTER
from ..core.model import (Algorithm, Dataset, Execution, Permission, Result,
                          Timeseries, UserInfo, View)
from ..core.session import initialize_aars
from .api_model import MessageResponse

http_app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/challenge")

origins = ["*"]

http_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = AlephApp(http_app=http_app)
global aars_client
aars_client: AARS


async def re_index():
    logger.info(f"API re-indexing channel {AARS.channel}")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@app.on_event("startup")
async def startup():
    global aars_client
    aars_client = initialize_aars()
    await re_index()


@app.get("/")
async def index():
    if os.path.exists("/opt/venv"):
        opt_venv = list(listdir("/opt/venv"))
    else:
        opt_venv = []
    return {
        "vm_name": "fishnet_api",
        "endpoints": [
            "/docs",
        ],
        "files_in_volumes": {
            "/opt/venv": opt_venv,
        },
    }


@app.delete("/clear/records")
async def empty_records() -> MessageResponse:
    await UserInfo.forget_all()
    await Timeseries.forget_all()
    await View.forget_all()
    await Dataset.forget_all()
    await Algorithm.forget_all()
    await Execution.forget_all()
    await Permission.forget_all()
    await Result.forget_all()
    return MessageResponse(response="All records are cleared")


@app.post("/event")
async def event(event: PostMessage):
    await fishnet_event(event)


@app.event(filters=API_MESSAGE_FILTER)
async def fishnet_event(event: PostMessage):
    record: Optional[Record]
    print("fishnet_event", event)
    if event.content.type in [
        "Execution",
        "Permission",
        "Dataset",
        "Timeseries",
        "Algorithm",
    ]:
        if Record.is_indexed(event.item_hash):
            return
        cls: Record = globals()[event.content.type]
        record = await cls.from_post(event)
    else:  # amend
        if Record.is_indexed(event.content.ref):
            return
        record = await Record.fetch(event.content.ref).first()
    assert record
    for inx in record.get_indices():
        inx.add_record(record)
