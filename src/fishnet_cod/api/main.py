import asyncio
import logging
import os
from os import listdir
from typing import Optional

from aars import AARS, Record
from aleph.sdk.vm.app import AlephApp
from aleph_message.models import PostMessage
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

from ..core.constants import API_MESSAGE_FILTER, FISHNET_MESSAGE_CHANNEL
from ..core.model import (
    Algorithm,
    Dataset,
    Execution,
    Permission,
    Result,
    Timeseries,
    UserInfo,
    View,
)
from ..core.session import initialize_aars
from .api_model import MessageResponse
from .routers import (
    algorithms,
    datasets,
    executions,
    permissions,
    results,
    timeseries,
    users,
)

logger = (
    logging.getLogger(__name__)
    if __name__ != "__main__"
    else logging.getLogger("uvicorn")
)
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

http_app.include_router(algorithms.router)
http_app.include_router(datasets.router)
http_app.include_router(executions.router)
http_app.include_router(permissions.router)
http_app.include_router(results.router)
http_app.include_router(timeseries.router)
http_app.include_router(users.router)

app = AlephApp(http_app=http_app)


async def re_index():
    logger.info(f"API re-indexing channel {AARS.channel}")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@app.on_event("startup")
async def startup():
    initialize_aars()
    await re_index()


@app.get("/")
async def index():
    if os.path.exists("/opt/venv"):
        opt_venv = list(listdir("/opt/venv"))
    else:
        opt_venv = []
    return {
        "vm_name": FISHNET_MESSAGE_CHANNEL,
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
        "View",
        "UserInfo",
        "Result",
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
