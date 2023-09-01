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
from fastapi_walletauth import jwt_authorization_router as authorization_routes
from pydantic import ValidationError
from starlette.responses import JSONResponse, RedirectResponse

from ..core.conf import settings
from ..core.session import initialize_aars
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
http_app.include_router(authorization_routes)

app = AlephApp(http_app=http_app)


@http_app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


async def re_index():
    logger.info(f"API re-indexing channel {AARS.channel}")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@app.on_event("startup")
async def startup():
    app.aars = await initialize_aars()
    print("Syncing indices...")
    await re_index()


@app.get("/")
def root():
    new_route_url = "/docs"
    return RedirectResponse(url=new_route_url)


@app.post("/event")
async def event(event: PostMessage):
    await fishnet_event(event)


@app.event(filters=settings.API_MESSAGE_FILTER)
async def fishnet_event(event: PostMessage):
    record: Optional[Record]
    try:
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
            print(f"Received event: {event.content.type} - {event.item_hash}")
            if Record.is_indexed(event.item_hash):
                return
            cls: Record = globals()[event.content.type]
            record = await cls.from_post(event)
        else:  # amend
            record = await Record.fetch(event.content.ref).first()
        assert record
        for inx in record.get_indices():
            inx.add_record(record)
    except ValidationError as e:
        logger.error(f"Invalid event: {e}")
