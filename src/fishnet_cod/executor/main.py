import asyncio
import logging
from typing import Optional

from aars import AARS
from aleph.sdk.vm.app import AlephApp
from aleph_message.models import MessageType, PostMessage
from fastapi import FastAPI

from ..core.constants import EXECUTOR_MESSAGE_FILTER
from ..core.execution import run_execution, try_get_execution_from_message
from ..core.model import Execution
from ..core.session import initialize_aars

logger = logging.getLogger("uvicorn")
logger.debug("imports done")

http_app = FastAPI()
app = AlephApp(http_app=http_app)
global aars_client


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


globals_snapshot = globals().copy()


@app.event(filters=EXECUTOR_MESSAGE_FILTER)
async def handle_execution(event: PostMessage) -> Optional[Execution]:
    logger.debug(f"Received event: {event.content.type}")
    execution = await try_get_execution_from_message(event)
    if execution is not None:
        logger.info(f"Running execution: {execution.item_hash}")
        try:
            execution = await run_execution(execution)
        except Exception as e:
            if execution is not None:
                logger.info(
                    f"Failed to run execution: {execution.json(exclude_unset=True)}"
                )
            logger.exception(e)
        finally:
            # clean up globals
            for key in list(globals().keys()):
                # TODO: check hashes?
                if key not in globals_snapshot:
                    del globals()[key]
    return None


async def listen():
    global aars_client
    logger.info(f"Listening for events on {EXECUTOR_MESSAGE_FILTER}")
    async for message in aars_client.session.watch_messages(
        message_type=MessageType(EXECUTOR_MESSAGE_FILTER[0]["type"]),
        content_types=EXECUTOR_MESSAGE_FILTER[0]["post_type"],
        channels=[EXECUTOR_MESSAGE_FILTER[0]["channel"]],
    ):
        if isinstance(message, PostMessage):
            await handle_execution(message)
        else:
            print(f"Received invalid message: {message}")
