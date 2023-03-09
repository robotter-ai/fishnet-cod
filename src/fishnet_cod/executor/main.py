import logging
from typing import Optional

from aleph_message.models import PostMessage, MessageType
from aleph.sdk.vm.cache import VmCache
from aleph.sdk.vm.app import AlephApp

from fastapi import FastAPI
from aars import AARS

from ..core.model import Execution
from ..core.constants import FISHNET_MESSAGE_CHANNEL
from ..core.execution import run_execution, try_get_execution_from_message

logger = logging.getLogger("uvicorn")
logger.debug("imports done")

http_app = FastAPI()
app = AlephApp(http_app=http_app)
cache = VmCache()
aars_client = AARS(channel=FISHNET_MESSAGE_CHANNEL)


@app.get("/")
async def index():
    return {"status": "ok"}


filters = [
    {
        "channel": aars_client.channel,
        "type": "POST",
        "post_type": ["Execution", "amend"],
    }
]

globals_snapshot = globals().copy()


@app.event(filters=filters)
async def handle_execution(event: PostMessage) -> Optional[Execution]:
    logger.debug(f"Received event: {event.content.type}")
    execution = await try_get_execution_from_message(event)
    if execution is not None:
        logger.info(f"Running execution: {execution.id_hash}")
        try:
            execution = await run_execution(execution)
        except Exception as e:
            if execution is not None:
                logger.info(f"Failed to run execution: {execution.json(exclude_unset=True)}")
            logger.exception(e)
        finally:
            # clean up globals
            for key in list(globals().keys()):
                # TODO: check hashes?
                if key not in globals_snapshot:
                    del globals()[key]
    return None


async def listen():
    logger.info(f"Listening for events on {filters}")
    async for message in aars_client.session.watch_messages(
        message_type=MessageType(filters[0]["type"]),
        content_types=filters[0]["post_type"],
        channels=[filters[0]["channel"]],
    ):
        if isinstance(message, PostMessage):
            await handle_execution(message)
        else:
            print(f"Received invalid message: {message}")
