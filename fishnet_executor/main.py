import logging
from typing import Optional

from aleph_message.models import PostMessage
from aleph.sdk.vm.cache import VmCache
from aleph.sdk.vm.app import AlephApp

from fastapi import FastAPI
from aars import AARS
from fishnet_cod import Execution, run_execution, try_get_execution_from_message

logger = logging.getLogger(__name__)
logger.debug("imports done")

http_app = FastAPI()
app = AlephApp(http_app=http_app)
cache = VmCache()
aars_client = AARS(channel="FISHNET_TEST")


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


@app.event(filters=filters)
async def handle_execution(event: PostMessage) -> Optional[Execution]:
    execution = await try_get_execution_from_message(event)
    return await run_execution(execution)
