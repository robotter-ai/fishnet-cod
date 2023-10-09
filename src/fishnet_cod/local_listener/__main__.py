import os
import time
from logging import INFO, basicConfig, getLogger

import requests
from aleph_message.models import MessageType, PostMessage

from ..core.conf import settings
from ..core.session import initialize_aars

basicConfig(level=INFO)
logger = getLogger(__name__)

API_URL = os.getenv("API_URL")


async def handle_message(event: PostMessage) -> None:
    logger.info(f"Received event: {event.content.type} {str(event.item_hash)}")
    if event.content.type in settings.API_MESSAGE_FILTER[0]["post_type"]:
        logger.debug(f"Sending event to API: {event}")
        # call the api POST /event endpoint on localhost:8000
        # retry if it fails every 5 seconds until it succeeds
        while True:
            try:
                requests.post(f"{API_URL}/event", data=event.json())
                break
            except:
                logger.info("Failed to send event to API, retrying in 5 seconds")
                time.sleep(5)
                continue
    return None


async def listen():
    aars_client = await initialize_aars()
    while True:
        listen_dict = {
            "start_date": time.time(),
            "message_type": MessageType(settings.API_MESSAGE_FILTER[0]["type"]),
            "content_types": settings.API_MESSAGE_FILTER[0]["post_type"],
            "channels": [settings.API_MESSAGE_FILTER[0]["channel"]],
        }
        logger.info(f"Listening for events on {listen_dict}")
        async for message in aars_client.session.watch_messages(**listen_dict):
            if isinstance(message, PostMessage):
                await handle_message(message)
            else:
                logger.warning(f"Received invalid message: {message.type}")
        logger.info("Restarting websocket connection to Aleph API")


async def main():
    await listen()


if __name__ == "__main__":
    import asyncio

    # listen forever
    fut = asyncio.ensure_future(main())
    fut.add_done_callback(lambda fut: asyncio.get_event_loop().stop())
    asyncio.get_event_loop().run_forever()
