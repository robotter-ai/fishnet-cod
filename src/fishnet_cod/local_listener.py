import time
from typing import Optional

import requests
from aleph_message.models import MessageType, PostMessage

from .core.constants import API_MESSAGE_FILTER
from .core.model import Execution
from .core.session import initialize_aars

aars_client = initialize_aars()


async def handle_message(event: PostMessage) -> Optional[Execution]:
    print(f"Received event: {event.content.type}")
    if event.content.type in API_MESSAGE_FILTER[0]["post_type"]:
        print(f"Sending event to API: {event}")
        # call the api POST /event endpoint on localhost:8000
        requests.post("http://localhost:8000/event", data=event.json())
    return None


async def listen():
    print(f"Listening for events on {API_MESSAGE_FILTER}")
    async for message in aars_client.session.watch_messages(
        start_date=time.time(),
        message_type=MessageType(API_MESSAGE_FILTER[0]["type"]),
        content_types=API_MESSAGE_FILTER[0]["post_type"],
        channels=[API_MESSAGE_FILTER[0]["channel"]],
    ):
        if isinstance(message, PostMessage):
            await handle_message(message)
        else:
            print(f"Received invalid message: {message.type}")


async def main():
    await listen()


if __name__ == "__main__":
    import asyncio

    fut = asyncio.ensure_future(main())
    fut.add_done_callback(lambda fut: asyncio.get_event_loop().stop())
    asyncio.get_event_loop().run_forever()
