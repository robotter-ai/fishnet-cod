from typing import Optional, List

from aleph.sdk import AlephClient
from aleph_message.models import ProgramMessage, MessageType

from ..constants import FISHNET_DEPLOYMENT_CHANNEL


async def discover_executors(
    owner: str,
    session: AlephClient,
    channel: str = FISHNET_DEPLOYMENT_CHANNEL,
    tags: Optional[List[str]] = None,
) -> List[ProgramMessage]:
    tags = tags if tags else ["executor"]
    resp = await session.get_messages(
        channels=[channel],
        addresses=[owner],
        tags=tags,
        message_type=MessageType.program,
    )
    return resp.messages


async def discover_apis(
    owner: str,
    session: AlephClient,
    channel: str = FISHNET_DEPLOYMENT_CHANNEL,
    tags: Optional[List[str]] = None,
) -> List[ProgramMessage]:
    tags = tags if tags else ["api"]
    resp = await session.get_messages(
        channels=[channel],
        addresses=[owner],
        tags=tags,
        message_type=MessageType.program,
    )
    return resp.messages
