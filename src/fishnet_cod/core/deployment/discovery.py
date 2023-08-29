from typing import List, Optional

from aleph.sdk.client import AuthenticatedUserSessionSync
from aleph_message.models import MessageType, ProgramMessage

from ..conf import settings


def discover_executors(
    owner: str,
    session: AuthenticatedUserSessionSync,
    channel: str = settings.CONFIG_CHANNEL,
    tags: Optional[List[str]] = None,
) -> List[ProgramMessage]:
    tags = tags if tags else ["executor"]
    resp = session.get_messages(
        channels=[channel],
        addresses=[owner],
        tags=tags,
        message_type=MessageType.program,
    )
    return resp.messages


def discover_apis(
    owner: str,
    session: AuthenticatedUserSessionSync,
    channel: str = settings.CONFIG_CHANNEL,
    tags: Optional[List[str]] = None,
) -> List[ProgramMessage]:
    tags = tags if tags else ["api"]
    resp = session.get_messages(
        channels=[channel],
        addresses=[owner],
        tags=tags,
        message_type=MessageType.program,
    )
    return resp.messages
