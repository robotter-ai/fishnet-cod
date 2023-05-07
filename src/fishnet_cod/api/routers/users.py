from typing import List, Optional

from ...core.model import (Dataset, Permission, PermissionStatus, Result,
                           UserInfo)
from ..api_model import Notification, NotificationType, PutUserInfo
from ..main import app


@app.get("/users")
async def get_all_users(page: int = 1, page_size: int = 20) -> List[UserInfo]:
    return await UserInfo.fetch_objects().page(page=page, page_size=page_size)


@app.get("/users/{address}")
async def get_specific_user(address: str) -> Optional[UserInfo]:
    return await UserInfo.where_eq(address=address).first()


@app.get("/users/{address}/permissions/incoming")
async def get_incoming_permission_requests(
    address: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.where_eq(authorizer=address).page(
        page=page, page_size=page_size
    )
    return permission_records


@app.get("/users/{address}/permissions/outgoing")
async def get_outgoing_permission_requests(
    address: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.where_eq(requestor=address).page(
        page=page, page_size=page_size
    )
    return permission_records


@app.get("/users/{address}/results")
async def get_user_results(
    address: str, page: int = 1, page_size: int = 20
) -> List[Result]:
    return await Result.where_eq(owner=address).page(page=page, page_size=page_size)


@app.put("/users")
async def put_user_info(user_info: PutUserInfo) -> UserInfo:
    user_record = None
    if user_info.address:
        user_record = await UserInfo.where_eq(address=user_info.address).first()
        if user_record:
            user_record.username = user_info.username
            user_record.address = user_info.address
            user_record.bio = user_info.bio
            user_record.email = user_info.email
            user_record.link = user_info.link
            await user_record.save()
    if user_record is None:
        user_record = await UserInfo(
            username=user_info.username,
            address=user_info.address,
            bio=user_info.bio,
            email=user_info.email,
            link=user_info.link,
        ).save()
    return user_record


@app.get("/users/{address}/notifications")
async def get_notification(address: str) -> List[Notification]:
    # requests permission for a whole dataset
    permission_records = await Permission.where_eq(
        authorizer=address, status=PermissionStatus.REQUESTED
    ).all()

    datasets_records = [
        await Dataset.where_eq(timeseriesIDs=permission.timeseriesID).all()
        for permission in permission_records
    ]

    datasets_list = [element for row in datasets_records for element in row]
    notifications = []

    for record in datasets_list:
        notifications.append(
            Notification(
                type=NotificationType.PermissionRequest,
                message_text=address + " has requested to access " + record.name,
            )
        )
    return notifications
