from typing import List, Optional

from fastapi import APIRouter
from fastapi_walletauth import JWTWalletAuthDep

from ...core.model import Dataset, Permission, PermissionStatus, UserInfo
from ..api_model import (
    Notification,
    NotificationType,
    PermissionRequestNotification,
    PutUserInfo,
)
from ..utils import AuthorizedRouterDep

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
    dependencies=[AuthorizedRouterDep],
)


@router.get("")
async def get_users(
    username: Optional[str] = None,
    address: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[UserInfo]:
    params = {}
    if username:
        params["username"] = username
    if address:
        params["address"] = address
    if params:
        return await UserInfo.filter(**params).page(page=page, page_size=page_size)
    return await UserInfo.fetch_objects().page(page=page, page_size=page_size)


@router.put("")
async def put_user_info(
    user_info: PutUserInfo,
    user: JWTWalletAuthDep,
) -> UserInfo:
    user_record = None
    if user.address:
        user_record = await UserInfo.filter(address=user.address).first()
        if user_record:
            user_record.username = user_info.username
            user_record.bio = user_info.bio
            user_record.email = user_info.email
            user_record.link = user_info.link
            await user_record.save()
    if user_record is None:
        user_record = await UserInfo(
            username=user_info.username,
            address=user.address,
            bio=user_info.bio,
            email=user_info.email,
            link=user_info.link,
        ).save()
    return user_record


@router.get("/{address}")
async def get_specific_user(address: str) -> Optional[UserInfo]:
    return await UserInfo.filter(address=address).first()


@router.get("/{address}/permissions/incoming")
async def get_incoming_permission_requests(
    address: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.filter(authorizer=address).page(
        page=page, page_size=page_size
    )
    return permission_records


@router.get("/{address}/permissions/outgoing")
async def get_outgoing_permission_requests(
    address: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.filter(requestor=address).page(
        page=page, page_size=page_size
    )
    return permission_records


@router.get("/{address}/notifications")
async def get_notification(address: str) -> List[Notification]:
    # requests permission for a whole dataset
    permissions = await Permission.filter(
        authorizer=address, status=PermissionStatus.REQUESTED
    ).all()

    # drop duplicates by dataset & requestor
    permissions = list({(p.datasetID, p.requestor): p for p in permissions}.values())

    datasets = await Dataset.fetch([p.datasetID for p in permissions]).all()
    dataset_map = {d.item_hash: d for d in datasets}

    notifications: List[Notification] = []
    for permission in permissions:
        notifications.append(
            PermissionRequestNotification(
                type=NotificationType.PermissionRequest,
                message_text=permission.requestor
                + " has requested to access "
                + dataset_map[permission.datasetID].name,
                requestor=permission.requestor,
                datasetID=permission.datasetID,
                uses=None,
            )
        )

    return notifications
