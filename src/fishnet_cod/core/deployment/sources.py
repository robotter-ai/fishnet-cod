import logging
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from aleph.sdk.client import AuthenticatedUserSessionSync
from aleph.sdk.types import StorageEnum
from aleph_message.models import ItemHash, MessageType, StoreMessage
from semver import VersionInfo

from ..constants import FISHNET_DEPLOYMENT_CHANNEL
from ..version import VERSION_STRING

logger = logging.getLogger(__name__)


class SourceType(Enum):
    EXECUTOR = "executor"
    REQUIREMENTS = "requirements"
    API = "api"


def fetch_latest_source(
    deployer_session: AuthenticatedUserSessionSync,
    source_code_refs: List[Union[str, ItemHash]],
):
    # Get latest version executors and source code
    source_messages = deployer_session.get_messages(
        hashes=source_code_refs, message_type=MessageType.store
    )
    latest_source: Optional[StoreMessage] = None
    for source in source_messages.messages:
        assert (
            source.content.protocol_version
        ), "[PANIC] Encountered source_code message with no version!\n" + str(
            source.json()
        )
        if not latest_source:
            latest_source = source
        elif VersionInfo.parse(source.content.protocol_version) == VersionInfo.parse(
            latest_source.content.protocol_version
        ):
            latest_source = source
    return latest_source


def upload_source(
    deployer_session: AuthenticatedUserSessionSync,
    path: Path,
    source_type: SourceType,
    channel=FISHNET_DEPLOYMENT_CHANNEL,
) -> StoreMessage:
    logger.debug(f"Reading {source_type.name} file")
    with open(path, "rb") as fd:
        file_content = fd.read()
    storage_engine = (
        StorageEnum.ipfs if len(file_content) > 4 * 1024 * 1024 else StorageEnum.storage
    )
    logger.debug(f"Uploading {source_type.name} sources to {storage_engine}")
    user_code, status = deployer_session.create_store(
        file_content=file_content,
        storage_engine=storage_engine,
        channel=channel,
        guess_mime_type=True,
        extra_fields={
            "source_type": source_type.name,
            "protocol_version": VERSION_STRING,
        },
    )
    logger.debug(f"{source_type.name} upload finished")
    return user_code


def build_and_upload_requirements(
    requirements_path: Path,
    deployer_session: AuthenticatedUserSessionSync,
    source_type: SourceType,
    channel: str = FISHNET_DEPLOYMENT_CHANNEL,
) -> StoreMessage:
    if source_type != SourceType.REQUIREMENTS:
        raise Exception(
            f"Source type {source_type.name} is not supported for requirements upload"
        )
    logger.debug(f"Building {source_type.name}")
    opt_packages = Path(
        "/opt/packages"
    )  # /opt/packages is by default imported into Python
    # check if directory exists, clean if necessary
    if not opt_packages.exists():
        opt_packages.mkdir()
    else:
        shutil.rmtree(opt_packages)
        opt_packages.mkdir()
    # install requirements
    subprocess.run(
        ["pip", "install", "-t", str(opt_packages), "-r", str(requirements_path)],
        check=True,
    )
    # build file system image
    squashfs_path = requirements_path.parent / "packages.squashfs"
    subprocess.run(
        ["mksquashfs", str(opt_packages), str(squashfs_path)],
        check=True,
    )
    # remove temporary directory
    shutil.rmtree(opt_packages)
    # upload requirements
    return upload_source(
        deployer_session, path=squashfs_path, source_type=source_type, channel=channel
    )
