import logging
import shutil
import subprocess
from base64 import b16decode, b32encode
from enum import Enum
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from aleph.sdk.conf import settings
from aleph_message.models.program import ImmutableVolume, PersistentVolume
from semver import VersionInfo

from aleph.sdk import AuthenticatedAlephClient
from aleph.sdk.types import StorageEnum
from aleph.sdk.utils import create_archive
from aleph_message.models import StoreMessage, MessageType, ProgramMessage

from .discovery import discover_executors, discover_apis
from ..constants import (
    FISHNET_DEPLOYMENT_CHANNEL,
    EXECUTOR_MESSAGE_FILTER,
    VM_URL_PATH,
    VM_URL_HOST,
    API_MESSAGE_FILTER,
)

from ..version import __version__, VERSION_STRING

logger = logging.getLogger(__name__)


class SourceType(Enum):
    EXECUTOR = "executor"
    EXECUTOR_REQUIREMENTS = "executor-requirements"
    API = "api"
    API_REQUIREMENTS = "api-requirements"


async def upload_source(
    deployer_session,
    path: Path,
    source_type: SourceType,
    channel=FISHNET_DEPLOYMENT_CHANNEL,
) -> StoreMessage:
    logger.debug(f"Reading {source_type.name} file")
    with open(path, "rb") as fd:
        file_content = fd.read()
        storage_engine = (
            StorageEnum.ipfs
            if len(file_content) > 4 * 1024 * 1024
            else StorageEnum.storage
        )
        logger.debug(f"Uploading {source_type.name} sources to {storage_engine}")
        with deployer_session as session:
            user_code, status = session.create_store(
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


async def build_and_upload_requirements(
    requirements_path: Path,
    deployer_session: AuthenticatedAlephClient,
    source_type: SourceType,
    channel: str = FISHNET_DEPLOYMENT_CHANNEL,
) -> StoreMessage:
    if (
        source_type != SourceType.EXECUTOR_REQUIREMENTS
        and source_type != SourceType.EXECUTOR_REQUIREMENTS
    ):
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
    return await upload_source(
        deployer_session, path=squashfs_path, source_type=source_type, channel=channel
    )


async def deploy_executors(
    executor_path: Path,
    time_slices: List[int],
    deployer_session: AuthenticatedAlephClient,
    channel: str = FISHNET_DEPLOYMENT_CHANNEL,
    vcpus: int = 1,
    memory: int = 1024,
    timeout_seconds: int = 900,
    persistent: bool = False,
    volume_size_mib: int = 1024 * 10,
) -> List[ProgramMessage]:
    # Discover existing executor VMs
    executor_messages = await discover_executors(
        deployer_session.account.get_address(), deployer_session, channel
    )
    source_code_refs = set(
        [executor.content.code.ref for executor in executor_messages]
    )

    # Get latest version executors and source code
    latest_source = await fetch_latest_source(deployer_session, source_code_refs)
    latest_protocol_version = VersionInfo.parse(latest_source.content.protocol_version)
    latest_executors = [
        executor
        for executor in executor_messages
        if executor.content.code.ref == latest_source.item_hash
    ]

    # Create new source archive from local files and hash it
    path_object, encoding = create_archive(executor_path)

    # Check versions of latest source code and latest executors
    if latest_protocol_version >= __version__:
        raise Exception(
            "Latest protocol version is equal or greater than current version, aborting deployment: "
            + f"({latest_protocol_version} >= {__version__})"
        )
    # TODO: Move file hashing methods to aleph-sdk-python
    # TODO: Compare hash with all past versions' content.item_hashes
    # If any are equal, throw error because of repeated deployment

    # Upload the source code with new version
    user_code = await upload_source(
        deployer_session, path_object, source_type=SourceType.EXECUTOR
    )

    # Upload the requirements image
    requirements = await build_and_upload_requirements(
        path_object / "requirements.txt",
        deployer_session,
        source_type=SourceType.EXECUTOR_REQUIREMENTS,
    )

    vm_messages: List[ProgramMessage] = []
    for i, slice_end in enumerate(time_slices[1:]):
        slice_start = time_slices[i - 1]
        # parse slice_end and slice_start to datetime
        if slice_end == -1:
            slice_end = datetime.max.timestamp()
        slice_end = datetime.fromtimestamp(slice_end)
        slice_start = datetime.fromtimestamp(slice_start)
        slice_string = f"{slice_start.isoformat()}-{slice_end.isoformat()}"
        name = f"executor-v{VERSION_STRING}_{slice_string}"

        # Create immutable volume with python dependencies
        volumes = [
            ImmutableVolume(ref=requirements.item_hash).dict(),
            PersistentVolume(
                persistence="host", name=name, size_mib=volume_size_mib
            ).dict(),
        ]

        # Register the program
        # TODO: Update existing VMs (if mutable deployment)
        # TODO: Otherwise create new VMs
        with deployer_session:
            message, status = deployer_session.create_program(
                program_ref=user_code.item_hash,
                entrypoint="main:app",
                runtime=settings.DEFAULT_RUNTIME_ID,
                storage_engine=StorageEnum.storage,
                channel=channel,
                memory=memory,
                vcpus=vcpus,
                timeout_seconds=timeout_seconds,
                persistent=persistent,
                encoding=encoding,
                volumes=volumes,
                subscriptions=EXECUTOR_MESSAGE_FILTER,
                metadata={
                    "tags": ["fishnet", SourceType.EXECUTOR.name, VERSION_STRING],
                    "time_slice": slice_string,
                },
            )
        logger.debug("Upload finished")

        hash: str = message.item_hash
        hash_base32 = b32encode(b16decode(hash.upper())).strip(b"=").lower().decode()

        logger.info(
            f"Executor {name} deployed. \n\n"
            "Available on:\n"
            f"  {VM_URL_PATH.format(hash=hash)}\n"
            f"  {VM_URL_HOST.format(hash_base32=hash_base32)}\n"
            "Visualise on:\n  https://explorer.aleph.im/address/"
            f"{message.chain}/{message.sender}/message/PROGRAM/{hash}\n"
        )

        vm_messages.append(message)

    return vm_messages


async def deploy_api(
    api_path: Path,
    deployer_session: AuthenticatedAlephClient,
    executors: List[ProgramMessage],
    channel: str = FISHNET_DEPLOYMENT_CHANNEL,
    vcpus: int = 1,
    memory: int = 1024 * 4,
    timeout_seconds: int = 900,
    persistent: bool = False,
) -> ProgramMessage:
    # Discover existing executor VMs
    api_messages = await discover_apis(
        deployer_session.account.get_address(), deployer_session, channel
    )
    source_code_refs = set([api.content.code.ref for api in api_messages])

    latest_source = await fetch_latest_source(deployer_session, source_code_refs)
    latest_protocol_version = VersionInfo.parse(latest_source.content.protocol_version)
    latest_apis = [
        api for api in api_messages if api.content.code.ref == latest_source.item_hash
    ]

    # Create new source archive from local files and hash it
    path_object, encoding = create_archive(api_path)

    # Check versions of latest source code and latest apis
    if latest_protocol_version >= __version__:
        raise Exception(
            "Latest protocol version is equal or greater than current version, aborting deployment: "
            + f"({latest_protocol_version} >= {__version__})"
        )

    # Upload the source code with new version
    user_code = await upload_source(
        deployer_session, path=path_object, source_type=SourceType.API
    )
    # Upload the requirements
    requirements = await build_and_upload_requirements(
        requirements_path=api_path / "requirements.txt",
        deployer_session=deployer_session,
        source_type=SourceType.API_REQUIREMENTS,
    )

    # Create immutable volume with python dependencies
    volumes = [
        ImmutableVolume(ref=requirements.item_hash).dict(),
    ]

    name = f"api-v{VERSION_STRING}"

    # Register the program
    with deployer_session as session:
        message, status = session.create_program(
            program_ref=user_code.item_hash,
            entrypoint="main:app",
            runtime="latest",
            storage_engine=StorageEnum(user_code.content.item_type),
            channel=channel,
            memory=memory,
            vcpus=vcpus,
            timeout_seconds=timeout_seconds,
            persistent=persistent,
            encoding=encoding,
            volumes=volumes,
            subscriptions=API_MESSAGE_FILTER,
            metadata={
                "tags": ["fishnet", SourceType.API.name, VERSION_STRING],
                "executors": [executor.item_hash for executor in executors],
            },
        )

    logger.debug("Upload finished")

    hash: str = message.item_hash
    hash_base32 = b32encode(b16decode(hash.upper())).strip(b"=").lower().decode()

    logger.info(
        f"Fishnet API {name} deployed. \n\n"
        "Available on:\n"
        f"  {VM_URL_PATH.format(hash=hash)}\n"
        f"  {VM_URL_HOST.format(hash_base32=hash_base32)}\n"
        "Visualise on:\n  https://explorer.aleph.im/address/"
        f"{message.chain}/{message.sender}/message/PROGRAM/{hash}\n"
    )

    return message


async def fetch_latest_source(deployer_session, source_code_refs):
    # Get latest version executors and source code
    with deployer_session:
        source_messages = await deployer_session.get_messages(
            hashes=source_code_refs, message_type=MessageType.store
        )
    latest_source: Optional[StoreMessage] = None
    for source in source_messages.messages:
        source: StoreMessage
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
