import logging
from base64 import b16decode, b32encode
from pathlib import Path
from typing import List
from datetime import datetime

from aleph.sdk.conf import settings
from aleph_message.models.program import ImmutableVolume, PersistentVolume
from semver import VersionInfo

from aleph.sdk import AuthenticatedAlephClient
from aleph.sdk.types import StorageEnum
from aleph.sdk.utils import create_archive
from aleph_message.models import StoreMessage, ProgramMessage

from .discovery import discover_executors, discover_apis
from .sources import (
    fetch_latest_source,
    upload_source,
    build_and_upload_requirements,
    SourceType,
)
from ..constants import (
    FISHNET_DEPLOYMENT_CHANNEL,
    EXECUTOR_MESSAGE_FILTER,
    VM_URL_PATH,
    VM_URL_HOST,
    API_MESSAGE_FILTER,
)

from ..version import __version__, VERSION_STRING

logger = logging.getLogger(__name__)


async def deploy_executors(
    executor_path: Path,
    time_slices: List[int],
    requirements: StoreMessage,
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
                    "tags": ["fishnet_cod", SourceType.EXECUTOR.name, VERSION_STRING],
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
                "tags": ["fishnet_cod", SourceType.API.name, VERSION_STRING],
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
