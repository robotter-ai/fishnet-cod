import logging

import pandas as pd
from aars import AARS
from aleph.sdk.chains.sol import SOLAccount
from aleph.sdk.client import AuthenticatedAlephClient
from aleph.sdk.conf import settings as aleph_settings
from aleph.sdk.vm.cache import TestVmCache, VmCache

from .conf import settings

logging.basicConfig(level=logging.INFO)


async def initialize_aars():
    if str(settings.TEST_CACHE).lower() == "false":
        cache = VmCache()
    else:
        cache = TestVmCache()

    aleph_account = SOLAccount(bytes.fromhex(settings.MESSAGES_KEY)[0:32])
    logging.info(f"Using account {aleph_account.get_address()}")
    aleph_session = AuthenticatedAlephClient(aleph_account, aleph_settings.API_HOST)

    if str(settings.TEST_CHANNEL).lower() == "true":
        channel = "FISHNET_TEST_" + str(pd.to_datetime("now", utc=True))
    else:
        channel = settings.MESSAGE_CHANNEL

    aars = AARS(
        account=aleph_account, channel=channel, cache=cache, session=aleph_session
    )

    if aleph_account.get_address() == settings.MANAGER_PUBKEY:
        try:
            resp, status = await aleph_session.fetch_aggregate(
                "security", aleph_account.get_address()
            )
            existing_authorizations = resp.json().get("authorizations", [])
        except:
            existing_authorizations = []
        needed_authorizations = [
            {
                "address": settings.MANAGER_PUBKEY,
                "channels": [settings.MESSAGE_CHANNEL],
            }
        ]
        if not all(auth in existing_authorizations for auth in needed_authorizations):
            aggregate = {
                "authorizations": needed_authorizations,
            }
            await aleph_session.create_aggregate(
                "security", aggregate, aleph_account.get_address(), channel="security"
            )

    return aars
