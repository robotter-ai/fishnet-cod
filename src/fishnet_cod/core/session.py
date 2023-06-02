from os import getenv

import pandas as pd
from aars import AARS
from aleph.sdk.client import AuthenticatedAlephClient, AuthenticatedUserSessionSync
from aleph.sdk.chains.sol import get_fallback_account
from aleph.sdk.conf import settings
from aleph.sdk.vm.cache import TestVmCache, VmCache

from .constants import FISHNET_MESSAGE_CHANNEL, FISHNET_MANAGER_PUBKEYS


async def initialize_aars():
    test_cache_flag = getenv("TEST_CACHE")
    if test_cache_flag is not None and test_cache_flag.lower() == "false":
        cache = VmCache()
    else:
        cache = TestVmCache()

    test_channel_flag = getenv("TEST_CHANNEL")
    custom_channel = getenv("CUSTOM_CHANNEL")
    if custom_channel:
        channel = custom_channel
    elif test_channel_flag is not None and test_channel_flag.lower() == "true":
        channel = "FISHNET_TEST_" + str(pd.to_datetime("now", utc=True))
    else:
        channel = FISHNET_MESSAGE_CHANNEL
    aleph_account = get_fallback_account()
    aleph_session = AuthenticatedAlephClient(aleph_account, settings.API_HOST)

    print("Using channel: " + channel)

    aars = AARS(
        account=aleph_account, channel=channel, cache=cache, session=aleph_session
    )

    if aleph_account.get_address() in FISHNET_MANAGER_PUBKEYS:
        try:
            resp, status = await aleph_session.fetch_aggregate(
                "security", aleph_account.get_address()
            )
            existing_authorizations = resp.json().get("authorizations", [])
        except:
            existing_authorizations = []
        needed_authorizations = [
            {
                "address": address,
                "channels": FISHNET_MESSAGE_CHANNEL,
            }
            for address in FISHNET_MANAGER_PUBKEYS
        ]
        if not all(auth in existing_authorizations for auth in needed_authorizations):
            aggregate = {
                "authorizations": needed_authorizations,
            }
            resp, status = await aleph_session.create_aggregate(
                "security", aggregate, aleph_account.get_address(), channel="security"
            )
            print("Created security aggregate:")
            print(resp.json())

    return aars
