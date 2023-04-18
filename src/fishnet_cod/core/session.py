from os import getenv

from aars import AARS
from aleph.sdk import AuthenticatedAlephClient
from aleph.sdk.conf import settings
from aleph.sdk.chains.sol import get_fallback_account
from aleph.sdk.vm.cache import TestVmCache, VmCache
import pandas as pd

from .constants import FISHNET_MESSAGE_CHANNEL


def initialize_aars():
    test_cache_flag = getenv("TEST_CACHE")
    if test_cache_flag is not None and test_cache_flag.lower() == "true":
        cache = TestVmCache()
    else:
        cache = VmCache()

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

    print(channel, 'channellll')


    aars = AARS(
        account=aleph_account, channel=channel, cache=cache, session=aleph_session
    )

    return aars
