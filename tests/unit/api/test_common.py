import asyncio
import os
from pathlib import Path
from tempfile import SpooledTemporaryFile
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aars import AARS
from aleph.sdk.chains.sol import SOLAccount, generate_key
from aleph.sdk.vm.cache import LocalVmCache
from fastapi import UploadFile

from src.fishnet_cod.api.controllers import (
    load_data_df,
    update_timeseries,
)

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def event_loop():
    account = SOLAccount(generate_key())
    aars_client = AARS(
        account=account,
        channel="FISHNET_TEST_" + str(pd.to_datetime("now", utc=True)),
        cache=LocalVmCache(),
    )
    yield aars_client.session.http_session.loop
    asyncio.run(aars_client.session.http_session.close())


@pytest.fixture()
def csv_file():
    filepath = Path("Binance_SOLBUSD_d.csv")
    spooled_file = SpooledTemporaryFile(max_size=1000, mode="w+b")
    with open(ABSOLUTE_PATH / filepath, "rb") as file:
        spooled_file.write(file.read())
    spooled_file.seek(0)
    return UploadFile(file=spooled_file, filename="test.csv")


@pytest.fixture()
def user():
    user = MagicMock()
    user.address = SOLAccount(generate_key()).get_address()
    return user


def test_load_data_file(csv_file):
    df = load_data_df(csv_file)
    assert df.index


@pytest.mark.asyncio
async def test_create_timeseries(csv_file, user):
    df = load_data_df(csv_file)
    metadata = []
    timeseries = None
    timeseries = await update_timeseries(df, metadata, timeseries, user)
    assert timeseries
    for col in df.columns:
        assert col in [ts.name for ts in timeseries]
