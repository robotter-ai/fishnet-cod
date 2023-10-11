import asyncio
import os

import base58
import pytest
from aleph.sdk.chains.sol import SOLAccount, generate_key

from fastapi_walletauth.common import SupportedChains
from nacl.signing import SigningKey
from starlette.testclient import TestClient

from fishnet_cod.api.main import app


ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def event_loop():
    yield app.aars.session.http_session.loop
    asyncio.run(app.aars.session.http_session.close())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def account():
    return SOLAccount(generate_key())


@pytest.fixture(scope="module")
def big_csv():
    with open(os.path.join(ABSOLUTE_PATH, "Binance_SOLBUSD_d.csv"), "rb") as f:
        yield f.read()


def login_with_signature(client, account: SOLAccount):
    chain = SupportedChains.Solana.value
    key = SigningKey(account.private_key)
    address = account.get_address()

    response = client.post(
        "/authorization/challenge",
        params={"address": address, "chain": chain},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["address"] == address
    assert data["chain"] == chain
    assert "challenge" in data
    assert "valid_til" in data

    signature = base58.b58encode(key.sign(data["challenge"].encode()).signature).decode(
        "utf-8"
    )

    response = client.post(
        "/authorization/solve",
        params={
            "address": address,
            "chain": chain,
            "signature": signature,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["address"] == address
    assert data["chain"] == chain
    assert "token" in data
    assert "valid_til" in data
    token = data["token"]
    return token
