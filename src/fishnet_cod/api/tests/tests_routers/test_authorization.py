from dataclasses import dataclass, asdict
from os import putenv
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from aleph.sdk.chains import ethereum
from aleph.sdk.exceptions import BadSignatureError
from fastapi import HTTPException
from starlette.testclient import TestClient
from ...security import AuthTokenManager, SupportedChains, NotAuthorizedError
from ...routers.authorization import refresh_token, router, solve_challenge

putenv("TEST_CHANNEL", "true")


@pytest.fixture
def ethereum_account() -> ethereum.ETHAccount:
    with NamedTemporaryFile(delete=False) as private_key_file:
        private_key_file.close()
        yield ethereum.get_fallback_account(path=Path("/var/folders/fd/np0pq8gx6y71chvf_fgvjd2h0000gn/T/tmpy1crw71q"))


@dataclass
class Message:
    chain: str
    sender: str
    type: str
    item_hash: str


@pytest.fixture
def client():
    return TestClient(router)


def test_create_challenge(client, ethereum_account):
    chain = SupportedChains.Ethereum.value
    pubkey = ethereum_account.get_address()

    AuthTokenManager.get_challenge(pubkey, chain)
    response = client.post(
        "/authorization/challenge",
        # "/challenge",
        params={"pubkey": pubkey, "chain": chain},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["pubkey"] == pubkey
    assert data["chain"] == chain
    assert "challenge" in data
    assert "valid_til" in data


# Isn't working right now
@pytest.mark.asyncio
async def test_solve_challenge(client, ethereum_account):
    chain = SupportedChains.Ethereum.value
    pubkey = ethereum_account.get_address()  # There's some kinda strange exception while comparing these adresses

    message = asdict(
        Message(
            "ETH",
            ethereum_account.get_address(),
            "POST",
            "SomeHash",
        )
    )
    await ethereum_account.sign_message(message)
    assert message["signature"]
    response = client.post(
        "/authorization/solve",
        params={
            "pubkey": pubkey,
            "chain": chain,
            "signature": message["signature"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["pubkey"] == pubkey
    assert data["chain"] == chain
    assert "token" in data
    assert "valid_til" in data

    wrong_signature = "0x" + "0" * 130
    with pytest.raises(BadSignatureError):
        AuthTokenManager.solve_challenge(pubkey, chain, wrong_signature)
        response = client.post("/solve", params={"pubkey": pubkey, "chain": chain, "signature": wrong_signature})
        assert response.status_code == 403
        assert HTTPException(403, "Challenge failed")


# Isn't working right now
@pytest.mark.asyncio
async def test_refresh_token(client, ethereum_account):
    chain = SupportedChains.Ethereum.value
    pubkey = ethereum_account.get_address()

    message = asdict(
        Message(
            "ETH",
            ethereum_account.get_address(),
            "POST",
            "SomeHash",
        )
    )
    await ethereum_account.sign_message(message)
    assert message["signature"]

    solve_response = solve_challenge(pubkey, chain, message["signature"])

    response = client.post(
        "/authorization/refresh",
        params={"token": solve_response.token},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["pubkey"] == pubkey
    assert data["chain"] == chain
    assert data["token"] == solve_response.token
    assert "valid_til" in data

    with pytest.raises(HTTPException):
        client.post(
            "/authorization/refresh",
            params={"token": "0x" + "0" * 130},
        )

    assert HTTPException(403, "Not authorized")


# Isn't working right now
@pytest.mark.asyncio
async def test_logout(client, ethereum_account):
    chain = SupportedChains.Ethereum.value
    pubkey = ethereum_account.get_address()

    message = asdict(
        Message(
            "ETH",
            ethereum_account.get_address(),
            "POST",
            "SomeHash",
        )
    )
    await ethereum_account.sign_message(message)
    assert message["signature"]

    solve_response = solve_challenge(pubkey, chain, message["signature"])
    refresh_response = refresh_token(solve_response.token)

    response = client.post(
        "/authorization/logout",
        params={"token": refresh_response.token},
    )

    assert response.status_code == 200


# Funktioniert noch nicht
def test_challenge_timeout(client):
    expired_token = ""  # Muss noch konfigurieret werden

    with pytest.raises(HTTPException):
        client.post(
            "/authorization/refresh",
            params={"token": expired_token},
        )

    assert HTTPException(403, "Token expired")
