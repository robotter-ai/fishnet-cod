import asyncio
from typing import List

import base58
import pytest
from fastapi.testclient import TestClient
from aleph.sdk.chains.sol import generate_key, SOLAccount
from fastapi_walletauth.common import SupportedChains
from nacl.signing import SigningKey

from fishnet_cod.api.api_model import (
    RequestDatasetPermissionsRequest,
    TimeseriesItem,
    UploadDatasetRequest,
    UploadTimeseriesRequest,
)
from fishnet_cod.api.main import app
from fishnet_cod.core.model import PermissionStatus


@pytest.fixture(scope="session")
def event_loop():
    yield app.aars.session.http_session.loop
    asyncio.run(app.aars.session.http_session.close())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


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


def test_integration(client):
    owner = SOLAccount(generate_key())
    owner_token = login_with_signature(client, owner)
    upload_timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            TimeseriesItem(name="test", data=[[1.0, 2.0], [3.0, 4.0]])
        ]
    )
    req_body = upload_timeseries_req.dict()
    print(req_body)
    response = client.put(
        "/timeseries", json=req_body, headers={"Authorization": f"Bearer {owner_token}"}
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()[0]["item_hash"] is not None
    timeseries_id = response.json()[0]["item_hash"]
    # - Upload dataset
    upload_dataset_req = UploadDatasetRequest(
        name="test",
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put(
        "/datasets",
        json=upload_dataset_req.dict(),
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    dataset_id = response.json()["item_hash"]

    requestor = SOLAccount(generate_key())
    requestor_token = login_with_signature(client, requestor)
    # - Request permission
    request_permission_req = RequestDatasetPermissionsRequest(timeseriesIDs=[])
    response = client.put(
        f"/permissions/datasets/{dataset_id}/request",
        json=request_permission_req.dict(),
        headers={"Authorization": f"Bearer {requestor_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    permission = response.json()[0]
    assert permission["status"] == PermissionStatus.REQUESTED

    # - Get notifications
    response = client.get(
        f"/users/{owner.get_address()}/notifications",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    notifications = response.json()
    assert isinstance(notifications, List)
    assert len(notifications) == 1

    # - Approve permission
    permission_ids = [permission["item_hash"]]

    response = client.put(
        "/permissions/approve",
        json=permission_ids,
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    new_permission = response.json()["updatedPermissions"][0]
    assert new_permission["status"] == PermissionStatus.GRANTED
