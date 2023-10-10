import asyncio
from typing import List

import pytest
from fastapi.testclient import TestClient

from ...core.model import PermissionStatus
from ..api_model import (
    RequestDatasetPermissionsRequest,
    TimeseriesItem,
    UploadDatasetRequest,
    UploadTimeseriesRequest,
)
from ..main import app


@pytest.fixture(scope="session")
def event_loop():
    yield app.aars.session.http_session.loop
    asyncio.run(app.aars.session.http_session.close())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


@pytest.skip
def test_get_notification(client):
    owner_address = "test_get_notification_owner"
    upload_timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            TimeseriesItem(name="test", owner="test", data=[[1.0, 2.0], [3.0, 4.0]])
        ]
    )
    req_body = upload_timeseries_req.dict()
    response = client.put("/timeseries", json=req_body)
    assert response.status_code == 200
    assert response.json()[0]["item_hash"] is not None
    timeseries_id = response.json()[0]["item_hash"]
    # Upload dataset
    upload_dataset_req = UploadDatasetRequest(
        name="test",
        owner=owner_address,
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put("/datasets", json=upload_dataset_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    dataset_id = response.json()["item_hash"]

    # Request permission
    request_permission_req = RequestDatasetPermissionsRequest(
        timeseriesIDs=[timeseries_id]
    )
    permission = response.json()["permissionRequests"][0]
    assert permission["status"] == PermissionStatus.REQUESTED

    # - Approve permission
    permission_ids = [permission["item_hash"]]

    response = client.put("/permissions/approve", json=permission_ids)
    assert response.status_code == 200
    new_permission = response.json()["updatedPermissions"][0]
    assert new_permission["status"] == PermissionStatus.GRANTED
    triggered_executions = response.json()["triggeredExecutions"]
    assert len(triggered_executions) == 1

    response = client.get(f"/users/{requestor_address}/notifications")
    assert response.status_code == 200
    notifications = response.json()
    assert isinstance(notifications, List)
    assert len(notifications) == 1
