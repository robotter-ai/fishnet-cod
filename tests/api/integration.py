from typing import List

from aleph.sdk.chains.sol import generate_key, SOLAccount

from .conftest import login_with_signature
from fishnet_cod.api.api_model import (
    RequestDatasetPermissionsRequest,
    PutTimeseriesRequest,
    UploadDatasetRequest,
    UploadTimeseriesRequest, UploadDatasetTimeseriesRequest,
)
from fishnet_cod.core.model import PermissionStatus


def test_integration(client, big_csv):
    owner = SOLAccount(generate_key())
    owner_token = login_with_signature(client, owner)

    # prepare csv
    response = client.post(
        "/timeseries/csv",
        headers={"Authorization": f"Bearer {owner_token}"},
        files={"data_file": ("test.csv", big_csv, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json() is not None
    timeseries = response.json()

    # create dataset
    upload_dataset_req = UploadDatasetTimeseriesRequest(
        dataset=UploadDatasetRequest(
            name="Binance_SOLBUST_1d",
        ),
        timeseries=timeseries,
    )
    response = client.post(
        "/datasets/upload/timeseries",
        json=upload_dataset_req.dict(),
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()["dataset"]["item_hash"] is not None

    # update timeseries
    upload_timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            PutTimeseriesRequest(name="test", data=[[1.0, 2.0], [3.0, 4.0]])
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
