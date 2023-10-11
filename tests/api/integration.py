from typing import List

import pytest
from aleph.sdk.chains.sol import generate_key, SOLAccount

from .conftest import login_with_signature
from fishnet_cod.api.api_model import (
    RequestDatasetPermissionsRequest,
    PutTimeseriesRequest,
    UploadDatasetRequest,
    UploadTimeseriesRequest, UploadDatasetTimeseriesRequest, PutViewRequest,
)
from fishnet_cod.core.model import PermissionStatus, Dataset, Granularity


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
            name="test",
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
    dataset = response.json()["dataset"]
    sol_timeseries = response.json()["timeseries"]

    # add new timeseries
    upload_timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            PutTimeseriesRequest(name="test", data=[[1.0, 2.0], [3.0, 4.0]])
        ]
    )
    response = client.put(
        "/timeseries", json=upload_timeseries_req.dict(), headers={"Authorization": f"Bearer {owner_token}"}
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()[0]["item_hash"] is not None
    assert response.json()[0]["max"] == 4.0
    timeseries_id = response.json()[0]["item_hash"]

    # update said timeseries
    upload_timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            PutTimeseriesRequest(item_hash=timeseries_id, name="test", data=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ]
    )
    response = client.put(
        "/timeseries", json=upload_timeseries_req.dict(), headers={"Authorization": f"Bearer {owner_token}"}
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()[0]["item_hash"] == timeseries_id
    assert response.json()[0]["max"] == 6.0

    # add timeseries to dataset
    upload_dataset_req = UploadDatasetRequest(
        item_hash=dataset["item_hash"],
        name="Binance_SOLBUST_1d",
        ownsAllTimeseries=True,
        timeseriesIDs=dataset["timeseriesIDs"] + [timeseries_id],
    )
    response = client.put(
        "/datasets",
        json=upload_dataset_req.dict(),
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()["item_hash"] == dataset["item_hash"]
    assert timeseries_id in response.json()["timeseriesIDs"]
    dataset_id = dataset["item_hash"]

    requestor = SOLAccount(generate_key())
    requestor_token = login_with_signature(client, requestor)

    high_timeseries = list(filter(lambda ts: ts["name"] == "High", sol_timeseries))[0]
    low_timeseries = list(filter(lambda ts: ts["name"] == "Low", sol_timeseries))[0]
    # generate view
    generate_view_req = [
        PutViewRequest(
            timeseriesIDs=[low_timeseries["item_hash"], high_timeseries["item_hash"]],
            granularity=Granularity.YEAR.value,
        ).dict()
    ]
    print(generate_view_req)
    response = client.put(
        f"/datasets/{dataset_id}/views",
        json=generate_view_req,
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()["views"][0]["item_hash"] is not None
    assert response.json()["views"][0]["item_hash"] in response.json()["dataset"]["viewIDs"]

    # request permission
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

    # get notifications
    response = client.get(
        f"/users/{owner.get_address()}/notifications",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    print(response.json())
    assert response.status_code == 200
    notifications = response.json()
    assert isinstance(notifications, List)
    assert len(notifications) == 1

    # approve permission
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

    # sleep a second
    import time

    time.sleep(1)

    # get datasets and view as requestor
    response = client.get(
        f"/datasets",
        headers={"Authorization": f"Bearer {requestor_token}"},
        params={"view_as": requestor.get_address()},
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json()[0]["permission_status"] == "GRANTED"
