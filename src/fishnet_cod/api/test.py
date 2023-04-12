from fastapi.testclient import TestClient

from os import putenv

putenv("TEST_CHANNEL", "true")

from .api_model import *
from .main import app
from ..core.model import ExecutionStatus, PermissionStatus

client = TestClient(app)


# TODO: Simulate a complete API lifecycle
# For example:
# - Reindex
# - Upload timeseries
# - Upload dataset
# - Upload algorithm
# - Request execution
# - Approve execution
# - Deny execution
# - Get execution result & status
# - Grant permission
# - Test all endpoints
# - At the end, delete all data
# IF YOU RELY ON DATA FROM A PREVIOUS TEST, THEN FUSE THE TESTS TOGETHER
# TEST ALL THE ENDPOINTS

def test_full_request_execution_flow_with_own_dataset():
    upload_timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            TimeseriesItem(name="test", owner="test", data=[[1.0, 2.0], [3.0, 4.0]])
        ]
    )
    req_body = upload_timeseries_req.dict()
    response = client.put("/timeseries/upload", json=req_body)
    assert response.status_code == 200
    assert response.json()[0]["id_hash"] is not None
    timeseries_id = response.json()[0]["id_hash"]

    upload_dataset_req = UploadDatasetRequest(
        name="test", owner="test", ownsAllTimeseries=True, timeseriesIDs=[timeseries_id]
    )
    response = client.put("/datasets/upload", json=upload_dataset_req.dict())
    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    dataset_id = response.json()["id_hash"]

    upload_algorithm_req = UploadAlgorithmRequest(
        name="test", desc="test", owner="test", code="test"
    )
    response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    algorithm_id = response.json()["id_hash"]

    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id, datasetID=dataset_id, owner="test"
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == ExecutionStatus.PENDING


def test_requests_approval_deny():
    authorizer_address = "Approve_test_authorizer"
    timeseries_item = TimeseriesItem(
        name="Approve_test",
        owner=authorizer_address,
        available=True,
        data=[[1.0, 2.0], [3.0, 4.0]],
    )
    response = client.put("/timeseries/upload", json=timeseries_item.dict())

    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    timeseries_id = response.json()["id_hash"]

    upload_dataset_req = UploadDatasetRequest(
        name="Approve_test",
        owner="test",
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put("/datasets/upload", json=upload_dataset_req.dict())
    assert response.status_code == 200
    dataset_id = response.json()["id_hash"]
    assert dataset_id is not None

    requestor_address = "Approve_test_requestor"
    upload_algorithm_req = UploadAlgorithmRequest(
        name="Approve_test", desc="Approve_test", owner=requestor_address, code="test"
    )
    response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
    assert response.status_code == 200
    algorithm_id = response.json()["id_hash"]
    assert algorithm_id is not None

    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id, datasetID=dataset_id, owner=requestor_address
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == ExecutionStatus.REQUESTED
    permission = response.json()["execution"]["permissionRequests"][0]
    assert permission.status == PermissionStatus.REQUESTED
    permission_ids = [permission.id_hash]

    response = client.put(
        "/permissions/approve", params={"permission_hashes": permission_ids}
    )
    assert response.status_code == 200
    new_permission = response.json()[0]
    assert new_permission.status == PermissionStatus.GRANTED

    # TODO: Check execution is now pending


# Each test does the following:
# - Create some data
# - Check that the data is created correctly
# - Maybe have some additional checks on other endpoints (like permissions when requested on an execution)
# - (if possible) Delete the data

def test_dataset():
    page = 1
    page_size = 1
    view_as = "Owner_of_TimeseriesId"
    by = "Ds_owner004"
    req = {"view_as": view_as, "by": by}
    response = client.get("/datasets", params=req)
    assert response.status_code == 200


def test_get_algorithm():
    id = "60b5e790149d12d0f4b1b7af0c27f3eeb9fa0d56edb7bd56832ef536e36c6115"
    name = "Al004"
    by = "Owner for Al004"
    req = {"id": id, "name": name, "by": by}
    response = client.get("/algorithms", params=req)
    assert response.status_code == 200

    view_as = "Owner_of_TimeseriesId"
    by = "Ds_owner004"
    req = {"view_as": view_as, "by": by}
    response = client.get("/datasets", params=req)
    returned_Algorithms = response.json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert isinstance(returned_Algorithms, list)


def test_incoming_permission():
    user_id = "authorizer001"
    page = 1
    page_size = 20
    expected_permissions = [
        Permission(
            timeseriesID="05436d1b8f6c627de504ff070e50ccc2c6c163340b823ca58a2a4fdf682f8584",
            algorithmID="9eea4f31386ec6d106fe23dbec1ecde58145189cdc7937bf58839b049843de51",
            authorizer=user_id,
            status="GRANTED",
            executionCount=10,
            maxExecutionCount=44,
            requestor="requestor003",
        ),
    ]

    response = client.get(f"/user/{user_id}/permissions/incoming?page={page}&page_size={page_size}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    permission_list = response.json()
    assert len(permission_list) > 0
    assert isinstance(permission_list, list)
    assert all(isinstance(permission, dict) for permission in permission_list)
    for i, permission in enumerate(permission_list):
        assert permission["timeseriesID"] == expected_permissions[i].timeseriesID
        assert permission["algorithmID"] == expected_permissions[i].algorithmID
        assert permission["authorizer"] == expected_permissions[i].authorizer
        assert permission["status"] == expected_permissions[i].status.value
        assert permission["executionCount"] == expected_permissions[i].executionCount
        assert permission["maxExecutionCount"] == expected_permissions[i].maxExecutionCount
        assert permission["requestor"] == expected_permissions[i].requestor


def test_outgoing_permission():
    user_id = "authorizer001"
    page = 1
    page_size = 20
    expected_permissions = [
        Permission(
            timeseriesID="2115cb403b1d83e4def86cd09be2bda067284245d28e48bb2dbf20600ce1a604",
            algorithmID="3a417f6b07ef1a04585a34a8152ece264b0bf88a20ce5c885ad144eaf4ce5cda",
            authorizer=user_id,
            status="REQUESTED",
            executionCount=10,
            maxExecutionCount=44,
            requestor="requestor002",
        ),
    ]

    response = client.get(f"/user/{user_id}/permissions/incoming?page={page}&page_size={page_size}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    permission_list = response.json()
    assert isinstance(permission_list, list)
    assert all(isinstance(permission, dict) for permission in permission_list)
    for i, permission in enumerate(permission_list):
        assert permission["timeseriesID"] == expected_permissions[i].timeseriesID
        assert permission["algorithmID"] == expected_permissions[i].algorithmID
        assert permission["authorizer"] == expected_permissions[i].authorizer
        assert permission["status"] == expected_permissions[i].status.value
        assert permission["executionCount"] == expected_permissions[i].executionCount
        assert permission["maxExecutionCount"] == expected_permissions[i].maxExecutionCount
        assert permission["requestor"] == expected_permissions[i].requestor


def test_get_dataset_permission():
    dataset_id = "8a716080c925619350ed153ee41d4a5c369fd6a61cb6a6fa0d204b0dfd89bb22"
    dataset = Dataset(name="king",
                      owner="Kingsley",
                      ownsAllTimeseries=True,
                      timeseriesIDs=["b29cb38fbf74a93eb33eef6741873ccbcfeff10b617356e0b2aec80b9e9e3755",
                                     "d90963ee301338d448288a472deb51f4bbb6d5d7484069772a0e2e52ce815fd2",
                                     "65ce1aad42a81e70f20725d090a90fd626aae977b5791d1b0f38839ce8eee3d6",
                                     "db5fdcc22feef7085ab7c251d7cc546fc69cf01736cc51e1387df7d0579aa381",
                                     "67f1e2575396869dd0a7862bd7988704a282798afbc61e29bb8282447f3e43bb",
                                     "b87d0c547a228f4695497d98c918e8bead1978c8c47c1522d82ca6dc8ee3cf30",
                                     "800cfde019f65a646ff4ab7859fe5bfca919050739d341dcb2b9065880341d4c",
                                     "b64b5eeef99721d4d9b51baeeb4eb9c4874900bc4810336008dca601c40bc390",
                                     "0bcb74f73e9a0c3d2b398a1236fc7b89fcf394506ee40a3981fb712668381dd5",
                                     "d463d050981ae6162fd6a7750ba48adb3f34acff460091f5c6234b9c8a4c9d26",
                                     "e6fe4f7745d470964718aa84e082400a8b76af2a83afe988fb586f34a70d64c6",
                                     ])
    permission1 = Permission(timeseriesID="b29cb38fbf74a93eb33eef6741873ccbcfeff10b617356e0b2aec80b9e9e3755",
                             authorizer="authorizer001",
                             status=PermissionStatus.REQUESTED,
                             executionCount=10,
                             requestor="requestor001")
    permission2 = Permission(timeseriesID="0bcb74f73e9a0c3d2b398a1236fc7b89fcf394506ee40a3981fb712668381dd5",
                             authorizer="authorizer003",
                             status=PermissionStatus.REQUESTED,
                             executionCount=10,
                             requestor="requestor001")

    response = client.get(f"/datasets/{dataset_id}/permissions")
    assert response.status_code == 200
    returned_permissions = response.json()
    assert len(returned_permissions) > 0
    assert permission1.dict() in returned_permissions
    assert permission2.dict() in returned_permissions


def test_get_notification():
    user_id = "authorizer001"
    response = client.get(f"/user/{user_id}/notifications")
    assert response.status_code == 200
    notifications = response.json()
    assert isinstance(notifications, List)
    assert all(isinstance(notification, Notification) for notification in notifications)
