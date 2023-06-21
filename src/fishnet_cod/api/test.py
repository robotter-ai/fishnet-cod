import asyncio
import pytest
from fastapi.testclient import TestClient

from .api_model import *
from .main import app


@pytest.fixture(scope="session")
def event_loop():
    yield app.aars.session.http_session.loop
    asyncio.run(app.aars.session.http_session.close())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_full_request_execution_flow_with_own_dataset(client):
    with client:
        upload_dataset_req = UploadDatasetTimeseriesRequest(
            dataset=UploadDatasetRequest(
                name="test",
                owner="test",
                ownsAllTimeseries=True,
                timeseriesIDs=[],
            ),
            timeseries=[
                TimeseriesItem(name="test", owner="test", data=[[1.0, 2.0], [3.0, 4.0]])
            ]
        )
        response = client.post("/datasets/upload/timeseries", json=upload_dataset_req.dict())
        assert response.status_code == 200
        print(response.json())
        assert response.json()["dataset"]["item_hash"] is not None
        dataset_id = response.json()["dataset"]["item_hash"]

        upload_algorithm_req = UploadAlgorithmRequest(
            name="test", desc="test", owner="test", code="test"
        )
        response = client.put("/algorithms", json=upload_algorithm_req.dict())
        assert response.status_code == 200
        assert response.json()["item_hash"] is not None
        algorithm_id = response.json()["item_hash"]

        request_execution_req = RequestExecutionRequest(
            algorithmID=algorithm_id, datasetID=dataset_id, owner="test"
        )
        response = client.post("/executions", json=request_execution_req.dict())
        assert response.status_code == 200
        assert response.json()["execution"]["status"] == ExecutionStatus.PENDING


def test_requests_approval_deny(client):
    authorizer_address = "Approve_test_authorizer"
    timeseries_item = TimeseriesItem(
        name="Approve_test",
        owner=authorizer_address,
        available=True,
        data=[[1.0, 2.0], [3.0, 4.0]],
    )
    response = client.put("/timeseries", json=timeseries_item.dict())

    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    timeseries_id = response.json()["item_hash"]

    upload_dataset_req = UploadDatasetRequest(
        name="Approve_test",
        owner="test",
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put("/datasets/upload", json=upload_dataset_req.dict())
    assert response.status_code == 200
    dataset_id = response.json()["item_hash"]
    assert dataset_id is not None

    requestor_address = "Approve_test_requestor"
    upload_algorithm_req = UploadAlgorithmRequest(
        name="Approve_test", desc="Approve_test", owner=requestor_address, code="test"
    )
    response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
    assert response.status_code == 200
    algorithm_id = response.json()["item_hash"]
    assert algorithm_id is not None

    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id, datasetID=dataset_id, owner=requestor_address
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == ExecutionStatus.REQUESTED
    permission = response.json()["execution"]["permissionRequests"][0]
    assert permission.status == PermissionStatus.REQUESTED
    permission_ids = [permission.item_hash]

    response = client.put(
        "/permissions/approve", params={"permission_hashes": permission_ids}
    )
    assert response.status_code == 200
    new_permission = response.json()[0]
    assert new_permission.status == PermissionStatus.GRANTED

    # TODO: Check execution is now pending


def test_get_algorithm(client):
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

    upload_dataset_req = UploadDatasetRequest(
        name="test",
        owner="test",
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put("/datasets/upload", json=upload_dataset_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    upload_algorithm_req = UploadAlgorithmRequest(
        name="test", desc="test", owner="test", code="test"
    )
    response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None

    # Testing the get_algorithm endpoint
    algo_response = client.get("/algorithms")
    algo_json = algo_response.json()
    assert algo_response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert isinstance(algo_json, list)
    # Clearing up the all records
    clear_response = client.delete("/clear/records")
    assert clear_response.status_code == 200


def test_incoming_permission(client):
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
    # - Upload dataset
    upload_dataset_req = UploadDatasetRequest(
        name="test",
        owner="test",
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put("/datasets/upload", json=upload_dataset_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    dataset_id = response.json()["item_hash"]

    # - Upload algorithm
    requestor_address = "Approve_test_requestor"
    upload_algorithm_req = UploadAlgorithmRequest(
        name="Approve_test",
        desc="Approve_test",
        owner=requestor_address,
        code="test",
    )
    response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    algorithm_id = response.json()["item_hash"]

    # - Approve execution
    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id,
        datasetID=dataset_id,
        owner=requestor_address,
        status=ExecutionStatus.REQUESTED,
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == ExecutionStatus.REQUESTED

    # - Deny execution
    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id,
        datasetID=dataset_id,
        owner=requestor_address,
        status=ExecutionStatus.PENDING,
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    # - Get execution result & status
    assert response.json()["execution"]["status"] == ExecutionStatus.PENDING

    # - Grant permission
    permission = response.json()["execution"]["permissionRequests"][0]
    assert permission.status == PermissionStatus.REQUESTED
    permission_ids = [permission.item_hash]

    response = client.put(
        "/permissions/approve", params={"permission_hashes": permission_ids}
    )
    assert response.status_code == 200
    new_permission = response.json()[0]
    assert new_permission.status == PermissionStatus.GRANTED

    user_id = "Authorizer_001"
    page = 1
    page_size = 20

    expected_permissions = PostPermission(
        timeseriesID=timeseries_id,
        algorithmID=algorithm_id,
        authorizer=user_id,
        status=PermissionStatus.GRANTED,
        executionCount=4,
        maxExecutionCount=44,
        requestor="test_requestor",
    )
    response = client.get(
        f"/user/{user_id}/permissions/incoming?page={page}&page_size={page_size}"
    )
    assert response.status_code == 200
    permission_list = response.json()
    assert response.headers["content-type"] == "application/json"
    assert isinstance(permission_list, list)
    assert all(isinstance(permission, dict) for permission in permission_list)
    for permission in permission_list:
        assert permission["timeseriesID"] == expected_permissions.timeseriesID
        assert permission["algorithmID"] == expected_permissions.algorithmID
        assert permission["authorizer"] == expected_permissions.authorizer
        assert permission["status"] == expected_permissions.status
        assert permission["executionCount"] == expected_permissions.executionCount
        assert (
            permission["maxExecutionCount"]
            == expected_permissions.maxExecutionCount
        )
        assert permission["requestor"] == expected_permissions.requestor

    # - At the end, delete all data
    response = client.delete("/clear/records")
    assert response.status_code == 200


def test_outgoing_permission(client):
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
    # - Upload dataset
    upload_dataset_req = UploadDatasetRequest(
        name="test",
        owner="test",
        ownsAllTimeseries=True,
        timeseriesIDs=[timeseries_id],
    )
    response = client.put("/datasets/upload", json=upload_dataset_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    dataset_id = response.json()["item_hash"]

    # - Upload algorithm
    requestor_address = "Approve_test_requestor"
    upload_algorithm_req = UploadAlgorithmRequest(
        name="Approve_test",
        desc="Approve_test",
        owner=requestor_address,
        code="test",
    )
    response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
    assert response.status_code == 200
    assert response.json()["item_hash"] is not None
    algorithm_id = response.json()["item_hash"]

    # - Approve execution
    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id,
        datasetID=dataset_id,
        owner=requestor_address,
        status=ExecutionStatus.REQUESTED,
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == ExecutionStatus.REQUESTED

    # - Deny execution
    request_execution_req = RequestExecutionRequest(
        algorithmID=algorithm_id,
        datasetID=dataset_id,
        owner=requestor_address,
        status=ExecutionStatus.PENDING,
    )
    response = client.post("/executions/request", json=request_execution_req.dict())
    assert response.status_code == 200
    # - Get execution result & status
    assert response.json()["execution"]["status"] == ExecutionStatus.PENDING

    # - Grant permission
    permission = response.json()["execution"]["permissionRequests"][0]
    assert permission.status == PermissionStatus.REQUESTED
    permission_ids = [permission.item_hash]

    response = client.put(
        "/permissions/approve", params={"permission_hashes": permission_ids}
    )
    assert response.status_code == 200
    new_permission = response.json()[0]
    assert new_permission.status == PermissionStatus.GRANTED

    user_id = "requestor_001"
    page = 1
    page_size = 20

    expected_permissions = PostPermission(
        timeseriesID=timeseries_id,
        algorithmID=algorithm_id,
        authorizer="test_authorizer",
        status=PermissionStatus.GRANTED,
        executionCount=4,
        maxExecutionCount=44,
        requestor=user_id,
    )
    response = client.get(
        f"/user/{user_id}/permissions/outgoing?page={page}&page_size={page_size}"
    )
    assert response.status_code == 200
    permission_list = response.json()
    assert response.headers["content-type"] == "application/json"
    assert isinstance(permission_list, list)
    assert all(isinstance(permission, dict) for permission in permission_list)
    for permission in permission_list:
        assert permission["timeseriesID"] == expected_permissions.timeseriesID
        assert permission["algorithmID"] == expected_permissions.algorithmID
        assert permission["authorizer"] == expected_permissions.authorizer
        assert permission["status"] == expected_permissions.status
        assert permission["executionCount"] == expected_permissions.executionCount
        assert (
            permission["maxExecutionCount"]
            == expected_permissions.maxExecutionCount
        )
        assert permission["requestor"] == expected_permissions.requestor

    # - At the end, delete all data
    response = client.delete("/clear/records")
    assert response.status_code == 200
