from fastapi.testclient import TestClient
from ..core.model import ExecutionStatus, PermissionStatus

from .main import app
from .api_model import *

client = TestClient(app)


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
    response = client.post("/Timeseries", json=timeseries_item.dict())
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


def test_execution_dataset():
    dataset_Id = "5fecb379a0efdbd88a3d06f9b587dd3161dc8da6a8497f280f86bb3aa05eea94"
    response = client.get(f"/executions/{dataset_Id}")
    assert response.json()
    data = response.json()
    print("data", data)


def test_dataset():
    page = 1
    page_size = 1
    response = client.get("/datasets")
