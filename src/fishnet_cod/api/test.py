from os import putenv

from fastapi.testclient import TestClient

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
# ALWAYS use "with client:" to ensure the client is correctly initialized and closed


def test_full_request_execution_flow_with_own_dataset():
    with client:
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
            name="test",
            owner="test",
            ownsAllTimeseries=True,
            timeseriesIDs=[timeseries_id],
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


def test_get_algorithm():
    with client:
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
            name="test",
            owner="test",
            ownsAllTimeseries=True,
            timeseriesIDs=[timeseries_id],
        )
        response = client.put("/datasets/upload", json=upload_dataset_req.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None
        upload_algorithm_req = UploadAlgorithmRequest(
            name="test", desc="test", owner="test", code="test"
        )
        response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None

        # Testing the get_algorithm endpoint
        algo_response = client.get("/algorithms")
        algo_json = algo_response.json()
        assert algo_response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert isinstance(algo_json, list)
        # Clearing up the all records
        clear_response = client.delete("/clear/records")
        assert clear_response.status == 200


def test_incoming_permission():
    # Uploadtimeseries
    with client:
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
        # - Upload dataset
        upload_dataset_req = UploadDatasetRequest(
            name="test",
            owner="test",
            ownsAllTimeseries=True,
            timeseriesIDs=[timeseries_id],
        )
        response = client.put("/datasets/upload", json=upload_dataset_req.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None
        dataset_id = response.json()["id_hash"]

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
        assert response.json()["id_hash"] is not None
        algorithm_id = response.json()["id_hash"]

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
        permission_ids = [permission.id_hash]

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


def test_outgoing_permission():
    with client:
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
        # - Upload dataset
        upload_dataset_req = UploadDatasetRequest(
            name="test",
            owner="test",
            ownsAllTimeseries=True,
            timeseriesIDs=[timeseries_id],
        )
        response = client.put("/datasets/upload", json=upload_dataset_req.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None
        dataset_id = response.json()["id_hash"]

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
        assert response.json()["id_hash"] is not None
        algorithm_id = response.json()["id_hash"]

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
        permission_ids = [permission.id_hash]

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


def test_get_dataset_permission():
    with client:
        # - Reindex
        # - Upload timeseries
        upload_timeseries_req_1 = UploadTimeseriesRequest(
            timeseries=[
                TimeseriesItem(
                    name="dataset_permission_timeseries001",
                    owner="Dataset_permission_001",
                    data=[[1.0, 2.0], [3.0, 4.0]],
                )
            ]
        )
        upload_timeseries_req_2 = UploadTimeseriesRequest(
            timeseries=[
                TimeseriesItem(
                    name="dataset_permission_timeseries002",
                    owner="Dataset_permission_002",
                    data=[[1.0, 2.0], [3.0, 4.0]],
                )
            ]
        )
        response = client.put("/timeseries/upload", json=upload_timeseries_req_1.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None
        timeseries_id_1 = response.json()["id_hash"]

        response = client.put("/timeseries/upload", json=upload_timeseries_req_2.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None
        timeseries_id_2 = response.json()["id_hash"]

        upload_dataset_req = UploadDatasetRequest(
            name="dataset_permission_test_upload_Request",
            owner="dataset_permission_test_owner",
            ownsAllTimeseries=True,
            timeseriesIDs=[timeseries_id_1, timeseries_id_2],
        )
        # Upload dataset
        response = client.put("/datasets/upload", json=upload_dataset_req.dict())
        assert response.status_code == 200
        dataset_id = response.json()["id_hash"]
        assert dataset_id is not None

        requestor_address = "Approve_test_requestor_001"
        upload_algorithm_req = UploadAlgorithmRequest(
            name="Approve_test_001",
            desc="Approve_test_001 from dataset permission_001,002",
            owner=requestor_address,
            code="test",
        )
        # - Upload algorithm
        response = client.put("/algorithms/upload", json=upload_algorithm_req.dict())
        assert response.status_code == 200
        algorithm_id = response.json()["id_hash"]
        assert algorithm_id is not None
        # Request execution
        request_execution_req = RequestExecutionRequest(
            algorithmID=algorithm_id,
            datasetID=dataset_id,
            owner=requestor_address,
            status=ExecutionStatus.REQUESTED,
        )
        response = client.post("/executions/request", json=request_execution_req.dict())
        assert response.status_code == 200
        # - Get execution result & status
        assert response.json()["execution"]["status"] == ExecutionStatus.REQUESTED
        permission = response.json()["execution"]["permissionRequests"][0]
        assert permission.status == PermissionStatus.REQUESTED
        requested_permission_ids = [permission.id_hash]

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
        pending_permission_ids = [permission.id_hash]

        response = client.put(
            "/permissions/approve",
            params={"permission_hashes": requested_permission_ids},
        )
        assert response.status_code == 200
        new_permission_granted = response.json()[0]
        assert new_permission_granted.status == PermissionStatus.GRANTED

        response = client.put(
            "/permissions/deny", params={"permission_hashes": pending_permission_ids}
        )
        assert response.status_code == 200
        new_permission_denied = response.json()[0]
        assert new_permission_denied.status == PermissionStatus.DENIED

        response = client.get(f"/datasets/{dataset_id}/permissions")
        assert response.status_code == 200
        returned_permissions = response.json()
        assert len(returned_permissions) > 0
        assert new_permission_granted.dict() in returned_permissions
        assert new_permission_denied.dict() in returned_permissions
        # - At the end, delete all data
        response = client.delete("/clear/records")
        assert response.status_code == 200
        assert response.json() == []


def test_get_notification():
    with client:
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
        # - Upload dataset
        upload_dataset_req = UploadDatasetRequest(
            name="test",
            owner="test",
            ownsAllTimeseries=True,
            timeseriesIDs=[timeseries_id],
        )
        response = client.put("/datasets/upload", json=upload_dataset_req.dict())
        assert response.status_code == 200
        assert response.json()["id_hash"] is not None
        dataset_id = response.json()["id_hash"]

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
        assert response.json()["id_hash"] is not None
        algorithm_id = response.json()["id_hash"]

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
        permission_ids = [permission.id_hash]

        response = client.put(
            "/permissions/approve", params={"permission_hashes": permission_ids}
        )
        assert response.status_code == 200
        new_permission = response.json()[0]
        assert new_permission.status == PermissionStatus.GRANTED

        user_id = "authorizer_001"

        expected_permissions = PostPermission(
            timeseriesID=timeseries_id,
            algorithmID=algorithm_id,
            authorizer=user_id,
            status=PermissionStatus.GRANTED,
            executionCount=4,
            maxExecutionCount=44,
            requestor="requestor",
        )

        response = client.post(
            "/authorizer/post/permission", json=expected_permissions.dict()
        )
        assert response.status_code == 200
        user_id = response.json()[0]["authorizer"]

        response = client.get(f"/user/{user_id}/notifications")
        assert response.status_code == 200
        notifications = response.json()
        assert isinstance(notifications, List)
        assert all(
            isinstance(notification, Notification) for notification in notifications
        )
