import asyncio
import time

from aleph.sdk.client import AlephClient
from aleph.sdk.conf import settings
from fastapi.testclient import TestClient

from ..core.model import ExecutionStatus
from ..api.api_model import (
    UploadTimeseriesRequest,
    TimeseriesItem,
    UploadDatasetRequest,
    UploadAlgorithmRequest,
    RequestExecutionRequest,
)
from ..api.main import app

from main import handle_execution

client = TestClient(app)
aleph_client = AlephClient(settings.API_HOST)


def test_execution():
    req: UploadTimeseriesRequest = UploadTimeseriesRequest(
        timeseries=[
            TimeseriesItem(
                name="test_execution_timeseries1",
                owner="executooor",
                data=[(i, 1) for i in range(100)],
            ),
            TimeseriesItem(
                name="test_execution_timeseries2",
                owner="executooor",
                data=[(i, 10) for i in range(100)],
            ),
        ]
    )
    req_body = req.dict()
    response = client.put("/timeseries/upload", json=req_body)
    assert response.status_code == 200
    assert response.json()[0]["id_hash"] is not None
    timeseries_ids = [ts["id_hash"] for ts in response.json()]

    req: UploadDatasetRequest = UploadDatasetRequest(
        name="test_execution_dataset",
        owner="executooor",
        ownsAllTimeseries=True,
        timeseriesIDs=timeseries_ids,
    )
    response = client.put("/datasets/upload", json=req.dict())
    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    dataset_id = response.json()["id_hash"]

    req: UploadAlgorithmRequest = UploadAlgorithmRequest(
        name="test_execution_algorithm",
        desc="sums all columns",
        owner="executooor",
        code="""
def run(df: pd.DataFrame):
    return df.sum(axis=0)
""",
    )
    response = client.put("/algorithms/upload", json=req.dict())
    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    algorithm_id = response.json()["id_hash"]

    req: RequestExecutionRequest = RequestExecutionRequest(
        algorithmID=algorithm_id, datasetID=dataset_id, owner="executooor"
    )
    response = client.post("/executions/request", json=req.dict())
    assert response.status_code == 200
    assert response.json()["execution"]["status"] == ExecutionStatus.PENDING

    time.sleep(3)
    execution_post = asyncio.run(
        aleph_client.get_messages(hashes=[response.json()["execution"]["id_hash"]])
    )

    assert execution_post.messages is not None

    execution = asyncio.run(handle_execution(execution_post.messages[0]))
    assert execution is not None
    assert execution.status == ExecutionStatus.SUCCESS
    assert execution.resultID is not None

    result_post = asyncio.run(aleph_client.get_messages(hashes=[execution.resultID]))
    assert result_post.messages is not None
    print(result_post.messages[0])
