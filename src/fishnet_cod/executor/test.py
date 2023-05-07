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

from .main import handle_execution

client = TestClient(app)
aleph_client = AlephClient(settings.API_HOST)


def test_execution() -> None:
    timeseries_req = UploadTimeseriesRequest(
        timeseries=[
            TimeseriesItem(
                id_hash=None,
                name="test_execution_timeseries1",
                desc=None,
                owner="executooor",
                data=[(i, 1) for i in range(100)],
            ),
            TimeseriesItem(
                id_hash=None,
                name="test_execution_timeseries2",
                desc=None,
                owner="executooor",
                data=[(i, 10) for i in range(100)],
            ),
        ]
    )
    req_body = timeseries_req.dict()
    response = client.put("/timeseries/upload", json=req_body)
    assert response.status_code == 200
    assert response.json()[0]["id_hash"] is not None
    timeseries_ids = [ts["id_hash"] for ts in response.json()]

    dataset_req = UploadDatasetRequest(
        id_hash=None,
        name="test_execution_dataset",
        desc=None,
        owner="executooor",
        ownsAllTimeseries=True,
        timeseriesIDs=timeseries_ids,
    )
    response = client.put("/datasets/upload", json=dataset_req.dict())
    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    dataset_id = response.json()["id_hash"]

    algo_req = UploadAlgorithmRequest(
        id_hash=None,
        name="test_execution_algorithm",
        desc="sums all columns",
        owner="executooor",
        code="""
def run(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return df.sum(axis=0)
""",
    )
    response = client.put("/algorithms/upload", json=algo_req.dict())
    assert response.status_code == 200
    assert response.json()["id_hash"] is not None
    algorithm_id = response.json()["id_hash"]

    exec_req: RequestExecutionRequest = RequestExecutionRequest(
        algorithmID=algorithm_id, datasetID=dataset_id, owner="executooor", status=None
    )
    response = client.post("/executions/request", json=exec_req.dict())
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
