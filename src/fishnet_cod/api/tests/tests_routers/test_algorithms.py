import json
from os import putenv
from fastapi.testclient import TestClient
from ...main import app

client = TestClient(app)

putenv("TEST_CHANNEL", "true")


# Getting all algorithms awaiting success http-response 200
def test_get_algorithms():
    response = client.get("/algorithms")
    assert response.status_code == 200
    assert response.json() == []


# Upload algorithm and check if the data is correct
def test_upload_algorithm():
    algorithm_data = {
        "name": "First Algorithm",
        "desc": "It's a first test algorithm",
        "code": "def test(): pass",
        "owner": "Max Mustermann",
    }
    response = client.put("/algorithms", json=json.dumps(algorithm_data))
    print(json.dumps(algorithm_data))
    assert response.status_code == 200
    algorithm = response.json()
    assert algorithm["name"] == algorithm_data["name"]
    assert algorithm["desc"] == algorithm_data["desc"]
    assert algorithm["code"] == algorithm_data["code"]
    assert algorithm["owner"] == algorithm_data["owner"]
    assert algorithm.get("item_hash") is not None


# Dump algorithm data and then try to get it and compare them
def test_get_algorithm():
    algorithm_data = {
        "name": "Second Algorithm",
        "desc": "It's a second test algorithm",
        "code": "def test(): pass",
        "owner": "Max Mustermann",
    }
    response = client.put("/algorithms", json=json.dumps(algorithm_data))
    algorithm = response.json()
    algorithm_id = algorithm["item_hash"]

    response = client.get(f"/algorithms/{algorithm_id}")
    assert response.status_code == 200
    fetched_algorithm = response.json()
    assert fetched_algorithm["name"] == algorithm_data["name"]
    assert fetched_algorithm["desc"] == algorithm_data["desc"]
    assert fetched_algorithm["code"] == algorithm_data["code"]
    assert fetched_algorithm["owner"] == algorithm_data["owner"]


# Trying to get a nonexistent algorithm, which raises HTTPError with code 404
def test_get_algorithm_not_found():
    response = client.get("/algorithms/no-id")
    assert response.status_code == 404
    assert response.json() == {"detail": "Algorithm not found"}
