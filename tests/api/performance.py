import time

from .conftest import login_with_signature


def test_preprocess_csv_performance(client, account, big_csv):
    token = login_with_signature(client, account)

    # measure time
    start = time.time()
    response = client.post(
        "/timeseries/csv",
        headers={"Authorization": f"Bearer {token}"},
        files={"data_file": ("test.csv", big_csv, "text/csv")},
    )
    end = time.time()
    print(f"Upload took {end - start} seconds")
    assert end - start < 1
    assert response.status_code == 200
    assert response.num_bytes_downloaded >= big_csv.__sizeof__()
