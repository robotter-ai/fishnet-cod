# fishnet-cod
P2P Financial Signal Hosting Network on Aleph

## Initial setup
Install the FastAPI library and Uvicorn: 
```shell
poetry install
```

### Running the API locally
Uvicorn is used to run ASGI compatible web applications, such as the `app`
web application from the example above. You need to specify it the name of the
Python module to use and the name of the app:
```shell
export TEST_CACHE=true  # when running locally, use the test cache

cd src  # the following command must be run from the src folder

python -m uvicorn fishnet_cod.api.main:app --reload
```

Then open the app in a web browser on http://localhost:8000

> Tip: With `--reload`, Uvicorn will automatically reload your code upon changes  

### Running the Executor locally
When the API is running, you can run the local executor to automatically
process the pending execution requests.
```shell
python /src/fishnet_cod/local_executor.py
```

## Environment variables

| Name            | Description                                               | Type     | Default |
|-----------------|-----------------------------------------------------------|----------|---------|
| `TEST_CACHE`    | Whether to use the test cache                             | `bool`   | `false` |
| `TEST_CHANNEL`  | Whether to use a fresh test channel                       | `bool`   | `false` |
| `ALEPH_CHANNEL` | The Aleph channel to use, is superseded by `TEST_CHANNEL` | `string` | `None`  |