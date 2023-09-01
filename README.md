# fishnet-cod
P2P Financial Signal Hosting Network on Aleph

## Initial setup
Install the FastAPI library and Uvicorn: 
```shell
poetry install
```
Activate the virtual environment, if not already done:
```shell
poetry shell
```

## Run on local
### Installing dev dependencies
Before you can run and develop the API locally, you need to install the
dev dependencies:
```shell
poetry install --dev
```

### Running the API
Uvicorn is used to run ASGI compatible web applications, such as the `app`
web application from the example above. You need to specify it the name of the
Python module to use and the name of the app:
```shell
python -m uvicorn src.fishnet_cod.api.main:app --reload
```

Then open the app in a web browser on http://localhost:8000

> Tip: With `--reload`, Uvicorn will automatically reload your code upon changes  

### Running the Executor
When the API is running, you can run the local executor to automatically
process the pending execution requests.
```shell
python /src/fishnet_cod/local_executor.py
```

## Testing
To run the tests, you need to [install the dev dependencies](#installing-dev-dependencies).

In order to avoid indexing all the messages and starting out with an empty database,
you need to set the `FISHNET_TEST_CHANNEL` environment variable to `true`:
```shell
export FISHNET_TEST_CHANNEL=true
```

Then, you can run the API tests with:
```shell
poetry run pytest src/fishnet_cod/api/test.py
```

**Note**: The tests run sequentially and if one fails, the following ones will also fail due to the event loop being closed.

## Environment variables

| Name                      | Description                                               | Type     | Default |
|---------------------------|-----------------------------------------------------------|----------|---------|
| `FISHNET_TEST_CACHE`      | Whether to use the test cache                             | `bool`   | `True`  |
| `FISHNET_TEST_CHANNEL`    | Whether to use a fresh test channel                       | `bool`   | `False` |
| `FISHNET_MESSAGE_CHANNEL` | The Aleph channel to use, is superseded by `TEST_CHANNEL` | `string` | `None`  |

Further environment variables are defined in the [conf.py](./src/fishnet_cod/core/conf.py) file.
Notice that the `FISHNET_` prefix is required for all environment variables listed there.