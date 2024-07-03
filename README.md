# fishnet-cod: P2P Financial Signal Hosting Network

Fishnet is a Compute-over-Data (CoD) system that uses the distributed aleph.im network as a substrate for storage and computation.

It is conceptualized as a decentralized, peer-to-peer, and serverless system that allows users to run statistical computations on their
timeseries data without having to upload it to a centralized server.

**As of now, the system is still in development and not ready for production use. Do not upload sensitive data.**

## Structure
### [fishnet_cod.core](fishnet_cod/core)
This module contains a common data model, built on the
[Aleph Active Record SDK (AARS)](https://github.com/aleph-im/active-record-sdk), that is being used by the Fishnet API
and Executor VMs. The data model is used to store and query `Datasets`, `Users` and `Permissions`.

Past revisions of this repository contain the Proof-of-Concept of a distributed compute system that allowed users to
upload Algorithms and request Executions, which were then processed by the Executor VMs. This functionality is currently
not available and will be re-implemented in the future, with a UX like [dask](https://dask.org/) and
[substra](https://substra.ai/) in mind.

### [fishnet_cod.api](fishnet_cod/api)
This module contains the API that is used to communicate with aleph.im's
message channels. Currently, it still contains code to upload and store datasets, which will later be moved to
its own `data_node` module.

Users can generate `Views` of their datasets through the API, as previews for potential buyers.

### [fishnet_cod.local_listener](fishnet_cod/local_listener)
Aleph.im offers a feature that allows VMs to listen to messages on a channel and react to them. This module contains a
service that emulates this feature locally, and forwards the messages to the API.

### ToDo: fishnet_cod.data_node
Data nodes store and process time slices of the financial data. The client is responsible for uploading the data to the
various data nodes, which each generate analytical statistics for their respective time slices. The analysis is posted
to aleph.im and subsequently aggregated by the API.

### ToDo: fishnet_cod.client
A simple Python-based client that allows users to upload and download data to and from the data nodes, as well as
interacting with the API to request access or to purchase datasets.

## Contributing
### Initial setup
Install the FastAPI library and Uvicorn: 
```shell
poetry install
```
Activate the virtual environment, if not already done:
```shell
poetry shell
```

### Run on local
#### Installing dev dependencies
Before you can run and develop the API locally, you need to install the
dev dependencies:
```shell
poetry install --dev
```

#### Running the API
Uvicorn is used to run ASGI compatible web applications, such as the `app`
web application from the example above. You need to specify it the name of the
Python module to use and the name of the app:
```shell
python -m uvicorn fishnet_cod.api.main:app --reload
```

Then open the app in a web browser on http://localhost:8000

> Tip: With `--reload`, Uvicorn will automatically reload your code upon changes

### Testing
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

### Environment variables

| Name                      | Description                                               | Type     | Default |
|---------------------------|-----------------------------------------------------------|----------|---------|
| `FISHNET_TEST_CACHE`      | Whether to use the test cache                             | `bool`   | `True`  |
| `FISHNET_TEST_CHANNEL`    | Whether to use a fresh test channel                       | `bool`   | `False` |
| `FISHNET_MESSAGE_CHANNEL` | The Aleph channel to use, is superseded by `TEST_CHANNEL` | `string` | `None`  |
| `FISHNET_DISABLE_AUTH`    | Whether mandatory authentication is disabled              | `bool`   | `False` |

Further environment variables are defined in the [conf.py](fishnet_cod/core/conf.py) file.
Notice that the `FISHNET_` prefix is required for all environment variables listed there.


## Roadmap

- [x] Basic message model
- [x] API for communicating with Fishnet system
  - [x] Basic CRUD operations
  - [x] Permission management
  - [x] Local VM caching
  - [x] Signature verification of requests
  - [x] Discovery of other API instances
  - [x] Dedicated API deploy function
  - [ ] Timeslice distribution across Executor nodes
  - [ ] Background tasks
    - [x] Message listener
    - [ ] View (re-)generation
    - [ ] Statistics aggregation
- [ ] Data Nodes *(WIP)*
  - [x] Basic data node upload & download
  - [ ] Data node discovery
  - [ ] Data node deployment
- [ ] Client *(WIP)*
  - [x] Login with signature
  - [ ] API interaction
  - [ ] Dataset preparation
  - [ ] Data node discovery
  - [ ] Data node interaction
  - [ ] [Brick Marketplace](https://www.brickprotocol.xyz/) interaction for buying datasets
- [x] Executor VM *(ON HOLD)*
  - [x] Listens for `Execution` messages and executes them
  - [x] Uploads results to aleph.im
  - [x] Pandas support
  - [x] Dedicated Executor deploy function
  - [ ] Distributed execution & aggregation
    - [x] Discovery of other Executor instances
    - [x] Uploading executors with metadata: assigned timeslice, code version
    - [ ] Deploy multiple executors
  - [ ] Different execution environments (e.g. PyTorch, Tensorflow)
  - [ ] GPU support
