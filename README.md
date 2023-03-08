# fishnet-cod
P2P Financial Signal Hosting Network on Aleph

## Initial setup
Install the FastAPI library and Uvicorn: 
```shell
pip install -r ./fishnet_api/requirements.txt
```

Uvicorn is used to run ASGI compatible web applications, such as the `app`
web application from the example above. You need to specify it the name of the
Python module to use and the name of the app:
```shell
python -m uvicorn fishnet_api:app --reload
```

Then open the app in a web browser on http://localhost:8000

> Tip: With `--reload`, Uvicorn will automatically reload your code upon changes  

## Upload on Aleph

The same `app` we just used with Gunicorn can be used by Aleph to run 
the web app, since Aleph attempts to be compatible with 
[ASGI](https://asgi.readthedocs.io/ASGI).

To upload the app, we can use the `aleph` command line tool. 
```shell
aleph program fishnet_api app
```

## Testing

Open the HTTP interface of a node running the VM Supervisor:

http://ip-of-supervisor:4020/vm/{message_hash}/

## Deployment
You can deploy your own Fishnet instance using the `fishnet_cod.deployment` package.

```python
from pathlib import Path

import asyncio

from aleph.sdk.client import AuthenticatedAlephClient
from aleph.sdk.chains.sol import get_fallback_account
from aleph.sdk.conf import settings

from fishnet_cod.deployment import deploy_api, deploy_executors

async def main():
    aleph_session = AuthenticatedAlephClient(
        get_fallback_account(),
        settings.API_HOST
    )  # you'll need tons of $ALEPH
    
    executors = await deploy_executors(
        executor_path=Path("/your/executor/folder"),
        time_slices=[0, -1],  # one executor for all data
        deployer_session=aleph_session,
        channel="MY_DEPLOYMENT_CHANNEL",
    )
    
    await deploy_api(
        api_path=Path("/your/api/folder"),
        deployer_session=aleph_session,
        executors=executors,
        channel="MY_DEPLOYMENT_CHANNEL",
    )

asyncio.run(main())
```