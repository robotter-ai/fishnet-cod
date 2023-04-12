# WARNING: This is a work in progress
## Upload on Aleph

The same `app` we just used with Gunicorn can be used by Aleph to run 
the web app, since Aleph is compatible with 
[ASGI](https://asgi.readthedocs.io/ASGI).

### Deployment
You can deploy your own Fishnet instance using the `fishnet_cod.deployment` package.

```python
from pathlib import Path

import asyncio

from aleph.sdk.client import AuthenticatedAlephClient
from aleph.sdk.chains.sol import get_fallback_account
from aleph.sdk.conf import settings

from fishnet_cod.deployer import deploy_api, deploy_executors

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