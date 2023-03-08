# Deployment script for fishnet_cod
# Path: src/fishnet_cod/deployer.py

import asyncio
from core.deployment import *
from core.session import authorized_session


async def main():
    module_path = Path(__file__).parent
    squashfs_path = module_path.parent.parent / "packages.squashfs"
    executor_path = module_path / "executor"
    api_path = module_path / "api"

    requirements = await upload_source(
        deployer_session=authorized_session,
        path=squashfs_path,
        source_type=SourceType.REQUIREMENTS,
    )
    executors = await deploy_executors(
        executor_path=executor_path,
        time_slices=[0, -1],
        requirements=requirements,
        deployer_session=authorized_session,
    )
    await deploy_api(
        api_path=api_path,
        requirements=requirements,
        executors=executors,
        deployer_session=authorized_session,
    )


if __name__ == "__main__":
    asyncio.run(main())
