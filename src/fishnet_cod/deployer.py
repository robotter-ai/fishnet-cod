# Deployment script for fishnet_cod
# Path: src/fishnet_cod/deployer.py

from .core.deployment import *
from .core.session import initialize_aars


def main():
    module_path = Path(__file__).parent
    squashfs_path = module_path.parent.parent / "packages.squashfs"
    executor_path = module_path / "executor"
    api_path = module_path / "api"
    aars_client = initialize_aars()
    authorized_session = aars_client.session

    with authorized_session as session:
        requirements = upload_source(
            deployer_session=session,
            path=squashfs_path,
            source_type=SourceType.REQUIREMENTS,
        )
        executors = deploy_executors(
            executor_path=executor_path,
            time_slices=[0, -1],
            requirements=requirements,
            deployer_session=session,
        )
        deploy_api(
            api_path=api_path,
            requirements=requirements,
            executors=executors,
            deployer_session=session,
        )


if __name__ == "__main__":
    main()
