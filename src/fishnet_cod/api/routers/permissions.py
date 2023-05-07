import asyncio
from typing import List

from ...core.model import (Dataset, Execution, ExecutionStatus, Permission,
                           PermissionStatus)
from ..api_model import ApprovePermissionsResponse, DenyPermissionsResponse
from ..common import request_permissions
from ..main import app
from ..utils import unique


@app.put("/permissions/approve")
async def approve_permissions(
    permission_hashes: List[str],
) -> ApprovePermissionsResponse:
    """
    Approve permission.
    This EndPoint will approve a list of permissions by their item hashes
    If an 'id_hashes' is provided, it will change all the Permission status
    to 'Granted'.
    """
    # TODO: Check if the user is the authorizer of the permissions
    permissions = await Permission.fetch(permission_hashes).all()
    if not permissions:
        return ApprovePermissionsResponse(
            updatedPermissions=[],
            triggeredExecutions=[],
        )

    # grant permissions
    dataset_ids = []
    permission_requests = []
    for permission in permissions:
        permission.status = PermissionStatus.GRANTED
        dataset_ids.append(permission.datasetID)
        permission_requests.append(permission.save())
    await asyncio.gather(*permission_requests)

    # get all requested executions and their datasets
    execution_requests = []
    dataset_requests = []
    for dataset_id in unique(dataset_ids):
        execution_requests.append(
            Execution.where_eq(
                datasetID=dataset_id, status=ExecutionStatus.REQUESTED
            ).all()
        )
        dataset_requests.append(Dataset.fetch(dataset_id).first())

    dataset_executions_map = {
        executions[0].datasetID: executions
        for executions in await asyncio.gather(*execution_requests)
        if executions and isinstance(executions[0], Execution)
    }

    # trigger executions if all permissions are granted
    execution_requests = []
    for dataset in await asyncio.gather(*dataset_requests):
        executions = dataset_executions_map.get(dataset.id_hash, [])
        for execution in executions:
            # TODO: Check if more efficient way to do this
            (
                created_permissions,
                updated_permissions,
                unavailable_timeseries,
            ) = await request_permissions(dataset, execution)
            if not created_permissions and not updated_permissions:
                execution.status = ExecutionStatus.PENDING
                execution_requests.append(await execution.save())
    triggered_executions = list(await asyncio.gather(*execution_requests))

    return ApprovePermissionsResponse(
        updatedPermissions=permissions,
        triggeredExecutions=triggered_executions,
    )


@app.put("/permissions/deny")
async def deny_permissions(permission_hashes: List[str]) -> DenyPermissionsResponse:
    """
    Deny permission.
    This EndPoint will deny a list of permissions by their item hashes
    If an `id_hashes` is provided, it will change all the Permission status
    to 'Denied'.
    """
    # TODO: Check if the user is the authorizer of the permissions
    permissions = await Permission.fetch(permission_hashes).all()
    if not permissions:
        return DenyPermissionsResponse(
            updatedPermissions=[],
            deniedExecutions=[],
        )

    # deny permissions and get dataset ids
    dataset_ids = []
    permission_requests = []
    for permission in permissions:
        permission.status = PermissionStatus.DENIED
        dataset_ids.append(permission.datasetID)
        permission_requests.append(permission.save())
    await asyncio.gather(*permission_requests)

    # get requested executions
    execution_requests = []
    for dataset_id in unique(dataset_ids):
        execution_requests.append(
            Execution.where_eq(
                datasetID=dataset_id, status=ExecutionStatus.REQUESTED
            ).all()
        )

    # deny executions
    execution_requests = []
    for executions in await asyncio.gather(*execution_requests):
        for execution in executions:
            execution.status = ExecutionStatus.DENIED
            execution_requests.append(await execution.save())
    denied_executions = list(await asyncio.gather(*execution_requests))

    return DenyPermissionsResponse(
        updatedPermissions=permissions, deniedExecutions=denied_executions
    )
