from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi_walletauth import WalletAuthDep

from ...core.model import Dataset, Execution, ExecutionStatus
from ..api_model import RequestExecutionRequest, RequestExecutionResponse
from ..common import request_permissions

router = APIRouter(
    prefix="/executions",
    tags=["executions"],
    responses={404: {"description": "Not found"}},
    dependencies=[WalletAuthDep],
)


@router.get("")
async def get_executions(
    dataset_id: Optional[str] = None,
    by: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[Execution]:
    params = {}
    if dataset_id:
        params["datasetID"] = dataset_id
    if by:
        params["owner"] = by
    if status:
        params["status"] = status
    if params:
        return await Execution.filter(**params).page(page=page, page_size=page_size)
    return await Execution.fetch_objects().page(page=page, page_size=page_size)


@router.post("")
async def request_execution(
    execution_request: RequestExecutionRequest,
) -> RequestExecutionResponse:
    """
    This endpoint is used to request an execution.
    If the user needs some permissions, the timeseries for which the user needs permissions are returned and
    the execution status is set to "requested". The needed permissions are also being requested. As soon as the
    permissions are granted, the execution is automatically executed.
    If some timeseries are not available, the execution is "denied" and the execution as well as the
    unavailable timeseries are returned.
    If the user has all permissions, the execution is started and the execution is returned.
    """
    execution = Execution(**execution_request.dict())
    if not execution.owner:
        raise HTTPException(status_code=400, detail="No owner specified")
    dataset = await Dataset.fetch(execution.datasetID).first()
    if not dataset:
        raise HTTPException(status_code=400, detail="Dataset not found")

    if dataset.owner == execution.owner and dataset.ownsAllTimeseries:
        execution.status = ExecutionStatus.PENDING
        return RequestExecutionResponse(
            execution=await execution.save(),
            permissionRequests=None,
            unavailableTimeseries=None,
        )
    (
        created_permissions,
        updated_permissions,
        unavailable_timeseries,
    ) = await request_permissions(dataset, execution)

    if unavailable_timeseries:
        execution.status = ExecutionStatus.DENIED
        return RequestExecutionResponse(
            execution=await execution.save(),
            unavailableTimeseries=unavailable_timeseries,
            permissionRequests=None,
        )
    if created_permissions or updated_permissions:
        new_permission_requests = created_permissions + updated_permissions
        execution.status = ExecutionStatus.REQUESTED
        return RequestExecutionResponse(
            execution=await execution.save(),
            unavailableTimeseries=None,
            permissionRequests=new_permission_requests,
        )
    else:
        execution.status = ExecutionStatus.PENDING
        return RequestExecutionResponse(
            execution=await execution.save(),
            unavailableTimeseries=None,
            permissionRequests=None,
        )
