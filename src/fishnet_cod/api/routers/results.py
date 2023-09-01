from fastapi import APIRouter, HTTPException
from fastapi_walletauth import WalletAuthDep

from ...core.model import Result

router = APIRouter(
    prefix="/results",
    tags=["results"],
    responses={404: {"description": "Not found"}},
    dependencies=[WalletAuthDep],
)


@router.get("/{result_id}")
async def get_result(result_id: str) -> Result:
    result = await Result.fetch(result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
