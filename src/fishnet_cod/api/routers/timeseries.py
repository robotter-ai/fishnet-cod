import asyncio
from typing import List

from fastapi import APIRouter
from fastapi_walletauth import JWTWalletAuthDep

from ..common import AuthorizedRouterDep
from ...core.model import Timeseries
from ..api_model import UploadTimeseriesRequest

router = APIRouter(
    prefix="/timeseries",
    tags=["timeseries"],
    responses={404: {"description": "Not found"}},
    dependencies=[AuthorizedRouterDep],
)


@router.put("")
async def upload_timeseries(
    req: UploadTimeseriesRequest,
    user: JWTWalletAuthDep
) -> List[Timeseries]:
    """
    Upload a list of timeseries. If the passed timeseries has an `item_hash` and it already exists,
    it will be overwritten. If the timeseries does not exist, it will be created.
    A list of the created/updated timeseries is returned. If the list is shorter than the passed list, then
    it might be that a passed timeseries contained illegal data.
    """
    ids_to_fetch = [ts.item_hash for ts in req.timeseries if ts.item_hash is not None]
    requests = []
    old_time_series = (
        {ts.item_hash: ts for ts in await Timeseries.fetch(ids_to_fetch).all()}
        if ids_to_fetch
        else {}
    )
    for ts in req.timeseries:
        if old_time_series.get(ts.item_hash) is None:
            requests.append(Timeseries(**dict(ts), owner=user.address).save())
            continue
        old_ts: Timeseries = old_time_series[ts.item_hash]
        old_ts.name = ts.name
        old_ts.desc = ts.desc
        requests.append(old_ts.save())
    upserted_timeseries = await asyncio.gather(*requests)
    return [ts for ts in upserted_timeseries if not isinstance(ts, BaseException)]
