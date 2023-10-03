from itertools import chain
from pathlib import Path
from typing import TypeVar

from fastapi import Depends
from fastapi_walletauth.middleware import BearerWalletAuth, jwt_credentials_manager

from ..core.model import Granularity

T = TypeVar("T")


def flatten(iterable):
    """Flatten one level of nesting"""
    return chain.from_iterable(iterable)


def granularity_to_interval(granularity: Granularity) -> str:
    """
    Get pandas-compatible interval from Granularity

    Args:
        start: start timestamp
        end: end timestamp
        granularity: granularity (frequency) of timestamps

    Returns:
        List of timestamps
    """
    if granularity == Granularity.DAY:
        return "5min"
    elif granularity == Granularity.WEEK:
        return "15min"
    elif granularity == Granularity.MONTH:
        return "H"
    elif granularity == Granularity.THREE_MONTHS:
        return "3H"
    else:  # granularity == Granularity.YEAR:
        return "D"


AuthorizedRouterDep = Depends(BearerWalletAuth(jwt_credentials_manager))


async def get_file_path(dataset_id):
    # TODO: Fix this
    file_path = Path(f"./files/{dataset_id}.parquet")
    return file_path
