from itertools import chain
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from fastapi import Depends, UploadFile
from fastapi_walletauth import JWTWalletCredentials
from fastapi_walletauth.manager import JWTCredentialsManager
from fastapi_walletauth.middleware import BearerWalletAuth, jwt_credentials_manager
from starlette.requests import Request

from ..core.conf import settings
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


def get_file_path(timeseries_id: str) -> Path:
    dir_path = Path(f"timeseries")
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    file_path = dir_path / Path(f"{timeseries_id}.parquet")
    return file_path


def find_first_row_with_comma(file: UploadFile) -> int:
    """
    Find the first row in a csv file that contains a comma.
    """
    for i, line in enumerate(file.file):
        if b"," in line:
            return i
    raise ValueError("No comma found in file")


def is_timestamp_column(col: str) -> bool:
    """
    Check if a column name is a timestamp column.
    """
    col = col.lower()
    return "date" in col or "time" in col or "unix" in col or "timestamp" in col


def determine_decimal_places(data: Union[pd.Series, List[float]]):
    """
    Determine the number of decimal places to round to based on the magnitude of the data.
    Args:
        data (iterable): Iterable containing the data values.
    Returns:
        int: Number of decimal places to round to.
    """
    if len(data) == 0:
        return 0  # No data, no decimal places
    # Find the maximum magnitude (power of 10) of the data
    max_magnitude = int(np.floor(np.log10(np.abs(max(data)))))
    # Determine the number of decimal places based on the magnitude
    if max_magnitude >= 0:
        # If the magnitude is non-negative, round to 2 decimal places
        return 2
    else:
        # If the magnitude is negative, round to -max_magnitude + 2 decimal places
        return -max_magnitude + 2


class ConditionalJWTWalletAuth(BearerWalletAuth[JWTWalletCredentials]):
    def __init__(
        self,
        manager: JWTCredentialsManager,
        condition: Callable[[Request], Awaitable[bool]],
    ):
        self.condition = condition
        super().__init__(
            manager=manager,
        )

    async def __call__(self, request: Request) -> Optional[JWTWalletCredentials]:
        if await self.condition(request):
            return await super().__call__(request)
        else:
            return None
