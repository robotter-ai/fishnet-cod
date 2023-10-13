from itertools import chain
from pathlib import Path
from typing import TypeVar

from fastapi import Depends, UploadFile
from fastapi_walletauth.middleware import BearerWalletAuth, jwt_credentials_manager

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
    dir_path = Path(settings.DATA_PATH) / Path(f"timeseries")
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
