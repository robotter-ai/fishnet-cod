from fastapi import HTTPException, UploadFile
import pandas as pd


def load_data_df(file: UploadFile):
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Please provide a filename")
    if file.filename.endswith(".csv"):
        row = find_first_row_with_comma(file)
        file.file.seek(0)
        df = pd.read_csv(file.file, skiprows=row)
    elif file.filename.endswith(".parquet"):
        df = pd.read_parquet(file.file)
    elif file.filename.endswith(".feather"):
        df = pd.read_feather(file.file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format (only CSV, parquet and feather are supported)",
        )

    # find the first column with a timestamp or ISO8601 date and use it as the index
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=col)
            continue
        if is_timestamp_column(col):
            df.index = pd.to_datetime(df[col], infer_datetime_format=True)
            df = df.drop(columns=col)
        # non-numeric values are not supported and dropped
        elif df[col].dtype == object:
            df = df.drop(columns=col)
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid columns found in file",
        )
    return df


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

