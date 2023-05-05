import asyncio
import logging
import os
from os import listdir
from typing import List, Optional, Dict

import pandas as pd
from aleph.sdk.exceptions import BadSignatureError
from aleph_message.models import PostMessage
from pydantic import ValidationError

from .api_model import (
    UploadTimeseriesRequest,
    UploadDatasetRequest,
    UploadAlgorithmRequest,
    RequestExecutionRequest,
    RequestExecutionResponse,
    PutUserInfo,
    PutViewRequest,
    PutViewResponse,
    Attribute,
    NotificationType,
    Notification,
    MultiplePermissions,
    FungibleAssetStandard,
    UploadDatasetTimeseriesRequest,
    UploadDatasetTimeseriesResponse,
    DatasetResponse,
    PostPermission,
    PermissionPostedResponse,
    MessageResponse,
    TokenChallengeResponse,
    BearerTokenResponse,
)
from .auth import AuthTokenManager, SupportedChains
from ..core.constants import API_MESSAGE_FILTER
from ..core.model import (
    Timeseries,
    UserInfo,
    Algorithm,
    Execution,
    Permission,
    DatasetPermissionStatus,
    PermissionStatus,
    ExecutionStatus,
    Result,
    Dataset,
    Granularity,
    View,
)
from ..core.session import initialize_aars

logger = logging.getLogger("uvicorn")

logger.debug("import aleph_client")
from aleph.sdk.vm.app import AlephApp

logger.debug("import aars")
from aars import AARS, Record

logger.debug("import fastapi")
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer


logger.debug("import project modules")

logger.debug("imports done")

http_app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/challenge")

origins = ["*"]

http_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = AlephApp(http_app=http_app)
global aars_client, auth_manager
aars_client: AARS
auth_manager: AuthTokenManager


async def re_index():
    logger.info(f"API re-indexing channel {AARS.channel}")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@app.on_event("startup")
async def startup():
    global aars_client, auth_manager
    aars_client = initialize_aars()
    auth_manager = AuthTokenManager()
    await re_index()


@app.get("/")
async def index():
    if os.path.exists("/opt/venv"):
        opt_venv = list(listdir("/opt/venv"))
    else:
        opt_venv = []
    return {
        "vm_name": "fishnet_api",
        "endpoints": [
            "/docs",
        ],
        "files_in_volumes": {
            "/opt/venv": opt_venv,
        },
    }


@app.post("/auth/challenge")
async def create_challenge(
    pubkey: str, chain: SupportedChains
) -> TokenChallengeResponse:
    global auth_manager
    challenge = auth_manager.get_challenge(pubkey=pubkey, chain=chain)
    return TokenChallengeResponse(
        pubkey=challenge.pubkey,
        chain=challenge.chain,
        challenge=challenge.challenge,
        valid_til=challenge.valid_til,
    )


@app.post("/auth/solve")
async def solve_challenge(
    pubkey: str, chain: SupportedChains, signature: str
) -> BearerTokenResponse:
    global auth_manager
    try:
        auth = auth_manager.solve_challenge(
            pubkey=pubkey, chain=chain, signature=signature
        )
        return BearerTokenResponse(
            pubkey=auth.pubkey,
            chain=auth.chain,
            token=auth.token,
            valid_til=auth.valid_til,
        )
    except BadSignatureError:
        raise HTTPException(403, "Challenge failed")
    except TimeoutError:
        raise HTTPException(403, "Challenge timeout")


@app.post("/auth/refresh")
async def refresh_token(token: str) -> BearerTokenResponse:
    global auth_manager
    try:
        auth = auth_manager.refresh_token(token)
    except TimeoutError:
        raise HTTPException(403, "Token expired")
    return BearerTokenResponse(
        pubkey=auth.pubkey, chain=auth.chain, token=auth.token, valid_til=auth.valid_til
    )


@app.post("/auth/logout")
async def logout(token: str):
    global auth_manager
    auth_manager.remove_token(token)


@app.get("/algorithms")
async def get_algorithms(
    name: Optional[str] = None,
    by: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[Algorithm]:
    """
    Get all algorithms filtered by `name` and/or owner (`by`). If no filters are given, all algorithms are returned.
    """
    if name or by:
        algo_request = Algorithm.where_eq(name=name, owner=by)
    else:
        algo_request = Algorithm.fetch_objects()
    return await algo_request.page(page=page, page_size=page_size)


@app.get("/algorithms/{algorithm_id}")
async def get_algorithm(algorithm_id: str) -> Algorithm:
    algorithm = await Algorithm.fetch(algorithm_id).first()
    if algorithm is None:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return algorithm


@app.put("/algorithms/upload")
async def upload_algorithm(algorithm: UploadAlgorithmRequest) -> Algorithm:
    """
    Upload an algorithm.
    If an `id_hash` is provided, it will update the algorithm with that id.
    """
    if algorithm.id_hash is not None:
        old_algorithm = await Algorithm.fetch(algorithm.id_hash).first()
        if old_algorithm is not None:
            if old_algorithm.owner != algorithm.owner:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot overwrite algorithm that is not owned by you",
                )
            old_algorithm.name = algorithm.name
            old_algorithm.desc = algorithm.desc
            old_algorithm.code = algorithm.code
            return await old_algorithm.save()
    return await Algorithm(**algorithm.dict()).save()


@app.get("/datasets")
async def get_datasets(
    id: Optional[str] = None,
    view_as: Optional[str] = None,
    by: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[DatasetResponse]:
    """
    Get all datasets. Returns a list of tuples of datasets and their permission status for the given `view_as` user.
    If `view_as` is not given, the permission status will be `none` for all datasets.
    If `id` is given, it will return the dataset with that id.
    If `by` is given, it will return all datasets owned by that user.
    """
    if id:
        datasets_resp = Dataset.fetch(id)
    elif by:
        datasets_resp = Dataset.where_eq(owner=by)
    else:
        datasets_resp = Dataset.fetch_objects()
    datasets = await datasets_resp.page(page=page, page_size=page_size)

    if view_as:
        ts_ids = []
        for rec in datasets:
            ts_ids.extend(rec.timeseriesIDs)
        ts_ids_unique = list(set(ts_ids))

        req = [
            Permission.where_eq(timeseriesID=ts_id, authorizer=view_as).all()
            for ts_id in ts_ids_unique
        ]
        resp = await asyncio.gather(*req)
        permissions = [item for sublist in resp for item in sublist]

        returned_datasets: List[DatasetResponse] = []
        for rec in datasets:
            dataset_permissions = []
            for ts_id in rec.timeseriesIDs:
                dataset_permissions.extend(
                    list(filter(lambda x: x.timeseriesID == ts_id, permissions))
                )
            if not dataset_permissions:
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(),
                        permission_status=DatasetPermissionStatus.NOT_REQUESTED,
                    )
                )
                continue

            permission_status = [perm_rec for perm_rec in dataset_permissions]
            if all(status == PermissionStatus.GRANTED for status in permission_status):
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(), permission_status=DatasetPermissionStatus.GRANTED
                    )
                )
            elif PermissionStatus.DENIED in permission_status:
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(), permission_status=DatasetPermissionStatus.DENIED
                    )
                )
            elif PermissionStatus.REQUESTED in permission_status:
                returned_datasets.append(
                    DatasetResponse(
                        **rec.dict(),
                        permission_status=DatasetPermissionStatus.REQUESTED,
                    )
                )
        return returned_datasets
    else:
        return [
            DatasetResponse(**rec.dict(), permission_status=None) for rec in datasets
        ]


@app.get("/datasets/{dataset_id}/permissions")
async def get_dataset_permissions(dataset_id: str) -> List[Permission]:
    """
    Get all granted permissions for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="No Dataset found")
    ts_ids = [ids for ids in dataset.timeseriesIDs]
    matched_permission_records = [
        Permission.where_eq(timeseriesID=ids, status=PermissionStatus.GRANTED).all()
        for ids in ts_ids
    ]
    records = await asyncio.gather(*matched_permission_records)
    permission_records = [element for row in records for element in row if element]

    return permission_records


@app.get("/dataset/{dataset_id}/metaplex")
async def get_dataset_metaplex_dataset(dataset_id: str) -> FungibleAssetStandard:
    """
    Get the metaplex metadata for a given dataset.
    """
    dataset = await Dataset.fetch(dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return FungibleAssetStandard(
        name=dataset.name,
        symbol=dataset.id_hash,
        description=dataset.desc,
        # TODO: Generate chart image
        image="https://ipfs.io/ipfs/Qma2eje8yY57pNuaUyo4dsjtB9xwPz5yV6pCbK2PxpjUzo",
        animation_url=None,
        external_url=f"http://localhost:5173/data/{dataset.id_hash}/details",
        attributes=[
            Attribute(trait_type="Owner", value=dataset.owner),
            Attribute(trait_type="Last Updated", value=dataset.timestamp),
            Attribute(trait_type="Columns", value=str(len(dataset.timeseriesIDs))),
        ],
    )


@app.put("/timeseries/upload")
async def upload_timeseries(req: UploadTimeseriesRequest) -> List[Timeseries]:
    """
    Upload a list of timeseries. If the passed timeseries has an `id_hash` and it already exists,
    it will be overwritten. If the timeseries does not exist, it will be created.
    A list of the created/updated timeseries is returned. If the list is shorter than the passed list, then
    it might be that a passed timeseries contained illegal data.
    """
    ids_to_fetch = [ts.id_hash for ts in req.timeseries if ts.id_hash is not None]
    requests = []
    old_time_series = (
        {ts.id_hash: ts for ts in await Timeseries.fetch(ids_to_fetch).all()}
        if ids_to_fetch
        else {}
    )
    for ts in req.timeseries:
        if old_time_series.get(ts.id_hash) is None:
            requests.append(Timeseries(**dict(ts)).save())
            continue
        old_ts: Timeseries = old_time_series[ts.id_hash]
        if ts.owner != old_ts.owner:
            raise HTTPException(
                status_code=403,
                detail="Cannot overwrite timeseries that is not owned by you",
            )
        old_ts.name = ts.name
        old_ts.data = ts.data
        old_ts.desc = ts.desc
        requests.append(old_ts.save())
    upserted_timeseries = await asyncio.gather(*requests)
    return [ts for ts in upserted_timeseries if not isinstance(ts, BaseException)]


@app.post("/timeseries/csv/preprocess")
async def upload_timeseries_csv(
    owner: str = Form(...), data_file: UploadFile = File(...)
) -> List[Timeseries]:
    """
    Upload a csv file with timeseries data. The csv file must have a header row with the following columns:
    `id_hash`, `name`, `desc`, `data`. The `data` column must contain a json string with the timeseries data.
    """
    if data_file.filename and not data_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a csv file")
    df = pd.read_csv(data_file.file)
    # find the first column with a timestamp or ISO8601 date and use it as the index
    for col in df.columns:
        if "unnamed" in col.lower():
            df = df.drop(columns=col)
            continue
        if "date" in col.lower() or "time" in col.lower():
            df.index = pd.to_datetime(df[col])
            print(f"Using column {col} as index")
            df = df.drop(columns=col)
    # create a timeseries object for each column
    timestamps = [dt.timestamp() for dt in df.index.to_pydatetime().tolist()]
    timeseries = []
    for col in df.columns:
        try:
            data = [(timestamps[i], value) for i, value in enumerate(df[col].tolist())]
            timeseries.append(
                Timeseries(
                    id_hash=None,
                    name=col,
                    desc=None,
                    data=data,
                    owner=owner,
                )
            )
        except ValidationError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data encountered in column {col}: {data}",
            )
    return timeseries


@app.put("/datasets/upload")
async def upload_dataset(dataset: UploadDatasetRequest) -> Dataset:
    """
    Upload a dataset.
    If an `id_hash` is provided, it will update the dataset with that id.
    """
    if dataset.ownsAllTimeseries:
        timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
        dataset.ownsAllTimeseries = all(
            [ts.owner == dataset.owner for ts in timeseries]
        )
    if dataset.id_hash is not None:
        old_dataset = await Dataset.fetch(dataset.id_hash).first()
        if old_dataset is not None:
            if old_dataset.owner != dataset.owner:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot overwrite dataset that is not owned by you",
                )
            old_dataset.name = dataset.name
            old_dataset.desc = dataset.desc
            old_dataset.timeseriesIDs = dataset.timeseriesIDs
            old_dataset.ownsAllTimeseries = dataset.ownsAllTimeseries
            return await old_dataset.save()
    return await Dataset(**dataset.dict()).save()


@app.post("/datasets/upload/timeseries")
async def upload_dataset_timeseries(
    upload_dataset_timeseries_request: UploadDatasetTimeseriesRequest,
) -> UploadDatasetTimeseriesResponse:
    """
    Upload a dataset and timeseries at the same time.
    """
    if upload_dataset_timeseries_request.dataset.id_hash is not None:
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update a dataset. Use PUT /datasets/upload instead.",
        )
    if any([ts.id_hash is None for ts in upload_dataset_timeseries_request.timeseries]):
        raise HTTPException(
            status_code=400,
            detail="Cannot use this POST endpoint to update timeseries. Use PUT /timeseries/upload instead.",
        )
    timeseries = await upload_timeseries(
        req=UploadTimeseriesRequest(
            timeseries=upload_dataset_timeseries_request.timeseries
        )
    )
    dataset = await upload_dataset(req=upload_dataset_timeseries_request.dataset)
    return UploadDatasetTimeseriesResponse(
        dataset=dataset,
        timeseries=[ts for ts in timeseries if not isinstance(ts, BaseException)],
    )


def get_timestamps_by_granularity(
    start: int, end: int, granularity: Granularity
) -> List[int]:
    """
    Get timestamps by granularity
    :param start: start timestamp
    :param end: end timestamp
    :param granularity: granularity
    :return: list of timestamps
    """
    if granularity == Granularity.DAY:
        interval = 60 * 5
    elif granularity == Granularity.WEEK:
        interval = 60 * 15
    elif granularity == Granularity.MONTH:
        interval = 60 * 60
    elif granularity == Granularity.THREE_MONTHS:
        interval = 60 * 60 * 3
    else:  # granularity == Granularity.YEAR:
        interval = 60 * 60 * 24
    timestamps = []
    for i in range(start, end, interval):
        timestamps.append(i)
    return timestamps


@app.put("/datasets/{dataset_id}/views")
async def generate_view(
    dataset_id: str, view_params: List[PutViewRequest]
) -> PutViewResponse:
    # get the dataset
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    view_requests = []
    for view_req in view_params:
        # get all the timeseries
        timeseries = await Timeseries.fetch(view_req.timeseriesIDs).all()
        view_values = {}
        for ts in timeseries:
            # normalize and round values
            values = [p[1] for p in ts.data]
            ts.min = min(values)
            ts.max = max(values)
            normalized = [
                (p[0], round((p[1] - ts.min) / (ts.max - ts.min)), 2) for p in ts.data
            ]
            # drop points according to granularity
            thinned = []
            i = 0  # cursor for normalized entries
            timestamps = get_timestamps_by_granularity(
                view_req.startTime, view_req.endTime, view_req.granularity
            )
            # append each point that is closest to the timestamp
            for timestamp in timestamps:
                while i < len(normalized) and normalized[i][0] < timestamp:
                    i += 1
                if i == len(normalized):
                    break
                if i == 0:
                    thinned.append(normalized[i])
                else:
                    if abs(normalized[i][0] - timestamp) < abs(
                        normalized[i - 1][0] - timestamp
                    ):
                        thinned.append(normalized[i])
                    else:
                        thinned.append(normalized[i - 1])

            view_values[ts.id_hash] = thinned

        # prepare view request
        view_requests.append(
            View(
                id_hash=view_req.id_hash,
                startTime=view_req.startTime,
                endTime=view_req.endTime,
                granularity=view_req.granularity,
                values=view_values,
            ).save()
        )

    # save all records
    views = await asyncio.gather(*view_requests)
    dataset.views = [view.id_hash for view in views]
    await dataset.save()

    return PutViewResponse(dataset=dataset, views=views)


@app.put("/datasets/{dataset_id}/available/{available}")
async def set_dataset_available(dataset_id: str, available: bool) -> Dataset:
    """
    Set a dataset to be available or not. This will also update the status of all
    executions that are waiting for permission on this dataset.
    param `dataset_id':put the dataset hash here
    param 'available':put the Boolean value
    """

    requests = []
    dataset = await Dataset.fetch(dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="No Dataset found")
    dataset.available = available
    requests.append(dataset.save())

    ts_list = await Timeseries.fetch(dataset.timeseriesIDs).all()
    if not ts_list:
        raise HTTPException(status_code=424, detail="No Timeseries found")

    for rec in ts_list:
        if rec.available != available:
            rec.available = available
            requests.append(rec.save())
    executions_records = await Execution.fetch(dataset_id).all()
    for rec in executions_records:
        if rec.status == ExecutionStatus.PENDING:
            rec.status = ExecutionStatus.DENIED
            requests.append(rec.save())

    await asyncio.gather(*requests)
    return dataset


@app.get("/executions")
async def get_executions(
    dataset_id: Optional[str] = None,
    by: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[Execution]:
    if dataset_id or by or status:
        execution_requests = Execution.where_eq(
            datasetID=dataset_id, owner=by, status=status
        )
    else:
        execution_requests = Execution.fetch_objects()
    return await execution_requests.page(page=page, page_size=page_size)


@app.post("/executions/request")
async def request_execution(
    execution: RequestExecutionRequest,
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
    if not execution.owner:
        raise HTTPException(status_code=400, detail="No owner specified")
    dataset = await Dataset.fetch(execution.datasetID).first()

    if dataset.owner == execution.owner and dataset.ownsAllTimeseries:
        execution.status = ExecutionStatus.PENDING
        return RequestExecutionResponse(
            execution=await Execution(**execution.dict()).save(),
            permissionRequests=None,
            unavailableTimeseries=None,
        )

    timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    requested_permissions = [
        Permission.where_eq(timeseriesID=tsID, requestor=execution.owner).first()
        for tsID in dataset.timeseriesIDs
    ]
    permissions: List[Permission] = list(await asyncio.gather(*requested_permissions))
    ts_permission_map: Dict[str, Permission] = {
        permission.timeseriesID: permission for permission in permissions if permission
    }
    requests = []
    unavailable_timeseries = []
    for ts in timeseries:
        if ts.owner == execution.owner:
            continue
        if not ts.available:
            unavailable_timeseries.append(ts)
        if timeseries:
            continue
        if ts.id_hash not in ts_permission_map:
            requests.append(
                Permission(
                    timeseriesID=ts.id_hash,
                    algorithmID=execution.algorithmID,
                    owner=ts.owner,
                    requestor=execution.owner,
                    status=PermissionStatus.REQUESTED,
                    executionCount=0,
                    maxExecutionCount=-1,
                ).save()
            )
        else:
            permission = ts_permission_map[ts.id_hash]
            needs_update = False
            if permission.status == PermissionStatus.DENIED:
                permission.status = PermissionStatus.REQUESTED
                needs_update = True
            if (
                permission.maxExecutionCount
                and permission.maxExecutionCount <= permission.executionCount
            ):
                permission.maxExecutionCount = permission.executionCount + 1
                permission.status = PermissionStatus.REQUESTED
                needs_update = True
            if needs_update:
                requests.append(permission.save())
    if unavailable_timeseries:
        execution.status = ExecutionStatus.DENIED
        return RequestExecutionResponse(
            execution=await Execution(**execution.dict()).save(),
            unavailableTimeseries=unavailable_timeseries,
            permissionRequests=None,
        )
    if requests:
        new_permission_requests = await asyncio.gather(*requests)
        execution.status = ExecutionStatus.REQUESTED
        return RequestExecutionResponse(
            execution=await Execution(**execution.dict()).save(),
            unavailableTimeseries=None,
            permissionRequests=new_permission_requests,
        )
    else:
        execution.status = ExecutionStatus.PENDING
        return RequestExecutionResponse(
            execution=await Execution(**execution.dict()).save(),
            unavailableTimeseries=None,
            permissionRequests=None,
        )


@app.put("/permissions/approve")
async def approve_permissions(permission_hashes: List[str]) -> List[Permission]:
    """
    Approve permission.
    This EndPoint will approve a list of permissions by their item hashes
    If an 'id_hashes' is provided, it will change all the Permission status
    to 'Granted'.
    """

    ts_ids = []
    requests = []

    permission_records = await Permission.fetch(permission_hashes).all()
    if not permission_records:
        raise HTTPException(
            status_code=404, detail="No Permission Found with this Hashes"
        )

    for rec in permission_records:
        rec.status = PermissionStatus.GRANTED
        ts_ids.append(rec.timeseriesID)
        requests.append(rec.save())

    ds_ids = []
    dataset_records = await Dataset.where_eq(timeseriesIDs=ts_ids).all()
    if not dataset_records:
        raise HTTPException(status_code=404, detail="No Dataset found")
    for rec in dataset_records:
        if rec.id_hash in ds_ids:
            ds_ids.append(rec.id_hash)

    executions_records = await Execution.where_eq(datasetID=ds_ids).all()
    for rec in executions_records:
        if ds_ids and rec.datasetID in ds_ids:
            rec.status = ExecutionStatus.PENDING
            requests.append(rec.save())
    await asyncio.gather(*requests)
    return permission_records


@app.put("/permissions/deny")
async def deny_permissions(permission_hashes: List[str]) -> List[Permission]:
    """
    Deny permission.
    This EndPoint will deny a list of permissions by their item hashes
    If an `id_hashes` is provided, it will change all the Permission status
    to 'Denied'.
    """
    permission_records = await Permission.fetch(permission_hashes).all()
    if not permission_records:
        raise HTTPException(
            status_code=404, detail="No Permission found with this Hashes"
        )

    ts_ids = []
    requests = []
    for rec in permission_records:
        rec.status = PermissionStatus.DENIED
        ts_ids.append(rec.timeseriesID)
        requests.append(rec.save())
    dataset_records = await Dataset.where_eq(timeseriesIDs=ts_ids).all()
    ds_ids = []
    if not dataset_records:
        raise HTTPException(status_code=424, detail="No Timeseries found")
    for rec in dataset_records:
        ds_ids.append(rec.id_hash)
    executions_records = await Execution.where_eq(datasetID=ds_ids).all()
    for rec in executions_records:
        if rec.datasetID in ds_ids and rec.status == ExecutionStatus.PENDING:
            rec.status = ExecutionStatus.DENIED
            requests.append(rec.save())

    await asyncio.gather(*requests)
    return permission_records


@app.get("/user/{address}")
async def get_user_info(address: str) -> Optional[UserInfo]:
    return await UserInfo.where_eq(address=address).first()


@app.get("/user/{user_id}/permissions/incoming")
async def get_incoming_permission_requests(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.where_eq(authorizer=user_id).page(
        page=page, page_size=page_size
    )
    return permission_records


@app.get("/user/{user_id}/permissions/outgoing")
async def get_outgoing_permission_requests(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.where_eq(requestor=user_id).page(
        page=page, page_size=page_size
    )
    return permission_records


@app.get("/results/{result_id}")
async def get_result(result_id: str) -> Optional[Result]:
    return await Result.fetch(result_id).first()


@app.get("/user/{address}/results")
async def get_user_results(
    address: str, page: int = 1, page_size: int = 20
) -> List[Result]:
    return await Result.where_eq(owner=address).page(page=page, page_size=page_size)


@app.put("/user")
async def put_user_info(user_info: PutUserInfo) -> UserInfo:
    user_record = None
    if user_info.address:
        user_record = await UserInfo.where_eq(address=user_info.address).first()
        if user_record:
            user_record.username = user_info.username
            user_record.address = user_info.address
            user_record.bio = user_info.bio
            user_record.email = user_info.email
            user_record.link = user_info.link
            await user_record.save()
    if user_record is None:
        user_record = await UserInfo(
            username=user_info.username,
            address=user_info.address,
            bio=user_info.bio,
            email=user_info.email,
            link=user_info.link,
        ).save()
    return user_record


@app.get("/user/all")
async def get_all_user() -> List[UserInfo]:
    return await UserInfo.fetch_objects().all()


# ------------------>Kingsley requirements<---------------------------#


# ACCESS REQUEST endpoint
@app.get("/user/{user_id}/dataset/incoming")
async def incoming_data_request(user_id: str) -> List[Dataset]:
    """
    Requested data published by me - INCOMING
    """
    return await Dataset.where_eq(owner=user_id).all()


# requested data published by another user - OUTGOING
@app.get("/user/{user_id}/dataset/outgoing")
async def outgoing_data_request(user_id: str) -> List[Dataset]:
    """
    Requested data published by another user - OUTGOING
    """
    return await Dataset.where_eq(name=user_id).all()


@app.get("/user/{user_id}/notifications")
async def get_notification(user_id: str) -> List[Notification]:
    # requests permission for a whole dataset
    permission_records = await Permission.where_eq(
        authorizer=user_id, status=PermissionStatus.REQUESTED
    ).all()

    # this can result in many permissions being requested

    datasets_records = [
        await Dataset.where_eq(timeseriesIDs=ts_id.timeseriesID).all()
        for ts_id in permission_records
    ]

    # Group them by datasetID and also requestor!

    datasets_list = [element for row in datasets_records for element in row]
    notifications = []

    for record in datasets_list:
        notifications.append(
            Notification(
                type=NotificationType.PermissionRequest,
                message_text=user_id + " has requested to access " + record.name,
            )
        )
    return notifications


# POST /permissions
@app.post("/post/permissions")
async def post_permission(permissions: MultiplePermissions) -> MessageResponse:
    req = []
    for rec in permissions.permissions:
        req.append(
            Permission(
                timeseriesID=rec.timeseriesID,
                algorithmID=rec.algorithmID,
                authorizer=rec.authorizer,
                status=rec.status,
                executionCount=rec.executionCount,
                maxExecutionCount=rec.maxExecutionCount,
                requestor=rec.requestor,
            ).save()
        )
    await asyncio.gather(*req)

    return MessageResponse(response="Permissions Posted Successfully")


# Create a new permission as the authorizer
@app.post("/authorizer/post/permission")
async def post_authorizer_permission(
    permission: PostPermission,
) -> PermissionPostedResponse:
    authorizer_permission = await Permission(
        timeseriesID=permission.timeseriesID,
        algorithmID=permission.algorithmID,
        authorizer=permission.authorizer,
        status=permission.status,
        executionCount=permission.executionCount,
        maxExecutionCount=permission.maxExecutionCount,
        requestor=permission.requestor,
    ).save()

    return PermissionPostedResponse(
        sender=authorizer_permission.authorizer,
        permissionResponse=authorizer_permission,
    )


# POST /datasets/{dataset_id}/request
@app.post("/datasets/dataset_id/request")
async def dataset_request(dataset_req: UploadDatasetRequest) -> MessageResponse:
    posted_dataset = await Dataset(
        id_hash=dataset_req.id_hash,
        name=dataset_req.name,
        desc=dataset_req.desc,
        owner=dataset_req.owner,
        ownsAllTimeseries=dataset_req.ownsAllTimeseries,
        timeseriesIDs=dataset_req.timeseriesIDs,
    ).save()
    return MessageResponse(response=posted_dataset.id_hash + " posted Successfully")


# Create a new permission request as the requestor of a specific dataset


@app.post("/requestor/permission/dataset")
async def requestor_permission_dataset(
    permission: PostPermission,
) -> PermissionPostedResponse:
    requestor_dataset = await Permission(
        timeseriesID=permission.timeseriesID,
        algorithmID=permission.algorithmID,
        authorizer=permission.authorizer,
        status=permission.status,
        executionCount=permission.executionCount,
        maxExecutionCount=permission.maxExecutionCount,
        requestor=permission.requestor,
    ).save()
    return PermissionPostedResponse(
        sender=requestor_dataset.requestor, permissionResponse=requestor_dataset
    )


@app.delete("/clear/records")
async def empty_records() -> MessageResponse:
    await UserInfo.forget_all()
    await Timeseries.forget_all()
    await View.forget_all()
    await Dataset.forget_all()
    await Algorithm.forget_all()
    await Execution.forget_all()
    await Permission.forget_all()
    await Result.forget_all()
    return MessageResponse(response="All records are cleared")


@app.get("/views")
async def get_views(view_ids: List[str]) -> List[View]:
    return await View.fetch(view_ids).all()


@app.post("/event")
async def event(event: PostMessage):
    await fishnet_event(event)


@app.event(filters=API_MESSAGE_FILTER)
async def fishnet_event(event: PostMessage):
    print("fishnet_event", event)
    if event.content.type in [
        "Execution",
        "Permission",
        "Dataset",
        "Timeseries",
        "Algorithm",
    ]:
        if Record.is_indexed(event.item_hash):
            return
        cls: Record = globals()[event.content.type]
        record = await cls.from_post(event)
    else:  # amend
        if Record.is_indexed(event.content.ref):
            return
        record = await Record.fetch(event.content.ref).first()
    for inx in record.get_indices():
        inx.add_record(record)
