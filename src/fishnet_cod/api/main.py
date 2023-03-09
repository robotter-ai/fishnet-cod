import asyncio
import logging
import os
from typing import List, Optional, Tuple
from datetime import datetime
from os import listdir, getenv

from aleph.sdk import AuthenticatedAlephClient
from aleph.sdk.chains.sol import get_fallback_account
from aleph.sdk.conf import settings
from aleph_message.models import PostMessage
from .api_model import (
    UploadTimeseriesRequest,
    UploadDatasetRequest,
    UploadAlgorithmRequest,
    RequestExecutionRequest,
    RequestExecutionResponse,
    PutUserInfo,
)
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
from ..core.constants import FISHNET_MESSAGE_CHANNEL

logger = logging.getLogger(__name__)

logger.debug("import aleph_client")
from aleph.sdk.vm.cache import VmCache, TestVmCache
from aleph.sdk.vm.app import AlephApp

logger.debug("import aars")
from aars import AARS, Record

logger.debug("import fastapi")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger.debug("import project modules")

logger.debug("imports done")

http_app = FastAPI()

origins = ["*"]

http_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEST_CACHE = getenv("TEST_CACHE")
if TEST_CACHE is not None and TEST_CACHE.lower() == "true":
    cache = TestVmCache()
else:
    cache = VmCache()
app = AlephApp(http_app=http_app)
session = AuthenticatedAlephClient(get_fallback_account(), settings.API_HOST)
aars = AARS(channel=FISHNET_MESSAGE_CHANNEL, cache=cache, session=session)


async def re_index():
    logger.info("API re-indexing")
    await asyncio.wait_for(AARS.sync_indices(), timeout=None)
    logger.info("API re-indexing done")


@http_app.on_event("startup")
async def startup():
    await re_index()


@app.get("/")
async def index():
    if os.path.exists("/opt/venv"):
        opt_venv = list(listdir("/opt/venv"))
    else:
        opt_venv = []
    # TODO: Show actual config instead of endpoints
    return {
        "vm_name": "fishnet_api",
        "endpoints": [
            "/docs",
        ],
        "files_in_volumes": {
            "/opt/venv": opt_venv,
        },
    }


@app.get("/indices")
async def indices():
    ts = [list(index.hashmap.items()) for index in Timeseries.get_indices()]
    ui = [list(index.hashmap.items()) for index in UserInfo.get_indices()]
    ds = [list(index.hashmap.items()) for index in Dataset.get_indices()]
    al = [list(index.hashmap.items()) for index in Algorithm.get_indices()]
    ex = [list(index.hashmap.items()) for index in Execution.get_indices()]
    pe = [list(index.hashmap.items()) for index in Permission.get_indices()]
    return ts, ui, ds, al, ex, pe


@app.get("/indices/reindex")
async def reindex():
    await re_index()


@app.get("/datasets")
async def get_datasets(
    view_as: Optional[str] = None,
    by: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[Tuple[Dataset, Optional[DatasetPermissionStatus]]]:
    """
    Get all datasets. Returns a list of tuples of datasets and their permission status for the given `view_as` user.
    If `view_as` is not given, the permission status will be `none` for all datasets.
    :param view_as: address of the user to view the datasets as and give additional permission information
    :param by: address of the dataset owner to filter by
    :param page_size: size of the pages to fetch
    :param page: page number to fetch
    """
    if by:
        datasets = await Dataset.where_eq(owner=by).all()
    else:
        datasets = await Dataset.fetch_objects().page(page=page, page_size=page_size)

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

    returned_datasets: List[Tuple[Dataset, DatasetPermissionStatus]] = []
    for rec in datasets:
        dataset_permissions = []
        for ts_id in rec.timeseriesIDs:
            dataset_permissions.extend(
                list(filter(lambda x: x.timeseriesID == ts_id, permissions))
            )
        if not dataset_permissions:
            returned_datasets.append((rec, DatasetPermissionStatus.NOT_REQUESTED))
            continue

        permission_status = [perm_rec for perm_rec in dataset_permissions]
        if all(status == PermissionStatus.GRANTED for status in permission_status):
            returned_datasets.append((rec, DatasetPermissionStatus.GRANTED))
        elif PermissionStatus.DENIED in permission_status:
            returned_datasets.append((rec, DatasetPermissionStatus.DENIED))
        elif PermissionStatus.REQUESTED in permission_status:
            returned_datasets.append((rec, DatasetPermissionStatus.REQUESTED))
    return returned_datasets


@app.get("/user/{userAddress}/permissions/incoming")
async def in_permission_requests(
    userAddress: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    if page is None:
        page = 1
    if page_size is None:
        page_size = 20
    permission_records = await Permission.where_eq(authorizer=userAddress).page(
        page=page, page_size=page_size
    )
    return permission_records


@app.get("/user/{userAddress}/permissions/outgoing")
async def out_permission_requests(
    userAddress: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Permission]:
    permission_records = await Permission.where_eq(requestor=userAddress).page(
        page=page, page_size=page_size
    )
    return permission_records


@app.get("/algorithms")
async def query_algorithms(
    id: Optional[str] = None,
    name: Optional[str] = None,
    by: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> List[Algorithm]:
    """
    - query for own algos
    - query other algos
    - page, page_size and by
    """

    if id:
        algo_request = Algorithm.fetch(id)
    elif name or by:
        algo_request = Algorithm.where_eq(name=name, owner=by)
    else:
        algo_request = Algorithm.fetch_objects()
    return await algo_request.page(page=page, page_size=page_size)


####------------>>Debug<<----------------------##


@app.get("/timeseries/debug/get")
async def get_timeseries():
    return await Timeseries.fetch_objects().all()


@app.post("/post/debug/datasets")
async def post_ds():
    await asyncio.sleep(1)
    await Dataset(
        name="Dataset004",
        owner="Ds_owner004",
        desc="Desc for 004",
        availabe=False,
        ownsAllTimeseries=True,
        timeseriesIDs=[
            "778dec63cf37cfb23a4ea53ef9c67e3931a5536687b44cc30148d16913aa02ea",
            "d481de53da312f875a550e695d7a54cfad8b65e68b8e99f44617c2b725a99d19",
        ],
        views=[
            "5ef639d13325f2628852c543a7834c23c966f1c0905dcc1aa7f8bf806862e150",
            "e339f1500070cd82cec452a9cc2fb52a2fdca2443f3b188a80f0aee8ff4c62f7",
        ],
    ).save()
    return "Posted"


@app.post("/post/debug/views")
async def post_views():
    await asyncio.sleep(1)
    await View(
        startTime=int(datetime.timestamp(datetime.now())),
        endTime=int(datetime.timestamp(datetime.now())),
        granularity=Granularity.YEAR,
        values={
            "d481de53da312f875a550e695d7a54cfad8b65e68b8e99f44617c2b725a99d19": [
                (1, 3.42231)
            ]
        },
    ).save()
    return "Posted"


@app.get("/get/debug/views")
async def get_Views():
    return await View.fetch_objects().all()


@app.post("/post/debug/Permission")
async def post_permission():
    await asyncio.sleep(1)
    await Permission(
        timeseriesID="c470221cf21e4f6fd8a1bf329532cad886aaac330915419033a6e17433bb3bc2",
        algorithmID="60b5e790149d12d0f4b1b7af0c27f3eeb9fa0d56edb7bd56832ef536e36c6115",
        authorizer="Owner_of_TimeseriesId004",
        status=PermissionStatus.DENIED,
        executionCount=0,
        maxExecutionCount=-1,
        requestor="Wa005",
    ).save()
    return "Posted"


@app.get("/get/debug/permission")
async def get_permission():
    return await Permission.fetch_objects().all()


@app.post("/post/debug/results")
async def post_result():
    await asyncio.sleep(1)
    await Result(
        executionID="fbd8b2289f01740fde2251ad3de7f349396f523880e4a0887981046910c87cdb",
        data="Result data ",
    ).save()
    return "Posted"


@app.get("/get/debug/result")
async def get_results():
    return await Result.fetch_objects().all()


@app.post("/post/debug/execution")
async def post_execution():
    await asyncio.sleep(1)
    await Execution(
        algorithmID="ef72bd7720ecaf357d1d39077ad199817def6a1811e828934a61d33996b98db7",
        datasetID="8caa4855024e68d19549d924d14fdb39c795ab649296401a077833e3094c6c89",
        owner="Executor003",
        status=ExecutionStatus.REQUESTED,
        resultID="130225c09d9372a6e007232d5e3a4ca2f6612eb4bdd89f0dda1a8c71b6d2f84a",
        params={"param1": "This is param1", "param2": "This is param2"},
    ).save()
    return "Posted"


@app.get("/get/debug/executions")
async def get_execution():
    return await Execution.fetch_objects().all()


@app.post("/post/debug/Algorithms")
async def post_algo():
    await Algorithm(
        name="Al004",
        desc="Desc for Al004",
        owner="Owner for Al004",
        code="""
def run(df: pd.DataFrame):
    return df.sum(axis=0)
""",
    ).save()


@app.get("/get/debug/algorithm")
async def get_algo():
    return await Algorithm.fetch_objects().all()


@app.post("/delete/debug/records")
async def delete_records():
    records = Execution.fetch_objects()
    async for i in records:
        await i.forget()

    return "Delete the records"


####------------>>Debug<<----------------------##


# @app.get('/timeviews')
# # async def generate_view(view_rec: List[ViewsRequest])->List[View]:
#     #get all the timeseries
#     timeseries_rec = [rec.timeseries_ids for rec in view_rec]
#     ts_ids_np = np.array(timeseries_rec)
#     ts_ids_lists = np.hstack(ts_ids_np)
#     ts_ids_unique = np.unique(ts_ids_lists)


# ts_ids_lst = list(ts_ids_unique)
# filter each value which are inside the timeframe
# for rec in view_rec:
# rec.start_time

# Calculate min and max ]
# normalize to 0 and 1
# where min =0 and max =1
# round to 2 decimals
# return


@app.put("/userInfo")
async def user_info(user_info: PutUserInfo) -> UserInfo:
    if user_info.address:
        user_records = await UserInfo.where_eq(address=user_info.address).first()
    else:
        user_records = await UserInfo(
            datasetIDs=user_info.datasetIDs,
            executionIDs=user_info.executionIDs,
            algorithmIDs=user_info.algorithmIDs,
            username=user_info.username,
            address=user_info.address,
            bio=user_info.bio,
            email=user_info.email,
            link=user_info.link,
        ).save()
    return user_records


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
    return await execution_requests.page(
        page=page, page_size=page_size
    )


@app.get("/user/{address}/results")
async def get_user_results(
    address: str,
    page: int = 1,
    page_size: int = 20,
) -> List[Result]:
    return await Result.where_eq(owner=address).page(page=page, page_size=page_size)


@app.get("/executions/{execution_id}/possible_execution_count")
async def get_possible_execution_count(execution_id: str) -> int:
    """
    THIS IS AN OPTIONAL ENDPOINT. It is a nice challenge to implement this endpoint, as the code is not trivial, and
    it might be still good to have this code in the future.

    This endpoint returns the number of times the execution can be executed.
    This is the maximum number of times
    the algorithm can be executed on the dataset, given the permissions of each timeseries.
    It can only be executed
    as many times as the least available timeseries can be executed.
    """

    return -1


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
    dataset = await Dataset.fetch(execution.datasetID).first()

    if dataset.owner == execution.owner and dataset.ownsAllTimeseries:
        execution.status = ExecutionStatus.PENDING
        return RequestExecutionResponse(
            execution=await Execution(**execution.dict()).save(),
            permissionRequests=None,
            unavailableTimeseries=None,
        )

    requested_timeseries = await Timeseries.fetch(dataset.timeseriesIDs).all()
    permissions = {
        permission.timeseriesID: permission
        for permission in await Permission.where_eq(
            timeseriesID=dataset.timeseriesIDs, requestor=execution.owner
        ).all()
    }
    requests = []
    unavailable_timeseries = []
    for ts in requested_timeseries:
        if ts.owner == execution.owner:
            continue
        if not ts.available:
            unavailable_timeseries.append(ts)
        if requested_timeseries:
            continue
        if ts.id_hash not in permissions:
            requests.append(
                Permission(
                    timeseriesID=ts.id_hash,
                    algorithmID=execution.algorithmID,
                    owner=ts.owner,
                    requestor=execution.owner,
                    status=PermissionStatus.REQUESTED,
                    executionCount=0,
                    maxExecutionCount=1,
                ).save()
            )
        else:
            permission = permissions[ts.id_hash]
            needs_update = False
            if permission.status == PermissionStatus.DENIED:
                permission.status = PermissionStatus.REQUESTED
                needs_update = True
            if permission.maxExecutionCount <= permission.executionCount:
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
            unavailableTimeseries = None,
            permissionRequests = None,
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


filters = [
    {
        "channel": aars.channel,
        "type": "POST",
        "post_type": [
            "Execution",
            "Permission",
            "Dataset",
            "Timeseries",
            "Algorithm",
            "amend",
        ],
    }
]


@app.event(filters=filters)
async def fishnet_event(event: PostMessage):
    print("fishnet_event", event)
    if event.content.type in [
        "Execution",
        "Permission",
        "Dataset",
        "Timeseries",
        "Algorithm",
    ]:
        cls: Record = globals()[event.content.type]
        record = await cls.from_post(event)
    else:
        record = Record.fetch(event.content.ref)
    [index.add_record(record) for index in record.get_indices()]
