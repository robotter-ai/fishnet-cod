FISHNET_MESSAGE_CHANNEL = "FISHNET_TEST_V1.14"
FISHNET_DEPLOYMENT_CHANNEL = "FISHNET_TEST_DEPLOYMENT_V1.14"
FISHNET_MANAGER_PUBKEYS = [
    "5cyWHnWcqk8QpGntEWUnJAiSg8P78pnvs47WZd8jeHDH",  # Kingsley
    "fishbsxxtW2iRwBgihKZEWGv4EMZ47G6ypx3P22Nhqx",  # Brick indexer 2
]

EXECUTOR_PATH = "../executor"
EXECUTOR_MESSAGE_FILTER = [
    {
        "channel": FISHNET_MESSAGE_CHANNEL,
        "type": "POST",
        "post_type": ["Execution", "amend"],
    }
]

API_PATH = "../api"
API_MESSAGE_FILTER = [
    {
        "channel": FISHNET_MESSAGE_CHANNEL,
        "type": "POST",
        "post_type": [
            "Execution",
            "Permission",
            "Dataset",
            "Timeseries",
            "Algorithm",
            "Result",
            "View",
            "UserInfo",
            "amend",
        ],
    }
]

VM_URL_PATH = "https://aleph.sh/vm/{hash}"
VM_URL_HOST = "https://{hash_base32}.aleph.sh"
