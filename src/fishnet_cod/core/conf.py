from aleph.sdk.chains.sol import SOLAccount, get_fallback_account
from pydantic import BaseSettings


class Settings(BaseSettings):
    MESSAGE_CHANNEL = "FISHNET_TEST_V1.14"
    """Name of the channel to use for the Fishnet network"""

    CONFIG_CHANNEL = "FISHNET_TEST_CONFIG_V1.14"
    """Name of the channel to use for the Fishnet network"""

    MESSAGES_KEY = [int(byte) for byte in get_fallback_account().private_key]
    """
    The private key of the Solana account as a uint8array.
    """

    MANAGER_PUBKEYS = [
        "5cyWHnWcqk8QpGntEWUnJAiSg8P78pnvs47WZd8jeHDH",  # Kingsley
        "fishbsxxtW2iRwBgihKZEWGv4EMZ47G6ypx3P22Nhqx",  # Brick indexer 2
    ]
    """List of public keys of the managers of the Fishnet channel"""

    TEST_CACHE = True
    """Whether to use the aleph.sdk.vm.TestVmCache or the aleph.sdk.vm.VmCache"""

    TEST_CHANNEL = False
    """Whether to use a new channel on each startup"""

    DISABLE_AUTH = False
    """Whether to disable authentication for the API"""

    EXECUTOR_MESSAGE_FILTER = [
        {
            "channel": MESSAGE_CHANNEL,
            "type": "POST",
            "post_type": ["Execution", "amend"],
        }
    ]
    """Filter for the messages to listen to for the executor"""

    API_MESSAGE_FILTER = [
        {
            "channel": MESSAGE_CHANNEL,
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
    """Filter for the messages to listen to for the API"""

    VM_URL_PATH = "https://aleph.sh/vm/{hash}"
    """URL to the VM load balancer"""

    VM_URL_HOST = "https://{hash_base32}.aleph.sh"
    """URL to the standard VM host"""

    class Config:
        env_prefix = "FISHNET_"
        case_sensitive = False
        env_file = ".env"


settings = Settings()
