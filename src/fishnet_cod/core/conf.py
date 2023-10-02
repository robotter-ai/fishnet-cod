from pydantic import BaseSettings


class Settings(BaseSettings):
    MESSAGE_CHANNEL = "FISHNET_TEST_V1.16"
    """Name of the channel to use for the Fishnet network"""

    CONFIG_CHANNEL = None
    """Name of the channel to use for the Fishnet network"""

    MANAGER_PUBKEY = "FishebefjVYAkRWvfdVqvgfzao9fx8R1S8fiwYF23zEq"
    """Pubkey of the manager account"""

    DATABASE_PATH = "fishnet.db"

    TEST_CACHE = True
    """Whether to use the aleph.sdk.vm.TestVmCache or the aleph.sdk.vm.VmCache"""

    TEST_CHANNEL = False
    """Whether to use a new channel on each startup"""

    API_MESSAGE_FILTER = [
        {
            "channel": MESSAGE_CHANNEL,
            "type": "POST",
            "post_type": [
                "Permission",
                "Dataset",
                "DatasetSlice",
                "Timeseries",
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
if settings.CONFIG_CHANNEL is None:
    settings.CONFIG_CHANNEL = settings.MESSAGE_CHANNEL + "_CONFIG"
