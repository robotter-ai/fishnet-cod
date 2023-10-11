from typing import Optional

import base58
from aleph.sdk.chains.sol import generate_key
from pydantic import BaseSettings


class Settings(BaseSettings):
    MESSAGE_CHANNEL: str = "FISHNET_TEST_V1.16"
    """Name of the channel to use for the Fishnet network"""

    CONFIG_CHANNEL: Optional[str] = None
    """Name of the channel to use for the Fishnet network"""

    MESSAGES_KEY: Optional[str] = None

    MANAGER_PUBKEY: str = "FishebefjVYAkRWvfdVqvgfzao9fx8R1S8fiwYF23zEq"
    """Pubkey of the manager account"""

    TEST_CACHE: bool = True
    """Whether to use the aleph.sdk.vm.TestVmCache or the aleph.sdk.vm.VmCache"""

    TEST_CHANNEL: bool = False
    """Whether to use a new channel on each startup"""

    API_MESSAGE_FILTER = [
        {
            "channel": MESSAGE_CHANNEL,
            "type": "POST",
            "post_type": [
                "Permission",
                "Dataset",
                "Timeseries",
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

    DATA_PATH = "/app/data"

    class Config:
        env_prefix = "FISHNET_"
        case_sensitive = False
        env_file = ".env"


settings = Settings()
if settings.CONFIG_CHANNEL is None:
    settings.CONFIG_CHANNEL = settings.MESSAGE_CHANNEL + "_CONFIG"


# parse keys from hex, base58 or uint8 array
def parse_key(key: str) -> str:
    if key.startswith("0x"):
        return key[2:]
    elif key.startswith("[") and key.endswith("]"):
        raw = bytes([int(x) for x in key[1:-1].split(",")])[:32]
        return raw.hex()
    else:
        try:
            return base58.b58decode(key).hex()
        except ValueError:
            return key


if settings.MESSAGES_KEY is None:
    settings.MESSAGES_KEY = generate_key().hex()
else:
    settings.MESSAGES_KEY = parse_key(settings.MESSAGES_KEY)
