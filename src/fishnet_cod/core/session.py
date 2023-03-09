from aleph.sdk import AuthenticatedAlephClient
from aleph.sdk.conf import settings
from aleph.sdk.chains.sol import get_fallback_account

authorized_session = AuthenticatedAlephClient(
    get_fallback_account(),
    settings.API_HOST,
)
