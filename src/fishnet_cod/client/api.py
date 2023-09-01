from typing import Optional

import requests
from aleph.sdk.account import AccountFromPrivateKey


class FishNetClient:
    token: str

    def __init__(self, account: AccountFromPrivateKey, fishnet_api: str = "https://api.fishnet.tech"):
        self.fishnet_api = fishnet_api
        self.login(account)

    def login(self, account: AccountFromPrivateKey):
        """
        Login to the FishNet network.

        :param account: The account to login with.
        :return: The JWT token.
        """
        challenge = requests.post(f"{self.fishnet_api}/authorization/challenge").json()
        signature = account.sign_raw(bytes(challenge["challenge"]))
        resp = requests.post(
            f"{self.fishnet_api}/authorization/solve",
            json={
                "address": account.get_public_key(),
                "chain": account.CHAIN,
                "signature": signature,
            },
        )
        resp.raise_for_status()
        self.token = resp.json()["token"]

    # TODO: try a generator: https://pypi.org/project/openapi-python-client/
