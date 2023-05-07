import time
from enum import Enum
from typing import Dict, Optional

from aleph.sdk.chains.ethereum import verify_signature as verify_signature_eth
from aleph.sdk.chains.sol import verify_signature as verify_signature_sol
from fastapi import HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from nacl.bindings.randombytes import randombytes
from pydantic import BaseModel
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN


class SupportedChains(Enum):
    Solana = "SOL"
    Ethereum = "ETH"


class AuthInfo(BaseModel):
    pubkey: str
    chain: SupportedChains
    valid_til: int


class WalletAuth(AuthInfo):
    challenge: str
    _token: Optional[str]

    def __init__(self, pubkey: str, chain: SupportedChains, ttl: int = 60):
        challenge = f"Authentication challenge {randombytes(64).hex()}"
        valid_til = int(time.time()) + ttl  # 60 seconds
        super().__init__(pubkey=pubkey, chain=chain, valid_til=valid_til, challenge=challenge)  # type: ignore

    @property
    def token(self):
        if self.expired:
            raise TimeoutError("Token Expired")
        return self._token

    @property
    def expired(self):
        return int(time.time() > self.valid_til)

    def solve_challenge(self, signature: str):
        if self.expired:
            raise TimeoutError("Challenge Expired")

        if self.chain == SupportedChains.Solana:
            verify_signature_sol(
                signature=signature, public_key=self.pubkey, message=self.challenge
            )
        elif self.chain == SupportedChains.Ethereum:
            verify_signature_eth(
                signature=signature, public_key=self.pubkey, message=self.challenge
            )
        else:
            raise NotImplementedError(
                f"{self.chain} has no verification function implemented"
            )

        self.refresh_token()

    def refresh_token(self, ttl: int = 60 * 60):
        self._token = randombytes(64).hex()
        self.valid_til = int(time.time()) + ttl  # 1 hour


class NotAuthorizedError(Exception):
    pass


class AuthTokenManager:
    """
    A self-updating dictionary keeping track of all authentication tokens and signature challenges.
    """

    __challenges: Dict[str, WalletAuth] = {}
    """Keeps track of all solved and unsolved challenges. Keys have format `<pubkey>-<chain>`."""
    __auths: Dict[str, WalletAuth] = {}
    """Maps all authentication tokens to their respective `WalletAuth` objects."""

    @classmethod
    def get_token(cls, pubkey: str, chain: SupportedChains) -> str:
        try:
            return cls.__challenges[pubkey + "-" + str(chain)].token
        except (TimeoutError, AttributeError) as e:
            if e is TimeoutError:
                cls.remove_challenge(pubkey, chain)
            if e is AttributeError:
                raise NotAuthorizedError("Not authorized") from e
            raise e

    @classmethod
    def get_auth(cls, token: str) -> WalletAuth:
        auth = cls.__auths.get(token)
        if not auth:
            raise NotAuthorizedError("Not authorized")
        if auth.expired:
            cls.remove_challenge(pubkey=auth.pubkey, chain=auth.chain)
            raise TimeoutError("Token expired")
        return auth

    @classmethod
    def get_challenge(cls, pubkey: str, chain: SupportedChains) -> WalletAuth:
        auth = cls.__challenges.get(pubkey + "-" + str(chain))
        if auth is None or int(time.time()) > auth.valid_til:
            auth = WalletAuth(pubkey=pubkey, chain=chain)
            cls.__challenges[pubkey + "-" + str(chain)] = auth
        return auth

    @classmethod
    def solve_challenge(
        cls, pubkey: str, chain: SupportedChains, signature: str
    ) -> WalletAuth:
        auth = cls.get_challenge(pubkey, chain)
        auth.solve_challenge(signature)
        return auth

    @classmethod
    def remove_challenge(cls, pubkey: str, chain: SupportedChains):
        cls.__challenges.pop(pubkey + "-" + str(chain))

    @classmethod
    def remove_auth(cls, auth: WalletAuth):
        cls.remove_challenge(auth.pubkey, auth.chain)
        cls.__auths.pop(auth.token)

    @classmethod
    def remove_token(cls, token: str):
        auth = cls.__auths.get(token)
        if auth:
            cls.remove_auth(auth)

    @classmethod
    def clear_expired(cls):
        now = int(time.time())
        for challenge in cls.__challenges.values():
            if now > challenge.valid_til:
                cls.remove_auth(challenge)

    @classmethod
    def refresh_token(cls, token: str, ttl: int = 60 * 60):
        auth = cls.__auths.get(token)
        if not auth:
            raise NotAuthorizedError("Not authorized")
        if auth.expired:
            raise TimeoutError("Token expired")
        auth.refresh_token(ttl)
        return auth


class AuthTokenChecker:
    def __init__(
        self,
        auth_manager: AuthTokenManager,
        challenge_endpoint: str,
        token_endpoint: str,
    ):
        self.auth_manager = auth_manager
        self.challenge_endpoint = challenge_endpoint
        self.token_endpoint = token_endpoint

    def __call__(self, request: Request) -> Optional[WalletAuth]:
        authorization = request.headers.get("Authorization")
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
            )
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid authentication credentials",
            )

        try:
            return self.auth_manager.get_auth(credentials)
        except (TimeoutError, NotAuthorizedError) as e:
            if e is TimeoutError:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED, detail="Token expired"
                )
            if e is NotAuthorizedError:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail=f"Not authorized. Request a challenge to sign from f{self.challenge_endpoint} "
                    f"and retrieve a token from f{self.token_endpoint}.",
                )
            raise e
