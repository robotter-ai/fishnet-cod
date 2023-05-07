from aleph.sdk.exceptions import BadSignatureError
from fastapi import APIRouter, HTTPException

from ..api_model import BearerTokenResponse, TokenChallengeResponse
from ..security import AuthTokenManager, NotAuthorizedError, SupportedChains

router = APIRouter(
    prefix="/authorization",
    tags=["authorization"],
    responses={404: {"description": "Not found"}},
)


@router.post("/challenge")
async def create_challenge(
    pubkey: str, chain: SupportedChains
) -> TokenChallengeResponse:
    challenge = AuthTokenManager.get_challenge(pubkey=pubkey, chain=chain)
    return TokenChallengeResponse(
        pubkey=challenge.pubkey,
        chain=challenge.chain,
        challenge=challenge.challenge,
        valid_til=challenge.valid_til,
    )


@router.post("/solve")
async def solve_challenge(
    pubkey: str, chain: SupportedChains, signature: str
) -> BearerTokenResponse:
    try:
        auth = AuthTokenManager.solve_challenge(
            pubkey=pubkey, chain=chain, signature=signature
        )
        return BearerTokenResponse(
            pubkey=auth.pubkey,
            chain=auth.chain,
            token=auth.token,
            valid_til=auth.valid_til,
        )
    except (BadSignatureError, ValueError):
        raise HTTPException(403, "Challenge failed")
    except TimeoutError:
        raise HTTPException(403, "Challenge timeout")


@router.post("/refresh")
async def refresh_token(token: str) -> BearerTokenResponse:
    try:
        auth = AuthTokenManager.refresh_token(token)
    except TimeoutError:
        raise HTTPException(403, "Token expired")
    except NotAuthorizedError:
        raise HTTPException(403, "Not authorized")
    return BearerTokenResponse(
        pubkey=auth.pubkey, chain=auth.chain, token=auth.token, valid_til=auth.valid_til
    )


@router.post("/logout")
async def logout(token: str):
    AuthTokenManager.remove_token(token)
