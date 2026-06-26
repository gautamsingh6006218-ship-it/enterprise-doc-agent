"""
api/auth.py

What problem does this solve?
- Every route needs tenant_id, user_id, and user_roles to enforce RBAC.
  Without centralised auth, each route would decode JWT independently —
  duplicated logic that would drift and introduce security gaps.

Why FastAPI Depends() instead of middleware?
- Middleware runs for every request including health checks and docs.
  Depends() attaches only to routes that declare it — cleaner and easier
  to override in tests via app.dependency_overrides.

JWT structure expected:
  {
    "sub":       "user-123",          # user_id
    "tenant_id": "acme",              # multi-tenant partition
    "roles":     ["hr", "legal"],     # access_roles
    "exp":       1234567890           # expiry (validated by PyJWT)
  }

Environment variables:
  JWT_SECRET   — HMAC-SHA256 signing secret (required in production)
  JWT_ALGORITHM — default "HS256"

Why HS256 over RS256?
- HS256 is simpler for single-service deployments. Switch to RS256 when
  multiple services need to verify tokens independently without sharing a secret.
"""

import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent.retrieval.models import RBACContext

_bearer = HTTPBearer(auto_error=False)

_JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
_JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def get_rbac_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> RBACContext:
    """
    What problem does this solve?
    - Decodes the Bearer JWT and builds an RBACContext for every route
      that declares `rbac: RBACContext = Depends(get_rbac_context)`.

    Why return RBACContext directly (not a dict)?
    - RBACContext is the type RetrievalService.query() expects. Returning
      it here means routes pass it straight through — no conversion needed.

    Why raise 401 on missing token?
    - No anonymous access to enterprise documents. Every request must
      identify its tenant and user for RBAC filtering to work.

    Why raise 403 on missing tenant_id claim?
    - A valid JWT without a tenant_id is a misconfigured token. Returning
      403 (Forbidden) rather than 401 (Unauthenticated) signals the token
      was accepted but the claim required for authorisation is absent.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            _JWT_SECRET,
            algorithms=[_JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token missing required claim: tenant_id",
        )

    return RBACContext(
        tenant_id=tenant_id,
        user_id=payload.get("sub", "unknown"),
        user_roles=payload.get("roles", []),
    )


def make_test_token(
    tenant_id: str = "test-tenant",
    user_id: str = "user-1",
    roles: list[str] | None = None,
    secret: str = _JWT_SECRET,
) -> str:
    """
    What problem does this solve?
    - Tests need a valid JWT without a running auth server.
      This helper creates one using the same secret as get_rbac_context,
      so tests can call authenticated routes without mocking the JWT layer.

    Only used in tests — not imported by any production route.
    """
    import time
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "exp": int(time.time()) + 86400,  # 24 hours
    }
    return jwt.encode(payload, secret, algorithm="HS256")
