"""OAuth2 / JWT authentication & broker SHA-256 auth-code signature helpers."""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("FINPULSE_JWT_SECRET", "finpulse-dev-hmac-sha256-secret-key-2026")
JWT_ALGORITHM = "HS256"
DEFAULT_CLIENT_ID = os.getenv("FINPULSE_CLIENT_ID", "finpulse-client")
DEFAULT_CLIENT_SECRET = os.getenv("FINPULSE_CLIENT_SECRET", "finpulse-secret-2026")

bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(
    subject: str,
    scope: str = "rag:read rag:write quant:execute agent:run",
    expires_in: int = 3600,
) -> str:
    """Create a signed HS256 JWT access token."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "scope": scope,
        "iat": now,
        "exp": now + expires_in,
        "iss": "finpulse-ai-agent",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_access_token(token: str) -> dict[str, Any]:
    """Decode and validate an HS256 JWT access token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], issuer="finpulse-ai-agent")
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc


def authenticate_client(client_id: str, client_secret: str) -> bool:
    """Constant-time verification of OAuth2 client credentials."""
    id_ok = hmac.compare_digest(client_id.encode("utf-8"), DEFAULT_CLIENT_ID.encode("utf-8"))
    sec_ok = hmac.compare_digest(client_secret.encode("utf-8"), DEFAULT_CLIENT_SECRET.encode("utf-8"))
    return id_ok and sec_ok


def compute_oauth2_app_id_hash(app_id: str, secret_key: str) -> str:
    """Compute SHA-256 `appIdHash` used in broker OAuth2 auth_code token exchanges."""
    if not app_id.strip() or not secret_key.strip():
        raise ValueError("app_id and secret_key must be non-empty")
    raw = f"{app_id.strip()}:{secret_key.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_jwt_claims(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    """FastAPI dependency enforcing Bearer JWT authentication."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token",
        )
    return verify_access_token(credentials.credentials)
