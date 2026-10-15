"""
Access control (Phase K)
========================

Simple bearer/header token guard for sensitive endpoints. Defaults:

- Token comes from the environment (`VITALS_API_TOKEN`) or is passed at
  app construction for tests. (No tokens, literal passwords or default
  credentials live in the source.)
- When no token is configured, the guard is DISABLED with a loud startup
  warning - appropriate for local grids/demo, NOT for shared deployments.

Nonclinical-use and consent/disclaimer banners surface in app metadata so
the demo cannot accidentally present itself as medical software.
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-API-Token"


def make_token() -> str:
    """Generate a cryptographically-random local token (callers persist it)."""
    return secrets.token_urlsafe(32)


class TokenGuard:
    """Dependency-usable token verifier."""

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self._enforced = bool(token)
        if not self._enforced:
            logger.warning(
                "API token NOT set (VITALS_API_TOKEN): access control DISABLED. "
                "Suitable for local demos only - never expose without a token."
            )

    @property
    def enforced(self) -> bool:
        return self._enforced

    def __call__(self, x_api_token: Optional[str] = Header(default=None)) -> None:
        if not self._enforced:
            return
        if not x_api_token or not secrets.compare_digest(self.token or "", x_api_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API token",
            )
