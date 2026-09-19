"""
AuthService.

Verifies a raw API key presented by a caller and resolves it to an
active tenant. This is the only place request-time authentication
logic lives — the API layer's auth dependency (app/api/deps.py) calls
this and nothing else.

Security note: every failure path here — malformed key, unknown hash,
revoked key, deactivated tenant — raises the SAME AuthenticationError
with the SAME generic message. This is deliberate: the caller must
never be able to distinguish "that key doesn't exist" from "that key
exists but is revoked" from "that tenant is deactivated". Giving any
of those away would let an attacker map out valid-but-revoked keys or
enumerate tenant state.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.security import API_KEY_PREFIX, hash_api_key
from app.db.models import ApiKey, Tenant
from app.repositories import ApiKeyRepository, TenantRepository
from app.services.exceptions import AuthenticationError

_GENERIC_AUTH_ERROR = "Invalid or missing API key."


class AuthService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self._api_key_repository = ApiKeyRepository(session)
        self._tenant_repository = TenantRepository(session)

    def verify_api_key(self, raw_key: str) -> uuid.UUID:
        """Verify a raw API key and return the resolved, active tenant_id.

        Raises AuthenticationError (generic message, see module
        docstring) for every failure case:
        - key doesn't start with the expected prefix
        - no ApiKey row matches the computed hash
        - the matching ApiKey is inactive (revoked)
        - the owning Tenant is inactive (deactivated)
        """
        if not raw_key or not raw_key.startswith(API_KEY_PREFIX):
            raise AuthenticationError(_GENERIC_AUTH_ERROR)

        key_hash = hash_api_key(raw_key)
        api_key: ApiKey | None = self._api_key_repository.get_by_key_hash(key_hash)

        if api_key is None or not api_key.is_active:
            raise AuthenticationError(_GENERIC_AUTH_ERROR)

        tenant: Tenant | None = self._tenant_repository.get_by_id(api_key.tenant_id)

        if tenant is None or not tenant.is_active:
            raise AuthenticationError(_GENERIC_AUTH_ERROR)

        self._touch_last_used(api_key)

        return tenant.id

    def _touch_last_used(self, api_key: ApiKey) -> None:
        """Best-effort update of last_used_at on successful verification.

        Failure to record this should never block the request that
        triggered it — this is observability, not a security control.
        """
        try:
            api_key.last_used_at = datetime.now(timezone.utc)
            self.session.commit()
        except Exception:
            # Never let a last_used_at write failure break authentication.
            self.session.rollback()
