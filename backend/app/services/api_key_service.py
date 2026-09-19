"""
ApiKeyService.

Authentication milestone change: `create_api_key` now generates the
raw key server-side (app/core/security.py) rather than accepting a
caller-supplied hash. The raw key is returned alongside the persisted
record ONLY at creation time — it is never stored and cannot be
recovered afterward.
"""

import uuid

from sqlalchemy.orm import Session

from app.core.security import generate_api_key, hash_api_key
from app.db.models import ApiKey
from app.repositories import ApiKeyRepository
from app.services.exceptions import NotFoundError


class ApiKeyService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = ApiKeyRepository(session)

    def create_api_key(self, tenant_id: uuid.UUID, label: str) -> tuple[ApiKey, str]:
        """Generate a new raw key, persist only its hash, and return
        both the record and the one-time raw key.

        Callers (the API layer) are responsible for making sure the raw
        key is included in the HTTP response and nowhere else —
        never logged, never re-fetchable.
        """
        raw_key = generate_api_key()
        key_hash = hash_api_key(raw_key)

        api_key = self.repository.create(tenant_id=tenant_id, label=label, key_hash=key_hash)
        self.session.commit()
        return api_key, raw_key

    def list_api_keys(self, tenant_id: uuid.UUID) -> list[ApiKey]:
        return self.repository.list_by_tenant(tenant_id)

    def get_api_key(self, tenant_id: uuid.UUID, api_key_id: uuid.UUID) -> ApiKey:
        api_key = self.repository.get_by_id(tenant_id, api_key_id)
        if api_key is None:
            raise NotFoundError(f"ApiKey '{api_key_id}' not found.")
        return api_key

    def revoke_api_key(self, tenant_id: uuid.UUID, api_key_id: uuid.UUID) -> ApiKey:
        api_key = self.repository.update(tenant_id, api_key_id, is_active=False)
        if api_key is None:
            raise NotFoundError(f"ApiKey '{api_key_id}' not found.")

        self.session.commit()
        return api_key
