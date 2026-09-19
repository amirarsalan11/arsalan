"""
ApiKeyRepository.

CRUD-foundation only — no hashing, no verification logic. `create` takes
`key_hash` as-is; the Authentication milestone owns how that hash is
produced and checked.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ApiKey
from app.repositories.base import BaseRepository


class ApiKeyRepository(BaseRepository[ApiKey]):
    model = ApiKey

    def create(self, tenant_id: uuid.UUID, label: str, key_hash: str) -> ApiKey:
        api_key = ApiKey(tenant_id=tenant_id, label=label, key_hash=key_hash)
        self.session.add(api_key)
        self.session.flush()
        return api_key

    def get_by_key_hash(self, key_hash: str) -> ApiKey | None:
        """Domain-specific lookup, needed later by Authentication to
        resolve a raw key's hash back to a tenant. Not tenant-scoped by
        definition, since the caller doesn't know the tenant yet — that's
        precisely what this lookup determines. Not used by this milestone's
        CRUD-foundation endpoints, included now because ApiKey is the only
        model where "look up without knowing the tenant first" is a real
        future need."""
        stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
        return self.session.execute(stmt).scalar_one_or_none()
