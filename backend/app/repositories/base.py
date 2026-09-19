"""
Generic base repository.

Per approval: shared only for genuinely common, structurally identical
operations — get_by_id and update. Domain-specific queries (e.g.
"get tenant by slug", "list active materials") are NOT pulled into this
base class; they live on the specific repository that needs them, to
avoid over-generalizing.

Every method that touches a tenant-scoped table requires tenant_id and
filters by it directly in the WHERE clause — callers can never
accidentally omit the tenant filter and leak rows across tenants.
"""

from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    """Shared get_by_id / update logic for tenant-scoped models.

    Subclasses must set `model` to the SQLAlchemy model class they wrap.
    This base class assumes the model has both an `id` and a `tenant_id`
    column. `TenantRepository` does NOT inherit from this base, since
    Tenant is the root entity and has no tenant_id of its own — see
    TenantRepository for its own get_by_id/update implementations.
    """

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, tenant_id: Any, entity_id: Any) -> ModelT | None:
        """Fetch a single row scoped to the given tenant, or None."""
        stmt = select(self.model).where(
            self.model.id == entity_id,
            self.model.tenant_id == tenant_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_by_tenant(self, tenant_id: Any) -> list[ModelT]:
        """List all rows for the given tenant."""
        stmt = select(self.model).where(self.model.tenant_id == tenant_id)
        return list(self.session.execute(stmt).scalars().all())

    def update(self, tenant_id: Any, entity_id: Any, **fields: Any) -> ModelT | None:
        """Update arbitrary fields on a tenant-scoped row, if it exists.

        Only touches columns explicitly passed in `fields` — callers are
        responsible for only passing validated, intended fields (the
        service layer is where that validation happens).
        """
        entity = self.get_by_id(tenant_id, entity_id)
        if entity is None:
            return None

        for field_name, value in fields.items():
            setattr(entity, field_name, value)

        self.session.flush()
        return entity
