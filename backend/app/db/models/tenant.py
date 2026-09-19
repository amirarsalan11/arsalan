"""
Tenant model.

The aggregate root for multi-tenancy. Every other tenant-scoped table
(ApiKey, AllowedDomain, Material, RenderJob, UsageRecord) carries a
non-nullable `tenant_id` foreign key back to this table.

No authentication or domain-validation *logic* lives here — this is a
pure data model. Those behaviors are implemented in the service layer in
a later milestone.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, GUID, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.allowed_domain import AllowedDomain
    from app.db.models.api_key import ApiKey
    from app.db.models.material import Material
    from app.db.models.render_job import RenderJob
    from app.db.models.usage_record import UsageRecord


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # No delete-orphan cascade: the FK `ondelete="RESTRICT"` on child tables
    # plus the "deactivate, don't delete" tenant lifecycle (is_active=False)
    # are the intended safeguards. ORM-level cascading deletes would fight
    # that RESTRICT constraint and make accidental data loss easier.
    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="tenant")
    allowed_domains: Mapped[list["AllowedDomain"]] = relationship(back_populates="tenant")
    materials: Mapped[list["Material"]] = relationship(back_populates="tenant")
    render_jobs: Mapped[list["RenderJob"]] = relationship(back_populates="tenant")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="tenant")

    def __repr__(self) -> str:
        return f"<Tenant id={self.id} slug={self.slug!r}>"
