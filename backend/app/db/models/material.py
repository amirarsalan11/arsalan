"""
Material model.

Tenant-scoped material catalog entry. Per approved adjustment, materials
are NOT global in this MVP — every material belongs to exactly one
tenant, even if this means duplicate texture entries across tenants for
now. A shared/global material catalog can be introduced later without
breaking this schema (e.g. by adding a nullable tenant_id in a future
migration), but that is explicitly out of scope here.

`texture_url` stores the location of the texture asset only — no storage
client logic (S3/R2) lives in this model.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, GUID, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.render_job import RenderJob
    from app.db.models.tenant import Tenant


class Material(Base, TimestampMixin):
    __tablename__ = "materials"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    texture_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)

    # Physical dimensions in millimeters, used later by the Material Engine
    # for scale estimation. Nullable because not every material entry is
    # guaranteed to have this metadata at creation time.
    width_mm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height_mm: Mapped[int | None] = mapped_column(Integer, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    tenant: Mapped["Tenant"] = relationship(back_populates="materials")
    render_jobs: Mapped[list["RenderJob"]] = relationship(back_populates="material")

    def __repr__(self) -> str:
        return f"<Material id={self.id} name={self.name!r}>"
