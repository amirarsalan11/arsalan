"""
AllowedDomain model.

Per-tenant whitelist of domains permitted to embed the widget. Domain
*validation logic* (checked at request time against the Referer/Origin
header) is implemented in the Authentication milestone — this model only
stores the whitelist data.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, GUID, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.tenant import Tenant


class AllowedDomain(Base, TimestampMixin):
    __tablename__ = "allowed_domains"
    __table_args__ = (
        UniqueConstraint("tenant_id", "domain", name="uq_allowed_domains_tenant_domain"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    tenant: Mapped["Tenant"] = relationship(back_populates="allowed_domains")

    def __repr__(self) -> str:
        return f"<AllowedDomain id={self.id} domain={self.domain!r}>"
