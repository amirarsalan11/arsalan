"""
UsageRecord model.

Immutable, append-only usage ledger — intentionally uses `CreatedAtMixin`
(created_at only, no updated_at) since usage events are never edited
after creation. `render_job_id` is nullable because usage events are not
guaranteed to always originate from a render job (e.g. future usage
event types unrelated to rendering).
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, CreatedAtMixin, GUID

if TYPE_CHECKING:
    from app.db.models.render_job import RenderJob
    from app.db.models.tenant import Tenant


class UsageRecord(Base, CreatedAtMixin):
    __tablename__ = "usage_records"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    render_job_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("render_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)

    tenant: Mapped["Tenant"] = relationship(back_populates="usage_records")
    render_job: Mapped["RenderJob | None"] = relationship(back_populates="usage_records")

    def __repr__(self) -> str:
        return f"<UsageRecord id={self.id} event_type={self.event_type!r}>"
