"""
RenderJob model.

Represents one async render request. Rows are created by the API layer
and updated by the Celery worker as the AI pipeline progresses — but no
such creation/update logic lives in this milestone, only the schema.

Per approved adjustment: `status` is stored as VARCHAR (not a native
PostgreSQL enum type), validated at the application layer via the
`RenderJobStatus` Python Enum. This trades a small amount of DB-level
strictness for the ability to add new statuses later without an
`ALTER TYPE` migration, which fits the MVP's flexibility goal.
"""

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, GUID, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.material import Material
    from app.db.models.tenant import Tenant
    from app.db.models.usage_record import UsageRecord


class RenderJobStatus(str, enum.Enum):
    """Allowed RenderJob lifecycle states.

    Backed by a VARCHAR column (see `native_enum=False` below), not a
    PostgreSQL native enum type — adding a new status later is a plain
    application-level change, no `ALTER TYPE ... ADD VALUE` required.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RenderJob(Base, TimestampMixin):
    __tablename__ = "render_jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    material_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("materials.id", ondelete="SET NULL"), nullable=True, index=True
    )

    status: Mapped[RenderJobStatus] = mapped_column(
        SAEnum(
            RenderJobStatus,
            name="render_job_status",
            native_enum=False,  # VARCHAR column, per approved adjustment
            length=20,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=RenderJobStatus.PENDING,
    )

    input_image_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    output_image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    tenant: Mapped["Tenant"] = relationship(back_populates="render_jobs")
    material: Mapped["Material | None"] = relationship(back_populates="render_jobs")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="render_job")

    def __repr__(self) -> str:
        return f"<RenderJob id={self.id} status={self.status.value}>"
