"""
UsageRecordService.

Read foundation only, per instruction — no create method is exposed
here or via the API. Usage records will be written internally by a
future milestone (e.g. when a render job completes), not through a
public CRUD endpoint.
"""

import uuid

from sqlalchemy.orm import Session

from app.db.models import UsageRecord
from app.repositories import UsageRecordRepository
from app.services.exceptions import NotFoundError


class UsageRecordService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = UsageRecordRepository(session)

    def list_usage_records(self, tenant_id: uuid.UUID) -> list[UsageRecord]:
        return self.repository.list_by_tenant(tenant_id)

    def get_usage_record(self, tenant_id: uuid.UUID, usage_record_id: uuid.UUID) -> UsageRecord:
        usage_record = self.repository.get_by_id(tenant_id, usage_record_id)
        if usage_record is None:
            raise NotFoundError(f"UsageRecord '{usage_record_id}' not found.")
        return usage_record
