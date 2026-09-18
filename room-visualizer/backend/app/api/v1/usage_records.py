"""
UsageRecord API routes — read foundation only.

No POST/create endpoint exists here, matching the service layer's
read-only scope. Usage records are written internally by future
milestones, not through a public endpoint.
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant_id, get_db
from app.schemas.usage_record import UsageRecordRead
from app.services import UsageRecordService

router = APIRouter(prefix="/usage-records", tags=["usage-records"])


@router.get("", response_model=list[UsageRecordRead])
def list_usage_records(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id), db: Session = Depends(get_db)
) -> list[UsageRecordRead]:
    service = UsageRecordService(db)
    usage_records = service.list_usage_records(tenant_id)
    return [UsageRecordRead.model_validate(r) for r in usage_records]


@router.get("/{usage_record_id}", response_model=UsageRecordRead)
def get_usage_record(
    usage_record_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: Session = Depends(get_db),
) -> UsageRecordRead:
    service = UsageRecordService(db)
    usage_record = service.get_usage_record(tenant_id, usage_record_id)
    return UsageRecordRead.model_validate(usage_record)
