import uuid

import pytest
from sqlalchemy.orm import Session

from app.db.models import UsageRecord
from app.services import NotFoundError, TenantService, UsageRecordService


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant, _, _ = TenantService(db_session).create_tenant(name="Acme", slug="acme")
    return tenant.id


def test_list_usage_records(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    db_session.add(UsageRecord(tenant_id=tenant_id, event_type="render_completed"))
    db_session.commit()

    service = UsageRecordService(db_session)
    records = service.list_usage_records(tenant_id)
    assert len(records) == 1


def test_get_usage_record_not_found_raises(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    service = UsageRecordService(db_session)

    with pytest.raises(NotFoundError):
        service.get_usage_record(tenant_id, uuid.uuid4())


def test_no_create_method_exposed() -> None:
    """Structural check: UsageRecordService must not expose a public
    create method, matching the read-foundation scope."""
    assert not hasattr(UsageRecordService, "create_usage_record")
