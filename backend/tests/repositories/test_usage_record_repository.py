import uuid

from sqlalchemy.orm import Session

from app.db.models import UsageRecord
from app.repositories import TenantRepository, UsageRecordRepository


def _tenant_id(db_session: Session) -> uuid.UUID:
    tenant = TenantRepository(db_session).create(name="Acme", slug="acme")
    db_session.commit()
    return tenant.id


def test_usage_record_list_and_get_are_tenant_scoped(db_session: Session) -> None:
    tenant_id = _tenant_id(db_session)
    other_tenant_id = uuid.uuid4()

    record = UsageRecord(tenant_id=tenant_id, event_type="render_completed")
    db_session.add(record)
    db_session.commit()

    repo = UsageRecordRepository(db_session)
    assert len(repo.list_by_tenant(tenant_id)) == 1
    assert len(repo.list_by_tenant(other_tenant_id)) == 0
    assert repo.get_by_id(tenant_id, record.id) is not None
    assert repo.get_by_id(other_tenant_id, record.id) is None
