"""
UsageRecordRepository.

Read-only surface, per the "UsageRecord read foundation" scope — no
`create` method is exposed here. get_by_id and list_by_tenant come
entirely from BaseRepository; this class exists only to bind `model`.
"""

from app.db.models import UsageRecord
from app.repositories.base import BaseRepository


class UsageRecordRepository(BaseRepository[UsageRecord]):
    model = UsageRecord
