from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from h3_service.config import Settings
from h3_service.task_store import TaskStore


class FakeResult:
    def __init__(self, *, one=None, many=None):
        self.one = one
        self.many = many or []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class FakeConnection:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        self.calls.append((statement, params or {}))
        return next(self.results)


class RecordingConnection(FakeConnection):
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((statement, params or {}))
        return FakeResult()


def settings() -> Settings:
    return Settings(
        database_url="postgresql://unused",
        storage_root=Path("/tmp/h3-test"),
        sglang_url="http://sglang",
        instance_id="worker-b",
        start_workers=False,
    )


def test_claim_running_task_returns_takeover_metadata_and_prioritizes_another_instance():
    task_id = UUID("00000000-0000-0000-0000-000000000001")
    token = UUID("00000000-0000-0000-0000-000000000002")
    expired = datetime.now(timezone.utc) - timedelta(seconds=10)
    row = {
        "task_id": task_id,
        "claim_token": token,
        "status": "running",
        "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        "_previous_status": "running",
        "_previous_executor_instance_id": "worker-a",
        "_previous_lease_expires_at": expired,
    }
    connection = FakeConnection([FakeResult(one=row), FakeResult(many=[])])
    store = TaskStore(settings())
    store.connect = lambda: connection

    claimed = store.claim_next_task("fl2va")

    claim_sql, params = connection.calls[0]
    assert "previous_executor_instance_id" in claim_sql
    assert "executor_instance_id IS DISTINCT FROM %(instance)s" in claim_sql
    assert "same_instance_cutoff" in claim_sql
    assert params["instance"] == "worker-b"
    assert claimed["_previous_status"] == "running"
    assert claimed["_previous_executor_instance_id"] == "worker-a"
    assert claimed["_previous_lease_expires_at"] == expired


def test_lease_renewal_is_anchored_to_last_backend_activity():
    healthy_at = datetime.now(timezone.utc)
    renewed_until = healthy_at + timedelta(seconds=300)
    connection = FakeConnection([FakeResult(one={"lease_expires_at": renewed_until})])
    store = TaskStore(settings())
    store.connect = lambda: connection

    result = store.extend_lease(
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        healthy_at,
    )

    renewal_sql, params = connection.calls[0]
    assert "lease_expires_at > %(now)s" in renewal_sql
    assert params["expires"] == renewed_until
    assert result == renewed_until


def test_claim_token_failure_update_also_requires_unexpired_lease():
    connection = FakeConnection([FakeResult(one=None)])
    store = TaskStore(settings())
    store.connect = lambda: connection

    updated = store.mark_failed(
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        "InternalError",
        "old worker must not win",
    )

    update_sql, _params = connection.calls[0]
    assert "claim_token=%(token)s AND lease_expires_at > %(now)s" in update_sql
    assert updated is False


def test_backend_submission_is_reserved_atomically_with_claim_and_limit():
    connection = FakeConnection([FakeResult(one={"backend_submit_count": 3})])
    store = TaskStore(settings())
    store.connect = lambda: connection

    count = store.reserve_backend_submission(
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        5,
    )

    update_sql, params = connection.calls[0]
    assert "backend_submit_count=backend_submit_count+1" in update_sql
    assert "backend_submit_count < %(max_attempts)s" in update_sql
    assert "claim_token=%(token)s" in update_sql
    assert "lease_expires_at > %(now)s" in update_sql
    assert params["max_attempts"] == 5
    assert count == 3


def test_backend_submission_reservation_returns_none_at_limit():
    connection = FakeConnection([FakeResult(one=None)])
    store = TaskStore(settings())
    store.connect = lambda: connection

    count = store.reserve_backend_submission(
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        5,
    )

    assert count is None


def test_schema_initializes_submission_count_for_existing_backend_jobs():
    connection = RecordingConnection()
    store = TaskStore(settings())
    store.connect = lambda: connection

    store.init()

    statements = [statement for statement, _params in connection.calls]
    assert any("ADD COLUMN IF NOT EXISTS backend_submit_count" in statement for statement in statements)
    assert any(
        "SET backend_submit_count=1" in statement and "backend_job_id IS NOT NULL" in statement
        for statement in statements
    )
