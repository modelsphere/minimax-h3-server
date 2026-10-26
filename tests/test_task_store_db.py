"""连真 PostgreSQL 的冒烟测试。

MemoryStore 那套单测**测不出 SQL 语法错**(2026-09-01 踩过:
`EXTRACT(...) FILTER (...)` 把 FILTER 套在了聚合函数外面,单测全绿、
上了集群 /metrics/queue 直接 500)。所以查询语句必须真跑一次。

  H3_TEST_DATABASE_URL=postgresql://h3:h3@<host>:5432/minimaxh3 pytest tests/test_task_store_db.py

没设这个环境变量就跳过,不影响离线跑测试。
"""
import os

import pytest

from h3_service.config import Settings
from h3_service.task_store import TaskStore

DSN = os.getenv("H3_TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(not DSN, reason="需要 H3_TEST_DATABASE_URL 指向一个可写的测试库")


@pytest.fixture()
def store(tmp_path):
    settings = Settings(database_url=DSN, storage_root=tmp_path, sglang_url="http://sglang")
    store = TaskStore(settings)
    store.init()
    return store


def test_metrics_snapshot_runs_against_postgres(store):
    snapshot = store.metrics_snapshot()

    assert set(snapshot) == {
        "by_status", "completed_24h", "oldest_queued_age_seconds",
        "oldest_running_age_seconds", "waiting_takeover", "backend_unavailable",
        "running_on_instance",
    }
    assert isinstance(snapshot["by_status"], list)
    for row in snapshot["by_status"]:
        assert set(row) == {"status", "capability", "count"}
        # by_status 只报本副本自己的 capability(方案 B):Prometheus 的 service 标签
        # 因此与 capability 1:1,可按 service 过滤某个池。fixture 的 capability 默认 fl2va。
        assert row["capability"] == store.settings.capability
    # 没有排队任务时是 None(不上报),不能是 0 —— 0 会被看板读成"队列很健康"
    assert snapshot["oldest_queued_age_seconds"] is None or snapshot["oldest_queued_age_seconds"] >= 0
    assert snapshot["waiting_takeover"] >= 0
    assert snapshot["backend_unavailable"] >= 0
    assert snapshot["running_on_instance"] >= 0
