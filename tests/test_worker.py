import threading
from datetime import datetime, timedelta, timezone

from h3_service.config import Settings
from h3_service.sglang import BackendError
from h3_service.storage import StoredAsset
from h3_service.worker import TaskWorker, _LeaseHeartbeat


class FakeStore:
    def __init__(self):
        self.status = "queued"
        self.events = []
        self.backend_submit_count = 0

    def reserve_backend_submission(self, task_id, token, max_attempts):
        if self.backend_submit_count >= max_attempts:
            return None
        self.backend_submit_count += 1
        return self.backend_submit_count

    def set_backend_job(self, task_id, token, backend_url, backend_job_id, backend_status):
        assert self.status == "queued"
        self.status = "running"
        self.events.append(("running", backend_status))
        return True

    def mark_running(self, task_id, token, backend_status):
        self.status = "running"
        self.events.append(("running", backend_status))
        return True

    def mark_succeeded(self, task_id, token, asset, backend_status):
        assert self.status == "running"
        self.status = "succeeded"
        self.events.append(("succeeded", backend_status))
        return True

    def mark_failed(self, *args, **kwargs):
        raise AssertionError("task should not fail")

    def release_for_retry(self, *args, **kwargs):
        raise AssertionError("task should not be retried")

    def mark_backend_unavailable(self, task_id, token, message):
        self.events.append(("backend_unavailable", message))
        return True

    def clear_backend_error(self, task_id, token, backend_status=None):
        self.events.append(("backend_recovered", backend_status))
        return True

    def replace_backend_job(self, task_id, token, backend_url, backend_job_id, backend_status):
        self.status = "running"
        self.events.append(("backend_replaced", backend_url, backend_job_id, backend_status))
        return True


class FakeStorage:
    def output_relative_path(self, task):
        return f"{task['task_id']}/output.mp4"

    def write_response_stream(self, relative_path, chunks):
        assert b"".join(chunks) == b"video"
        return StoredAsset("output", "generated_video", relative_path, "video/mp4", 5)

    def delete(self, relative_path):
        raise AssertionError("successful output should not be deleted")


class FailingSuccessStore(FakeStore):
    def mark_succeeded(self, task_id, token, asset, backend_status):
        raise RuntimeError("database update failed")

    def mark_failed(self, *args, **kwargs):
        self.status = "failed"
        self.events.append(("failed", "InternalError"))


class CleanupStorage(FakeStorage):
    def __init__(self):
        self.deleted = []

    def delete(self, relative_path):
        self.deleted.append(relative_path)


class FastSGLangClient:
    submitted = 0

    def __init__(self, settings, storage, base_url=None):
        self.base_url = base_url or settings.sglang_url

    def submit(self, task):
        type(self).submitted += 1
        return {"id": "job-1", "status": "queued"}

    def status(self, job_id):
        return {"id": job_id, "status": "completed"}

    def content_chunks(self, job_id):
        yield b"video"

    def cancel(self, job_id):
        raise AssertionError("accepted job should not be cancelled")


def settings(tmp_path):
    return Settings(
        database_url="postgresql://unused",
        storage_root=tmp_path,
        sglang_url="http://sglang",
        start_workers=False,
    )


def task(**overrides):
    value = {
        "task_id": "00000000-0000-0000-0000-000000000001",
        "claim_token": "00000000-0000-0000-0000-000000000002",
        "created_at": datetime.now(timezone.utc),
        "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        "status": "queued",
        "backend_job_id": None,
        "backend_url": None,
        "backend_status": None,
        "backend_submit_count": 0,
    }
    value.update(overrides)
    return value


def test_fast_completed_job_still_transitions_through_running(tmp_path, monkeypatch):
    FastSGLangClient.submitted = 0
    monkeypatch.setattr("h3_service.worker.SGLangClient", FastSGLangClient)
    store = FakeStore()
    worker = TaskWorker(settings(tmp_path), store, FakeStorage(), threading.Event())

    worker._process(task())

    assert store.events == [("running", "queued"), ("succeeded", "completed")]
    assert FastSGLangClient.submitted == 1


def test_recovered_queued_job_is_marked_running_before_poll(tmp_path, monkeypatch):
    FastSGLangClient.submitted = 0
    monkeypatch.setattr("h3_service.worker.SGLangClient", FastSGLangClient)
    store = FakeStore()
    worker = TaskWorker(settings(tmp_path), store, FakeStorage(), threading.Event())

    worker._process(task(backend_job_id="job-existing", backend_status="queued"))

    assert store.events == [("running", "queued"), ("succeeded", "completed")]
    assert FastSGLangClient.submitted == 0


def test_output_is_deleted_when_success_state_cannot_be_stored(tmp_path, monkeypatch):
    monkeypatch.setattr("h3_service.worker.SGLangClient", FastSGLangClient)
    store = FailingSuccessStore()
    storage = CleanupStorage()
    worker = TaskWorker(settings(tmp_path), store, storage, threading.Event())

    worker._process(task())

    assert store.status == "failed"
    assert storage.deleted == ["00000000-0000-0000-0000-000000000001/output.mp4"]


class TakeoverSGLangClient:
    def __init__(self, settings, storage, base_url=None):
        self.base_url = base_url or settings.sglang_url

    def submit(self, task):
        assert self.base_url == "http://new-sglang"
        return {"id": "job-new", "status": "queued"}

    def status(self, job_id):
        if self.base_url == "http://old-sglang":
            raise BackendError("SGLang task no longer exists", missing=True)
        assert job_id == "job-new"
        return {"id": job_id, "status": "completed"}

    def content_chunks(self, job_id):
        yield b"video"

    def cancel(self, job_id):
        raise AssertionError("replacement job should remain active")


def test_expired_running_task_resubmits_to_takeover_workers_backend(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("h3_service.worker.SGLangClient", TakeoverSGLangClient)
    store = FakeStore()
    store.status = "running"
    config = settings(tmp_path)
    config = Settings(**{**config.__dict__, "sglang_url": "http://new-sglang", "instance_id": "worker-b"})
    worker = TaskWorker(config, store, FakeStorage(), threading.Event())
    expired_at = datetime.now(timezone.utc) - timedelta(seconds=3)

    with caplog.at_level("INFO", logger="h3.worker"):
        worker._process(task(
            status="running",
            backend_url="http://old-sglang",
            backend_job_id="job-old",
            backend_status="running",
            _previous_status="running",
            _previous_executor_instance_id="worker-a",
            _previous_lease_expires_at=expired_at,
        ))

    assert store.events == [
        ("backend_replaced", "http://new-sglang", "job-new", "queued"),
        ("succeeded", "completed"),
    ]
    assert "event=task_expired_lease_observed" in caplog.text
    assert "event=task_takeover_started" in caplog.text
    assert "event=task_takeover_resubmit_succeeded" in caplog.text


class ResumableTakeoverClient(TakeoverSGLangClient):
    def status(self, job_id):
        assert self.base_url == "http://old-sglang"
        assert job_id == "job-old"
        return {"id": job_id, "status": "completed"}

    def submit(self, task):
        raise AssertionError("existing backend job should be resumed")


def test_takeover_resumes_existing_backend_job_before_resubmitting(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("h3_service.worker.SGLangClient", ResumableTakeoverClient)
    store = FakeStore()
    store.status = "running"
    config = settings(tmp_path)
    config = Settings(**{**config.__dict__, "sglang_url": "http://new-sglang", "instance_id": "worker-b"})
    worker = TaskWorker(config, store, FakeStorage(), threading.Event())
    current = datetime.now(timezone.utc)

    with caplog.at_level("INFO", logger="h3.worker"):
        worker._process(task(
            status="running",
            backend_url="http://old-sglang",
            backend_job_id="job-old",
            backend_status="running",
            _previous_status="running",
            _previous_executor_instance_id="worker-a",
            _previous_lease_expires_at=current - timedelta(seconds=2),
        ))

    assert store.events == [("backend_recovered", "completed"), ("succeeded", "completed")]
    assert "event=task_takeover_resumed" in caplog.text
    assert "event=task_takeover_resubmit_started" not in caplog.text


class RecoveringSGLangClient(FastSGLangClient):
    calls = 0

    def status(self, job_id):
        type(self).calls += 1
        if type(self).calls == 1:
            raise BackendError("connection reset by peer", retryable=True)
        return {"id": job_id, "status": "completed"}


def test_running_task_retries_transient_backend_error_without_releasing_claim(tmp_path, monkeypatch, caplog):
    RecoveringSGLangClient.calls = 0
    monkeypatch.setattr("h3_service.worker.SGLangClient", RecoveringSGLangClient)
    monkeypatch.setattr(_LeaseHeartbeat, "wait", lambda self, seconds: False)
    store = FakeStore()
    worker = TaskWorker(settings(tmp_path), store, FakeStorage(), threading.Event())

    with caplog.at_level("INFO", logger="h3.worker"):
        worker._process(task())

    assert store.events[0] == ("running", "queued")
    assert store.events[1][0] == "backend_unavailable"
    assert store.events[2] == ("backend_recovered", "completed")
    assert store.events[3] == ("succeeded", "completed")
    assert store.backend_submit_count == 1
    assert not any(event[0] == "retry" for event in store.events)
    assert "event=task_lease_renewal_suspended" in caplog.text
    assert "event=task_backend_recovered" in caplog.text


class TerminalStore(FakeStore):
    def mark_failed(self, task_id, token, code, message, backend_status=None):
        self.status = "failed"
        self.events.append(("failed", code, message))
        return True


class MissingLoopSGLangClient(FastSGLangClient):
    submitted = 0

    def submit(self, task):
        type(self).submitted += 1
        return {"id": f"job-{type(self).submitted}", "status": "queued"}

    def status(self, job_id):
        raise BackendError(f"{job_id} no longer exists", missing=True)

    def cancel(self, job_id):
        return None


def test_missing_backend_jobs_stop_after_maximum_submission_count(tmp_path, monkeypatch, caplog):
    MissingLoopSGLangClient.submitted = 0
    monkeypatch.setattr("h3_service.worker.SGLangClient", MissingLoopSGLangClient)
    store = TerminalStore()
    worker = TaskWorker(settings(tmp_path), store, FakeStorage(), threading.Event())

    with caplog.at_level("INFO", logger="h3.worker"):
        worker._process(task())

    assert MissingLoopSGLangClient.submitted == 5
    assert store.backend_submit_count == 5
    assert store.status == "failed"
    assert store.events[-1][0:2] == ("failed", "TooManyAttempts")
    assert "event=task_attempt_limit_exceeded" in caplog.text


class RetryableSubmissionFailureClient(FastSGLangClient):
    submitted = 0

    def submit(self, task):
        type(self).submitted += 1
        raise BackendError("SGLang returned HTTP 500", retryable=True)


def test_last_allowed_submission_failure_reaches_terminal_state_immediately(tmp_path, monkeypatch):
    RetryableSubmissionFailureClient.submitted = 0
    monkeypatch.setattr("h3_service.worker.SGLangClient", RetryableSubmissionFailureClient)
    store = TerminalStore()
    config = settings(tmp_path)
    config = Settings(**{**config.__dict__, "max_attempts": 1})
    worker = TaskWorker(config, store, FakeStorage(), threading.Event())

    worker._process(task())

    assert RetryableSubmissionFailureClient.submitted == 1
    assert store.backend_submit_count == 1
    assert store.events[-1][0:2] == ("failed", "TooManyAttempts")


def test_expired_queued_task_fails_without_submitting(tmp_path, monkeypatch, caplog):
    FastSGLangClient.submitted = 0
    monkeypatch.setattr("h3_service.worker.SGLangClient", FastSGLangClient)
    store = TerminalStore()
    worker = TaskWorker(settings(tmp_path), store, FakeStorage(), threading.Event())

    with caplog.at_level("INFO", logger="h3.worker"):
        worker._process(task(created_at=datetime.now(timezone.utc) - timedelta(days=2)))

    assert FastSGLangClient.submitted == 0
    assert store.events[-1][0:2] == ("failed", "QueueTimeout")
    assert "event=task_timeout_exceeded" in caplog.text


class AgingSGLangClient(FastSGLangClient):
    submitted = 0
    active_task = None
    cancelled = []

    def submit(self, task):
        type(self).submitted += 1
        type(self).active_task = task
        return {"id": "job-aging", "status": "queued"}

    def status(self, job_id):
        type(self).active_task["created_at"] = datetime.now(timezone.utc) - timedelta(seconds=2)
        return {"id": job_id, "status": "running"}

    def cancel(self, job_id):
        type(self).cancelled.append(job_id)


def test_running_task_crossing_total_lifetime_is_failed_and_cancelled(tmp_path, monkeypatch, caplog):
    AgingSGLangClient.submitted = 0
    AgingSGLangClient.active_task = None
    AgingSGLangClient.cancelled = []
    monkeypatch.setattr("h3_service.worker.SGLangClient", AgingSGLangClient)
    monkeypatch.setattr(_LeaseHeartbeat, "wait", lambda self, seconds: False)
    store = TerminalStore()
    config = settings(tmp_path)
    config = Settings(**{**config.__dict__, "queue_timeout_seconds": 1})
    worker = TaskWorker(config, store, FakeStorage(), threading.Event())

    with caplog.at_level("INFO", logger="h3.worker"):
        worker._process(task())

    assert AgingSGLangClient.submitted == 1
    assert AgingSGLangClient.cancelled == ["job-aging"]
    assert store.events[-1][0:2] == ("failed", "TaskTimeout")
    assert "event=task_timeout_exceeded" in caplog.text


def test_worker_waits_for_backend_readiness_before_claiming(tmp_path, monkeypatch):
    """SGLang 未 ready 时任务留在数据库,恢复后才允许 claim。"""
    import threading

    claims = []
    readiness_checks = []
    stop = threading.Event()

    class Store:
        def claim_next_task(self, capability):
            claims.append(capability)
            stop.set()
            return None

    def is_ready(_client):
        # 前两次检查期间绝不能已经 claim;第三次恢复后才允许领取。
        readiness_checks.append(len(claims))
        return len(readiness_checks) >= 3

    monkeypatch.setattr("h3_service.worker.SGLangClient.is_ready", is_ready)
    config = Settings(
        database_url="postgresql://x/y", storage_root=str(tmp_path),
        sglang_url="http://sglang", sglang_media_root="/data", capability="fl2va",
        poll_seconds=0.01, backend_ready_timeout_seconds=0.01,
    )
    worker = TaskWorker(config, Store(), None, stop, threading.Event())

    worker.run()

    assert readiness_checks[:2] == [0, 0]
    assert claims == ["fl2va"]


def test_worker_can_drain_while_waiting_for_backend_readiness(tmp_path, monkeypatch):
    """SGLang 启动很慢时,排空信号仍能终止 Worker,且不会误领任务。"""
    import threading

    checked = threading.Event()
    drain = threading.Event()
    claims = []

    class Store:
        def claim_next_task(self, capability):
            claims.append(capability)
            return None

    def is_ready(_client):
        checked.set()
        return False

    monkeypatch.setattr("h3_service.worker.SGLangClient.is_ready", is_ready)
    config = Settings(
        database_url="postgresql://x/y", storage_root=str(tmp_path),
        sglang_url="http://sglang", sglang_media_root="/data", capability="fl2va",
        poll_seconds=0.05, backend_ready_timeout_seconds=0.01,
    )
    worker = TaskWorker(config, Store(), None, threading.Event(), drain)
    thread = threading.Thread(target=worker.run, daemon=True)

    thread.start()
    assert checked.wait(1)
    drain.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert claims == []


def test_worker_stops_claiming_and_exits_when_draining(tmp_path, monkeypatch):
    """drain_event 置位后:不再领新任务,且线程直接结束。

    这是优雅关闭能收敛的前提 —— TaskWorker 是**从数据库拉任务**的,Pod 进 Terminating
    只摘 Service 端点、拦不住它 claim。不停止领活的话,关闭流程会边等边接新活,永远等不到空。
    """
    import threading
    import time

    from h3_service.config import Settings
    from h3_service.worker import TaskWorker

    settings = Settings(
        database_url="postgresql://x/y", storage_root=str(tmp_path),
        sglang_url="http://127.0.0.1:1", sglang_media_root="/data", capability="fl2va",
    )
    claimed = []

    class Store:
        def claim_next_task(self, capability):
            claimed.append(capability)
            return None

    stop, drain = threading.Event(), threading.Event()
    w = TaskWorker(settings, Store(), None, stop, drain)
    monkeypatch.setattr("h3_service.worker.SGLangClient.is_ready", lambda _client: True)

    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(1.2)
    assert claimed, "未排空时应当在领任务"

    drain.set()
    t.join(timeout=5)
    assert not t.is_alive(), "排空且空闲时,worker 线程应当自行结束"

    n = len(claimed)
    time.sleep(0.5)
    assert len(claimed) == n, f"排空后不该再领任务,又领了 {len(claimed) - n} 次"


def test_worker_busy_flag_tracks_processing(tmp_path):
    """busy 标志要如实反映"正在处理一条任务" —— 关闭流程靠它判断能不能退。

    不用库里的 running_on_instance:那算的是"挂着本实例名的 running 任务",
    而历史上 StatefulSet 复用 Pod 名、H3_INSTANCE_ID 又取 Pod 名,同名旧 Pod 留下的僵尸任务
    会被算进来,导致关闭时白等到超时。
    """
    import threading

    from h3_service.config import Settings
    from h3_service.worker import TaskWorker

    settings = Settings(
        database_url="postgresql://x/y", storage_root=str(tmp_path),
        sglang_url="http://127.0.0.1:1", sglang_media_root="/data", capability="fl2va",
    )
    w = TaskWorker(settings, None, None, threading.Event(), threading.Event())
    assert not w.busy.is_set(), "刚建出来不该是 busy"


def test_busy_covers_claim_not_just_processing(tmp_path, monkeypatch):
    """claim 期间也必须算"忙"。

    claim 是一次阻塞的数据库调用。若它执行期间收到 SIGTERM 而 busy 还没置位,
    关闭流程会判定 worker 空闲、直接 stop_event.set() —— 而 claim 已经把任务领到手,
    _process 的轮询循环随即因 stop_event 退出,任务被丢在 running 等租约接管
    (单副本的池没有接管者)。
    """
    import threading
    import time

    from h3_service.config import Settings
    from h3_service.worker import TaskWorker

    settings = Settings(
        database_url="postgresql://x/y", storage_root=str(tmp_path),
        sglang_url="http://127.0.0.1:1", sglang_media_root="/data", capability="fl2va",
    )
    busy_during_claim = []
    entered = threading.Event()

    class Store:
        def claim_next_task(self, capability):
            entered.set()
            time.sleep(1.0)                      # 模拟一次慢的 claim
            busy_during_claim.append(w.busy.is_set())
            return None

    stop, drain = threading.Event(), threading.Event()
    w = TaskWorker(settings, Store(), None, stop, drain)
    monkeypatch.setattr("h3_service.worker.SGLangClient.is_ready", lambda _client: True)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    entered.wait(5)
    assert w.busy.is_set(), "claim 进行中时 busy 必须是置位的,否则关闭流程会误判空闲"
    stop.set()
    t.join(timeout=5)
    assert busy_during_claim and all(busy_during_claim), busy_during_claim


def test_drain_landing_between_check_and_busy_does_not_claim(tmp_path, monkeypatch):
    """SIGTERM 精确落在「查完 drain」和「busy.set()」之间时,不能再领新任务。

    这是 busy 包住 claim 之后**剩下的**那个窗口:关闭流程此刻读到 busy=false,
    会判定 worker 空闲并 stop_event.set();worker 若接着 claim + 提交作业,
    _poll 会因 stop_event 立刻退出,任务被丢在 running 等租约接管。

    这里用一个 busy 事件的 set() 钩子把 drain 精确置在那一刻 —— 等价于
    "关闭流程在这两条语句之间抢跑",不靠 sleep 赌时序。
    """
    import threading

    from h3_service.config import Settings
    from h3_service.worker import TaskWorker

    settings = Settings(
        database_url="postgresql://x/y", storage_root=str(tmp_path),
        sglang_url="http://127.0.0.1:1", sglang_media_root="/data", capability="fl2va",
    )
    claims = []

    class Store:
        def claim_next_task(self, capability):
            claims.append(capability)
            return None

    stop, drain = threading.Event(), threading.Event()
    w = TaskWorker(settings, Store(), None, stop, drain)
    monkeypatch.setattr("h3_service.worker.SGLangClient.is_ready", lambda _client: True)

    class BusyEvent(threading.Event):
        """busy.set() 的瞬间置 drain —— 正是那个窗口的最坏时刻。"""
        def set(self):
            super().set()
            drain.set()

    w.busy = BusyEvent()
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "drain 已置位,run() 应当退出"
    assert claims == [], f"drain 落在窗口里时不该再领任务,实际领了 {claims}"
    assert not w.busy.is_set(), "退出前必须把 busy 清掉,否则关闭流程会一直等到超时"
