from __future__ import annotations

import logging
import os
import threading
import time

from .config import Settings
from .sglang import BackendError, SGLangClient
from .storage import SharedStorage, validate_remote_url
from .task_store import TaskStore, now


logger = logging.getLogger("h3.worker")


class _ClaimLost(Exception):
    pass


class _TaskTerminal(Exception):
    pass


class _LeaseHeartbeat:
    def __init__(self, settings: Settings, store: TaskStore, task: dict, stop_event: threading.Event):
        self.settings = settings
        self.store = store
        self.task_id = task["task_id"]
        self.claim_token = task["claim_token"]
        self.executor_instance_id = settings.instance_id
        self.backend_url = task.get("backend_url") or settings.sglang_url
        self.backend_job_id = task.get("backend_job_id")
        self.lease_expires_at = task["lease_expires_at"]
        self.global_stop = stop_event
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.lock = threading.Lock()
        self.activity_sequence = 0
        self.renewed_sequence = 0
        self.last_backend_activity_at = None
        self.outage_started_at = None
        self.outage_error = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"h3-lease-heartbeat-{self.task_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)

    def wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            if self.global_stop.is_set() or self.stop_event.is_set() or self.lost_event.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.stop_event.wait(min(remaining, 0.5))

    def update_backend(self, backend_url: str, backend_job_id: str) -> None:
        with self.lock:
            self.backend_url = backend_url
            self.backend_job_id = backend_job_id

    def record_backend_activity(self, backend_status: str | None = None) -> bool:
        if self.lost_event.is_set():
            return False
        recovered = None
        with self.lock:
            self.activity_sequence += 1
            self.last_backend_activity_at = now()
            if self.outage_started_at is not None:
                recovered = max(time.monotonic() - self.outage_started_at, 0.0)
                self.outage_started_at = None
                self.outage_error = None
        if recovered is not None:
            logger.info(
                "event=task_backend_recovered task_id=%s executor_instance_id=%s "
                "backend_url=%s backend_job_id=%s backend_status=%s outage_seconds=%.3f",
                self.task_id,
                self.executor_instance_id,
                self.backend_url,
                self.backend_job_id,
                backend_status or "unknown",
                recovered,
            )
        return recovered is not None

    def record_backend_failure(self, error: Exception | str) -> None:
        first = False
        with self.lock:
            if self.outage_started_at is None:
                self.outage_started_at = time.monotonic()
                first = True
            self.outage_error = str(error)
            last_activity = self.last_backend_activity_at
            lease_expires_at = self.lease_expires_at
        if first:
            remaining = max((lease_expires_at - now()).total_seconds(), 0.0)
            logger.warning(
                "event=task_lease_renewal_suspended task_id=%s executor_instance_id=%s "
                "backend_url=%s backend_job_id=%s last_backend_activity_at=%s "
                "lease_expires_at=%s seconds_remaining=%.3f error_type=%s error=%s",
                self.task_id,
                self.executor_instance_id,
                self.backend_url,
                self.backend_job_id,
                last_activity.isoformat() if last_activity else "none",
                lease_expires_at.isoformat(),
                remaining,
                type(error).__name__,
                str(error),
            )

    def _mark_lost(self, event: str, reason: str) -> None:
        if self.lost_event.is_set():
            return
        self.lost_event.set()
        with self.lock:
            lease_expires_at = self.lease_expires_at
            outage_started_at = self.outage_started_at
            outage_error = self.outage_error
        outage_seconds = max(time.monotonic() - outage_started_at, 0.0) if outage_started_at else 0.0
        logger.warning(
            "event=%s task_id=%s executor_instance_id=%s backend_url=%s backend_job_id=%s "
            "lease_expires_at=%s outage_seconds=%.3f reason=%s error=%s",
            event,
            self.task_id,
            self.executor_instance_id,
            self.backend_url,
            self.backend_job_id,
            lease_expires_at.isoformat(),
            outage_seconds,
            reason,
            outage_error or "none",
        )

    def _run(self) -> None:
        interval = max(1.0, min(60.0, self.settings.lease_seconds / 3))
        while not self.stop_event.wait(interval) and not self.global_stop.is_set():
            with self.lock:
                has_activity = self.activity_sequence > self.renewed_sequence
                sequence = self.activity_sequence
                last_backend_activity_at = self.last_backend_activity_at
                lease_expires_at = self.lease_expires_at
            if not has_activity:
                if now() >= lease_expires_at:
                    self._mark_lost("task_lease_expired", "no backend activity before lease expiry")
                    return
                continue
            try:
                renewed_until = self.store.extend_lease(
                    self.task_id,
                    self.claim_token,
                    last_backend_activity_at,
                )
            except Exception as exc:
                logger.warning(
                    "event=task_lease_renewal_failed task_id=%s executor_instance_id=%s "
                    "lease_expires_at=%s error_type=%s error=%s",
                    self.task_id,
                    self.executor_instance_id,
                    lease_expires_at.isoformat(),
                    type(exc).__name__,
                    str(exc),
                )
                if now() >= lease_expires_at:
                    self._mark_lost("task_lease_expired", "database lease renewal failed")
                    return
                continue
            if renewed_until is None:
                self._mark_lost("task_lease_expired", "claim token or lease is no longer valid")
                return
            with self.lock:
                self.renewed_sequence = sequence
                self.lease_expires_at = renewed_until
            logger.debug(
                "event=task_lease_renewed task_id=%s executor_instance_id=%s "
                "backend_url=%s backend_job_id=%s lease_expires_at=%s",
                self.task_id,
                self.executor_instance_id,
                self.backend_url,
                self.backend_job_id,
                renewed_until.isoformat(),
            )


class TaskWorker:
    def __init__(self, settings: Settings, store: TaskStore, storage: SharedStorage,
                 stop_event: threading.Event, drain_event: threading.Event | None = None):
        self.settings = settings
        self.store = store
        self.storage = storage
        self.stop_event = stop_event
        # drain_event:只停止**领新任务**,手上那条继续跑完(优雅关闭的第一步)。
        # stop_event:硬停,连在途的也放手(兜底,超时后才用)。两者语义不同,别混。
        self.drain_event = drain_event or threading.Event()
        # busy:本 worker 是否正在处理一条任务。关闭流程据此判断能不能退 ——
        # 不查库,因为库里"挂着我名字的 running 任务"可能是**同名旧 Pod** 留下的僵尸
        # (历史上 Server 是 StatefulSet,Pod 名会复用,而 H3_INSTANCE_ID 取的就是 Pod 名;
        # 现在改成 Deployment、名字带随机后缀不复用,这个具体场景没有了。但**仍然不查库**:
        # 判断"我忙不忙"本来就该看进程内状态,查库既慢又会把别人的行算进来),那条没人在处理,
        # 拿它当"我还忙着"会白等到超时。
        self.busy = threading.Event()

    def run(self) -> None:
        """领任务 -> 跑完 -> 再领,严格串行(每个 Pod 只有一个 TaskWorker,手上最多一条)。

        drain_event 置位后不再领新的:那是关闭流程的第一步(见 server.py 的 SIGTERM 处理)。
        手上那条照常跑完 —— 它的记账(轮询状态、下载产物、标 succeeded)只有本进程做得了,
        提前放手就得靠别的副本在租约过期后接管,而单副本的池根本没有"别的副本"。
        跑完且已排空时线程直接结束,关闭流程据此判断"可以退了"。
        """
        announced = False
        backend_ready: bool | None = None
        readiness_client = SGLangClient(
            self.settings,
            self.storage,
            timeout=self.settings.backend_ready_timeout_seconds,
        )
        while not self.stop_event.is_set():
            try:
                if self.drain_event.is_set():
                    if not announced:
                        logger.info(
                            "event=task_worker_draining instance=%s capability=%s",
                            self.settings.instance_id, self.settings.capability,
                        )
                        announced = True
                    return
                # Pod 里的 server/sglang 容器是并行启动的。Kubernetes readiness
                # 只决定 Service 是否给 Pod 导流,拦不住这个后台线程直接从数据库领任务。
                # 必须在 claim 前自己等 /health=200;等待期间不占租约、不发生成请求,
                # 也不消耗 backend_submit_count。
                if not readiness_client.is_ready():
                    if backend_ready is not False:
                        logger.warning(
                            "event=backend_not_ready instance=%s capability=%s backend_url=%s",
                            self.settings.instance_id,
                            self.settings.capability,
                            readiness_client.base_url,
                        )
                    backend_ready = False
                    self.stop_event.wait(self.settings.poll_seconds)
                    continue
                if backend_ready is not True:
                    logger.info(
                        "event=backend_ready instance=%s capability=%s backend_url=%s",
                        self.settings.instance_id,
                        self.settings.capability,
                        readiness_client.base_url,
                    )
                backend_ready = True
                # **领取本身也要算"忙"**:claim 是一次阻塞的数据库调用,
                # 如果在它执行期间收到 SIGTERM,而 busy 还没置位,关闭流程会判定
                # "worker 空闲"直接 stop_event.set() —— 而 claim 已经把任务领到手了,
                # 紧接着 _process 的轮询循环因 stop_event 立刻退出,任务被丢在 running
                # 等租约接管(单副本的池根本没有接管者)。
                self.busy.set()
                # 置 busy 之后**再查一次 drain**:上面那次检查和 busy.set() 之间有个窗口,
                # SIGTERM 正好落在里面时,关闭流程会读到 busy=false、判定空闲并 stop_event.set(),
                # 而这里接着 claim + 提交作业,_poll 却因 stop_event 立刻退出 → 任务变僵尸。
                #
                # 两边是一对方向相反的标志(Dekker):关闭流程先置 drain 再读 busy,
                # 这里先置 busy 再读 drain —— 至少有一方看得见对方。
                #   关闭读到 busy=false ⇒ 此刻 busy 还没置位 ⇒ 这里随后读 drain 必然已置位 ⇒ 不领新任务;
                #   这里读到 drain=false ⇒ busy 已先置位 ⇒ 关闭必然读到 busy=true ⇒ 会等它跑完。
                if self.drain_event.is_set():
                    self.busy.clear()
                    continue          # 回到循环开头,由上面那个分支打日志并退出
                try:
                    task = self.store.claim_next_task(self.settings.capability)
                    if task is not None:
                        # 领到手了就跑完:此时 busy 是置位的,关闭流程会等它
                        self._process(task)
                finally:
                    self.busy.clear()
                if task is None:
                    self.stop_event.wait(1)
            except Exception:
                logger.exception("task worker iteration failed")
                self.stop_event.wait(2)

    def _enforce_task_timeout(
        self,
        task: dict,
        client: SGLangClient | None = None,
        job_id: str | None = None,
    ) -> None:
        age_seconds = (now() - task["created_at"]).total_seconds()
        if age_seconds <= self.settings.queue_timeout_seconds:
            return
        task_id = task["task_id"]
        token = task["claim_token"]
        has_backend_job = bool(task.get("backend_job_id"))
        error_code = "TaskTimeout" if has_backend_job or task.get("status") == "running" else "QueueTimeout"
        message = (
            "task exceeded the configured total lifetime"
            if error_code == "TaskTimeout"
            else "task exceeded the configured queue timeout"
        )
        if not self.store.mark_failed(task_id, token, error_code, message):
            raise _ClaimLost
        logger.warning(
            "event=task_timeout_exceeded task_id=%s executor_instance_id=%s "
            "status=%s backend_url=%s backend_job_id=%s age_seconds=%.3f "
            "timeout_seconds=%s error_code=%s",
            task_id,
            self.settings.instance_id,
            task.get("status") or "unknown",
            task.get("backend_url") or "none",
            task.get("backend_job_id") or "none",
            age_seconds,
            self.settings.queue_timeout_seconds,
            error_code,
        )
        active_job_id = job_id or task.get("backend_job_id")
        if active_job_id:
            cancel_client = client or SGLangClient(
                self.settings,
                self.storage,
                task.get("backend_url") or self.settings.sglang_url,
            )
            cancel_client.cancel(active_job_id)
        raise _TaskTerminal

    def _fail_attempt_limit(self, task: dict) -> None:
        message = f"task reached the maximum of {self.settings.max_attempts} SGLang submissions"
        if not self.store.mark_failed(task["task_id"], task["claim_token"], "TooManyAttempts", message):
            raise _ClaimLost
        logger.warning(
            "event=task_attempt_limit_exceeded task_id=%s executor_instance_id=%s "
            "backend_url=%s backend_job_id=%s backend_submit_count=%s max_attempts=%s",
            task["task_id"],
            self.settings.instance_id,
            task.get("backend_url") or "none",
            task.get("backend_job_id") or "none",
            task.get("backend_submit_count", self.settings.max_attempts),
            self.settings.max_attempts,
        )
        raise _TaskTerminal

    def _reserve_backend_submission(self, task: dict) -> None:
        count = self.store.reserve_backend_submission(
            task["task_id"],
            task["claim_token"],
            self.settings.max_attempts,
        )
        if count is None:
            self._fail_attempt_limit(task)
        task["backend_submit_count"] = count
        logger.info(
            "event=task_backend_submission_reserved task_id=%s executor_instance_id=%s "
            "backend_submit_count=%s max_attempts=%s",
            task["task_id"],
            self.settings.instance_id,
            count,
            self.settings.max_attempts,
        )

    def _process(self, task: dict) -> None:
        task_id = task["task_id"]
        token = task["claim_token"]
        takeover = task.get("_previous_status") == "running"
        if takeover:
            previous_executor = task.get("_previous_executor_instance_id")
            previous_expiry = task.get("_previous_lease_expires_at")
            overdue = max((now() - previous_expiry).total_seconds(), 0.0) if previous_expiry else 0.0
            logger.warning(
                "event=task_expired_lease_observed task_id=%s capability=%s "
                "previous_executor_instance_id=%s new_executor_instance_id=%s "
                "previous_lease_expires_at=%s overdue_seconds=%.3f backend_url=%s backend_job_id=%s",
                task_id,
                self.settings.capability,
                previous_executor or "none",
                self.settings.instance_id,
                previous_expiry.isoformat() if previous_expiry else "none",
                overdue,
                task.get("backend_url") or "none",
                task.get("backend_job_id") or "none",
            )
            logger.info(
                "event=task_takeover_started task_id=%s capability=%s claim_token=%s "
                "previous_executor_instance_id=%s new_executor_instance_id=%s "
                "backend_url=%s backend_job_id=%s attempt_count=%s backend_submit_count=%s",
                task_id,
                self.settings.capability,
                token,
                previous_executor or "none",
                self.settings.instance_id,
                task.get("backend_url") or "none",
                task.get("backend_job_id") or "none",
                task.get("attempt_count", 0),
                task.get("backend_submit_count", 0),
            )
        client = SGLangClient(self.settings, self.storage, task.get("backend_url") or self.settings.sglang_url)
        try:
            self._enforce_task_timeout(task, client)
            if not task.get("backend_job_id") and task.get("status") == "queued":
                self._reserve_backend_submission(task)
                try:
                    submitted = client.submit(task)
                except BackendError as exc:
                    if exc.retryable and task["backend_submit_count"] >= self.settings.max_attempts:
                        self._fail_attempt_limit(task)
                    raise
                task["backend_job_id"] = submitted["id"]
                task["backend_url"] = client.base_url
                task["backend_status"] = submitted.get("status", "queued")
                if not self.store.set_backend_job(task_id, token, client.base_url, submitted["id"], task["backend_status"]):
                    client.cancel(submitted["id"])
                    return
                task["status"] = "running"
                task["backend_url"] = client.base_url
            elif task.get("status") == "queued":
                if not self.store.mark_running(
                    task_id, token, task.get("backend_status") or "submitted"
                ):
                    return
                task["status"] = "running"
            self._poll(task, client, takeover=takeover, backend_was_just_accepted=not takeover and bool(task.get("backend_job_id")))
        except _TaskTerminal:
            return
        except _ClaimLost:
            logger.warning(
                "event=task_claim_lost task_id=%s executor_instance_id=%s claim_token=%s",
                task_id,
                self.settings.instance_id,
                token,
            )
        except BackendError as exc:
            if exc.ambiguous:
                self.store.mark_failed(task_id, token, "BackendSubmissionUnknown", str(exc))
            elif exc.retryable:
                self.store.release_for_retry(task_id, token, str(exc), keep_status=bool(task.get("backend_job_id")))
            else:
                self.store.mark_failed(task_id, token, "BackendError", str(exc))
        except Exception as exc:
            logger.exception("task %s failed", task_id)
            self.store.mark_failed(task_id, token, "InternalError", str(exc))

    def _poll(
        self,
        task: dict,
        client: SGLangClient,
        *,
        takeover: bool,
        backend_was_just_accepted: bool,
    ) -> None:
        task_id = task["task_id"]
        token = task["claim_token"]
        job_id = task["backend_job_id"]
        heartbeat = _LeaseHeartbeat(self.settings, self.store, task, self.stop_event)
        if backend_was_just_accepted:
            heartbeat.record_backend_activity(task.get("backend_status"))
        heartbeat.start()
        first_query = takeover
        try:
            self._enforce_task_timeout(task, client, job_id)
            if not job_id:
                missing_reason = "running task did not contain a backend job id"
                heartbeat.record_backend_failure(missing_reason)
                replacement = self._replace_backend(task, heartbeat, missing_reason)
                if replacement is None:
                    return
                client, job_id = replacement
                first_query = False
            while not self.stop_event.is_set() and not heartbeat.lost_event.is_set():
                self._enforce_task_timeout(task, client, job_id)
                try:
                    data = client.status(job_id)
                    status = str(data.get("status", "")).lower()
                    recovered = heartbeat.record_backend_activity(status)
                    if heartbeat.lost_event.is_set():
                        raise _ClaimLost
                    if (recovered or first_query) and not self.store.clear_backend_error(task_id, token, status):
                        raise _ClaimLost
                    if first_query:
                        logger.info(
                            "event=task_takeover_resumed task_id=%s executor_instance_id=%s "
                            "backend_url=%s backend_job_id=%s backend_status=%s",
                            task_id,
                            self.settings.instance_id,
                            client.base_url,
                            job_id,
                            status or "unknown",
                        )
                        first_query = False
                    if status in {"in_progress", "processing", "running"}:
                        if task.get("status") != "running" or task.get("backend_status") != status:
                            if not self.store.mark_running(task_id, token, status):
                                raise _ClaimLost
                            task["status"] = "running"
                            task["backend_status"] = status
                    elif status in {"completed", "succeeded", "success"}:
                        relative = self.storage.output_relative_path(task)

                        def tracked_chunks():
                            for chunk in client.content_chunks(job_id):
                                self._enforce_task_timeout(task, client, job_id)
                                if heartbeat.lost_event.is_set():
                                    raise _ClaimLost
                                heartbeat.record_backend_activity(status)
                                yield chunk

                        asset = self.storage.write_response_stream(relative, tracked_chunks())
                        try:
                            marked = self.store.mark_succeeded(task_id, token, asset, status)
                        except Exception:
                            self.storage.delete(asset.relative_path)
                            raise
                        if not marked:
                            self.storage.delete(asset.relative_path)
                            raise _ClaimLost
                        return
                    elif status in {"failed", "error", "cancelled", "canceled"}:
                        error = data.get("error")
                        message = error.get("message") if isinstance(error, dict) else data.get("message")
                        if not self.store.mark_failed(
                            task_id,
                            token,
                            "BackendGenerationFailed",
                            message or "SGLang generation failed",
                            status,
                        ):
                            raise _ClaimLost
                        return
                except BackendError as exc:
                    if exc.missing or (first_query and exc.retryable):
                        heartbeat.record_backend_failure(exc)
                        replacement = self._replace_backend(task, heartbeat, str(exc))
                        if replacement is None:
                            return
                        client, job_id = replacement
                        first_query = False
                        continue
                    if not exc.retryable:
                        raise
                    if not self.store.mark_backend_unavailable(task_id, token, str(exc)):
                        raise _ClaimLost
                    heartbeat.record_backend_failure(exc)
                    if heartbeat.wait(30):
                        return
                    continue
                if heartbeat.wait(self.settings.poll_seconds):
                    return
        finally:
            heartbeat.stop()

    def _replace_backend(
        self,
        task: dict,
        heartbeat: _LeaseHeartbeat,
        reason: str,
    ) -> tuple[SGLangClient, str] | None:
        task_id = task["task_id"]
        token = task["claim_token"]
        old_backend_url = task.get("backend_url") or "none"
        old_backend_job_id = task.get("backend_job_id") or "none"
        client = SGLangClient(self.settings, self.storage, self.settings.sglang_url)
        started = time.monotonic()
        logger.warning(
            "event=task_takeover_resubmit_started task_id=%s executor_instance_id=%s "
            "old_backend_url=%s old_backend_job_id=%s new_backend_url=%s reason=%s",
            task_id,
            self.settings.instance_id,
            old_backend_url,
            old_backend_job_id,
            client.base_url,
            reason,
        )
        while not self.stop_event.is_set() and not heartbeat.lost_event.is_set():
            self._enforce_task_timeout(task)
            self._reserve_backend_submission(task)
            try:
                submitted = client.submit(task)
            except BackendError as exc:
                if not exc.retryable:
                    raise
                if task["backend_submit_count"] >= self.settings.max_attempts:
                    self._fail_attempt_limit(task)
                if not self.store.mark_backend_unavailable(task_id, token, str(exc)):
                    raise _ClaimLost
                heartbeat.record_backend_failure(exc)
                logger.warning(
                    "event=task_takeover_deferred task_id=%s executor_instance_id=%s "
                    "stage=resubmit backend_url=%s retry_seconds=30 error_type=%s error=%s",
                    task_id,
                    self.settings.instance_id,
                    client.base_url,
                    type(exc).__name__,
                    str(exc),
                )
                if heartbeat.wait(30):
                    return None
                continue
            backend_status = submitted.get("status", "queued")
            if not self.store.replace_backend_job(
                task_id,
                token,
                client.base_url,
                submitted["id"],
                backend_status,
            ):
                client.cancel(submitted["id"])
                raise _ClaimLost
            task["backend_url"] = client.base_url
            task["backend_job_id"] = submitted["id"]
            task["backend_status"] = backend_status
            heartbeat.update_backend(client.base_url, submitted["id"])
            heartbeat.record_backend_activity(backend_status)
            logger.info(
                "event=task_takeover_resubmit_succeeded task_id=%s executor_instance_id=%s "
                "old_backend_url=%s old_backend_job_id=%s new_backend_url=%s "
                "new_backend_job_id=%s backend_status=%s elapsed_seconds=%.3f",
                task_id,
                self.settings.instance_id,
                old_backend_url,
                old_backend_job_id,
                client.base_url,
                submitted["id"],
                backend_status,
                time.monotonic() - started,
            )
            return client, submitted["id"]
        return None


class CallbackWorker:
    def __init__(self, settings: Settings, store: TaskStore, stop_event: threading.Event):
        self.settings = settings
        self.store = store
        self.stop_event = stop_event

    def run(self) -> None:
        import httpx

        while not self.stop_event.is_set():
            try:
                delivery = self.store.claim_callback()
                if delivery is None:
                    self.stop_event.wait(1)
                    continue
                if delivery["attempt_count"] >= self.settings.callback_attempts:
                    self.store.abandon_callback(delivery["delivery_id"], "callback retry limit reached")
                    continue
                status = None
                error = None
                try:
                    validate_remote_url(delivery["callback_url"], self.settings.allow_private_urls)
                    with httpx.stream(
                        "POST", delivery["callback_url"], json=delivery["payload"],
                        timeout=10, follow_redirects=False,
                    ) as response:
                        status = response.status_code
                        if not 200 <= status < 300:
                            detail = bytearray()
                            for chunk in response.iter_bytes():
                                detail.extend(chunk)
                                if len(detail) >= 1000:
                                    break
                            error = bytes(detail[:1000]).decode("utf-8", errors="replace")
                except Exception as exc:
                    error = str(exc)
                self.store.finish_callback(delivery["delivery_id"], status, error)
            except Exception:
                logger.exception("callback worker iteration failed")
                self.stop_event.wait(2)


class MaintenanceWorker:
    def __init__(self, store: TaskStore, storage: SharedStorage, stop_event: threading.Event):
        self.store = store
        self.storage = storage
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.wait(60):
            try:
                self.store.reconcile_expired_tasks()
                for asset in self.store.expired_asset_paths():
                    self.storage.delete(asset["relative_path"])
                    self.store.mark_asset_deleted(asset["asset_id"])
                self.store.purge_expired_metadata()
            except Exception:
                logger.exception("asset cleanup failed")
