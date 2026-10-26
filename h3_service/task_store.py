from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Settings
from .models import ParsedRequest
from .storage import StoredAsset


DASHBOARD_RETENTION_DAYS = 30
DASHBOARD_STATUS_NAMES = {
    "queued": "PENDING",
    "running": "RUNNING",
    "succeeded": "SUCCEEDED",
    "failed": "FAILED",
    "cancelled": "CANCELED",
}


def now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid(value: str | UUID | None) -> UUID | None:
    if value is None:
        return None
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (ValueError, TypeError):
        return None


class TaskStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def connect(self):
        conn = psycopg.connect(self.settings.database_url, row_factory=dict_row)
        self._mark_db_ok()
        return conn

    def _mark_db_ok(self) -> None:
        self._last_db_ok = time.monotonic()

    def health(self, ttl: float = 5.0, timeout: float = 2.0) -> tuple[bool, str]:
        """(ok, detail):数据库当前是否可用,给 /readyz 用。

        任务 Worker 每秒都会 claim_next_task 摸一次库,所以绝大多数时候直接复用
        它最近一次成功的时间戳,一条多余的查询都不用发。只有在 ttl 内没有任何成功
        记录时(例如 H3_START_WORKERS=false 的纯 API 实例)才真的探一次 SELECT 1。
        """
        last_ok = getattr(self, "_last_db_ok", None)
        if last_ok is not None and time.monotonic() - last_ok < ttl:
            return True, "recent-activity"
        try:
            with psycopg.connect(self.settings.database_url, connect_timeout=timeout) as conn:
                conn.execute("SELECT 1")
            self._mark_db_ok()
            return True, "ok"
        except Exception as exc:      # noqa: BLE001 —— 任何连不上/查不动都算不健康
            return False, f"{type(exc).__name__}: {exc}"

    def init(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS h3_video_tasks (
              task_id uuid PRIMARY KEY,
              model text NOT NULL,
              task_mode text NOT NULL,
              required_capability text NOT NULL,
              request_payload jsonb NOT NULL,
              content jsonb NOT NULL,
              prompt text NOT NULL,
              resolution text NOT NULL,
              ratio text NOT NULL,
              duration integer NOT NULL,
              seed bigint NOT NULL,
              status text NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
              error_code text,
              error_message text,
              usage jsonb NOT NULL DEFAULT '{}'::jsonb,
              callback_url text,
              request_base_url text,
              backend_url text,
              backend_job_id text,
              backend_status text,
              owner_instance_id text NOT NULL,
              executor_instance_id text,
              claim_token uuid,
              lease_expires_at timestamptz,
              attempt_count integer NOT NULL DEFAULT 0,
              backend_submit_count integer NOT NULL DEFAULT 0,
              next_attempt_at timestamptz NOT NULL DEFAULT now(),
              created_at timestamptz NOT NULL,
              scheduled_at timestamptz,
              started_at timestamptz,
              completed_at timestamptz,
              query_expires_at timestamptz NOT NULL,
              deleted_at timestamptz,
              updated_at timestamptz NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS h3_task_assets (
              asset_id bigserial PRIMARY KEY,
              task_id uuid NOT NULL REFERENCES h3_video_tasks(task_id) ON DELETE CASCADE,
              asset_type text NOT NULL,
              role text NOT NULL,
              relative_path text NOT NULL,
              media_type text NOT NULL,
              byte_size bigint NOT NULL,
              width integer,
              height integer,
              source_index integer NOT NULL DEFAULT 0,
              metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
              created_at timestamptz NOT NULL,
              expires_at timestamptz NOT NULL,
              deleted_at timestamptz
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS h3_callback_deliveries (
              delivery_id bigserial PRIMARY KEY,
              task_id uuid NOT NULL REFERENCES h3_video_tasks(task_id) ON DELETE CASCADE,
              task_status text NOT NULL,
              callback_url text NOT NULL,
              payload jsonb NOT NULL,
              attempt_count integer NOT NULL DEFAULT 0,
              next_attempt_at timestamptz NOT NULL DEFAULT now(),
              last_http_status integer,
              last_error text,
              delivered_at timestamptz,
              abandoned_at timestamptz,
              lease_owner text,
              lease_expires_at timestamptz,
              created_at timestamptz NOT NULL DEFAULT now(),
              UNIQUE (task_id, task_status)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_h3_tasks_claim ON h3_video_tasks (required_capability, status, next_attempt_at, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_h3_tasks_lease ON h3_video_tasks (lease_expires_at) WHERE status IN ('queued','running')",
            "CREATE INDEX IF NOT EXISTS idx_h3_tasks_dashboard ON h3_video_tasks (created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_h3_tasks_status_created ON h3_video_tasks (status, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_h3_assets_expiry ON h3_task_assets (expires_at) WHERE deleted_at IS NULL",
            "ALTER TABLE h3_video_tasks ADD COLUMN IF NOT EXISTS request_base_url text",
            "ALTER TABLE h3_video_tasks ADD COLUMN IF NOT EXISTS backend_submit_count integer NOT NULL DEFAULT 0",
            "UPDATE h3_video_tasks SET backend_submit_count=1 WHERE backend_submit_count=0 AND backend_job_id IS NOT NULL",
            "ALTER TABLE h3_task_assets ADD COLUMN IF NOT EXISTS source_index integer NOT NULL DEFAULT 0",
            "ALTER TABLE h3_task_assets ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE h3_task_assets DROP CONSTRAINT IF EXISTS h3_task_assets_task_id_asset_type_role_key",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_h3_assets_identity ON h3_task_assets (task_id, asset_type, role, source_index)",
            "ALTER TABLE h3_callback_deliveries ADD COLUMN IF NOT EXISTS abandoned_at timestamptz",
            "CREATE INDEX IF NOT EXISTS idx_h3_callbacks_due ON h3_callback_deliveries (next_attempt_at) WHERE delivered_at IS NULL AND abandoned_at IS NULL",
        )
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('h3_schema_init'))")
            for statement in statements:
                conn.execute(statement)

    def insert_task(
        self,
        task_id: str,
        payload: dict[str, Any],
        parsed: ParsedRequest,
        seed: int,
        assets: list[StoredAsset],
        request_base_url: str,
    ) -> dict[str, Any]:
        created = now()
        task_uuid = _uuid(task_id)
        params = {
            "task_id": task_uuid,
            "model": parsed.model,
            "task_mode": parsed.mode,
            "required_capability": parsed.required_capability,
            "request_payload": Jsonb(payload),
            "content": Jsonb(parsed.content),
            "prompt": parsed.prompt,
            "resolution": parsed.resolution,
            "ratio": parsed.ratio,
            "duration": parsed.duration,
            "seed": seed,
            "status": "queued",
            "usage": Jsonb({}),
            "callback_url": parsed.callback_url,
            "request_base_url": request_base_url,
            "owner_instance_id": self.settings.instance_id,
            "created_at": created,
            "query_expires_at": created + timedelta(days=self.settings.query_retention_days),
            "updated_at": created,
        }
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO h3_video_tasks (
                  task_id, model, task_mode, required_capability,
                  request_payload, content, prompt, resolution, ratio, duration,
                  seed, status, usage, callback_url, request_base_url, owner_instance_id,
                  created_at, query_expires_at, updated_at
                ) VALUES (
                  %(task_id)s, %(model)s, %(task_mode)s, %(required_capability)s,
                  %(request_payload)s, %(content)s, %(prompt)s, %(resolution)s,
                  %(ratio)s, %(duration)s, %(seed)s, %(status)s, %(usage)s,
                  %(callback_url)s, %(request_base_url)s, %(owner_instance_id)s, %(created_at)s,
                  %(query_expires_at)s, %(updated_at)s
                ) RETURNING *
                """,
                params,
            ).fetchone()
            for asset in assets:
                self._insert_asset(conn, task_uuid, asset, created)
            self._enqueue_callback(conn, row, "queued")
        return self._normalize(row)

    def get_task(self, task_id: str, include_expired: bool = False, include_deleted: bool = False) -> dict[str, Any] | None:
        task_uuid = _uuid(task_id)
        if task_uuid is None:
            return None
        clauses = ["task_id = %(task_id)s"]
        if not include_expired:
            clauses.append("query_expires_at > %(now)s")
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM h3_video_tasks WHERE {' AND '.join(clauses)}",
                {"task_id": task_uuid, "now": now()},
            ).fetchone()
            if row:
                row["assets"] = conn.execute(
                    "SELECT * FROM h3_task_assets WHERE task_id=%(task_id)s AND deleted_at IS NULL ORDER BY asset_id",
                    {"task_id": task_uuid},
                ).fetchall()
        return self._normalize(row) if row else None

    def claim_next_task(self, capability: str) -> dict[str, Any] | None:
        claimed = now()
        token = uuid4()
        same_instance_cutoff = claimed - timedelta(seconds=self.settings.lease_seconds)
        with self.connect() as conn:
            row = conn.execute(
                """
                WITH candidate AS (
                  SELECT task_id,
                         status AS previous_status,
                         executor_instance_id AS previous_executor_instance_id,
                         lease_expires_at AS previous_lease_expires_at
                  FROM h3_video_tasks
                  WHERE required_capability=%(capability)s
                    AND status IN ('queued','running')
                    AND deleted_at IS NULL
                    AND query_expires_at > %(claimed)s
                    AND next_attempt_at <= %(claimed)s
                    AND (
                      (status='queued' AND (lease_expires_at IS NULL OR lease_expires_at <= %(claimed)s))
                      OR
                      (status='running'
                        AND COALESCE(lease_expires_at, '-infinity'::timestamptz) <= %(claimed)s
                        AND (
                          executor_instance_id IS DISTINCT FROM %(instance)s
                          OR COALESCE(lease_expires_at, '-infinity'::timestamptz) <= %(same_instance_cutoff)s
                        )
                      )
                    )
                  ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at
                  FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE h3_video_tasks task
                SET claim_token=%(token)s,
                    lease_expires_at=%(lease_expires)s,
                    executor_instance_id=%(instance)s,
                    scheduled_at=COALESCE(scheduled_at, %(claimed)s),
                    updated_at=%(claimed)s
                FROM candidate WHERE task.task_id=candidate.task_id
                RETURNING task.*,
                          candidate.previous_status AS _previous_status,
                          candidate.previous_executor_instance_id AS _previous_executor_instance_id,
                          candidate.previous_lease_expires_at AS _previous_lease_expires_at
                """,
                {
                    "capability": capability,
                    "claimed": claimed,
                    "token": token,
                    "lease_expires": claimed + timedelta(seconds=self.settings.lease_seconds),
                    "instance": self.settings.instance_id,
                    "same_instance_cutoff": same_instance_cutoff,
                },
            ).fetchone()
            if row:
                row["assets"] = conn.execute(
                    "SELECT * FROM h3_task_assets WHERE task_id=%(task_id)s AND deleted_at IS NULL ORDER BY asset_id",
                    {"task_id": row["task_id"]},
                ).fetchall()
        return self._normalize(row) if row else None

    def extend_lease(
        self,
        task_id: str,
        claim_token: str,
        healthy_at: datetime | None = None,
    ) -> datetime | None:
        current = now()
        expires = (healthy_at or current) + timedelta(seconds=self.settings.lease_seconds)
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks SET lease_expires_at=%(expires)s, updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status='running' AND lease_expires_at > %(now)s
                   RETURNING lease_expires_at""",
                {
                    "task_id": _uuid(task_id),
                    "token": _uuid(claim_token),
                    "expires": expires,
                    "now": current,
                },
            ).fetchone()
        return row["lease_expires_at"] if row else None

    def reserve_backend_submission(
        self,
        task_id: str,
        claim_token: str,
        max_attempts: int,
    ) -> int | None:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks
                   SET backend_submit_count=backend_submit_count+1, updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status IN ('queued','running') AND lease_expires_at > %(now)s
                     AND backend_submit_count < %(max_attempts)s
                   RETURNING backend_submit_count""",
                {
                    "task_id": _uuid(task_id),
                    "token": _uuid(claim_token),
                    "max_attempts": max_attempts,
                    "now": current,
                },
            ).fetchone()
        return row["backend_submit_count"] if row else None

    def set_backend_job(self, task_id: str, claim_token: str, backend_url: str, backend_job_id: str, backend_status: str) -> bool:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks
                   SET backend_url=%(url)s, backend_job_id=%(job)s,
                       backend_status=%(backend_status)s, status='running',
                       started_at=COALESCE(started_at,%(now)s), updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status='queued' AND lease_expires_at > %(now)s RETURNING *""",
                {"task_id": _uuid(task_id), "token": _uuid(claim_token), "url": backend_url,
                 "job": backend_job_id, "backend_status": backend_status, "now": current},
            ).fetchone()
            if row:
                self._enqueue_callback(conn, row, "running")
        return row is not None

    def mark_running(self, task_id: str, claim_token: str, backend_status: str) -> bool:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks SET status='running', backend_status=%(backend_status)s,
                       started_at=COALESCE(started_at,%(now)s), updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status IN ('queued','running') AND lease_expires_at > %(now)s
                   RETURNING *""",
                {"task_id": _uuid(task_id), "token": _uuid(claim_token), "backend_status": backend_status, "now": current},
            ).fetchone()
            if row:
                self._enqueue_callback(conn, row, "running")
        return row is not None

    def mark_succeeded(self, task_id: str, claim_token: str, asset: StoredAsset, backend_status: str) -> bool:
        current = now()
        usage = {"total_seconds": None, "input_seconds": 0, "output_seconds": None, "input_image_count": None}
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks SET status='succeeded', backend_status=%(backend_status)s,
                       completed_at=%(now)s, updated_at=%(now)s, usage=%(usage)s,
                       claim_token=NULL, lease_expires_at=NULL
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status IN ('queued','running') AND lease_expires_at > %(now)s
                   RETURNING *""",
                {"task_id": _uuid(task_id), "token": _uuid(claim_token), "backend_status": backend_status,
                 "now": current, "usage": Jsonb(usage)},
            ).fetchone()
            if not row:
                return False
            input_usage = conn.execute(
                """SELECT
                     count(*) FILTER (WHERE left(media_type, 6) = 'image/') AS image_count,
                     COALESCE(sum((metadata->>'duration_seconds')::double precision)
                       FILTER (WHERE role='reference_video'), 0) AS video_seconds,
                     COALESCE(sum((metadata->>'duration_seconds')::double precision)
                       FILTER (WHERE role='reference_audio'), 0) AS audio_seconds
                   FROM h3_task_assets
                   WHERE task_id=%(task_id)s AND asset_type='input'""",
                {"task_id": row["task_id"]},
            ).fetchone()
            usage.update(
                total_seconds=row["duration"],
                input_seconds=float(input_usage["video_seconds"]),
                output_seconds=row["duration"],
                input_image_count=input_usage["image_count"],
            )
            if row["task_mode"] == "ref2va":
                usage["input_audio_seconds"] = float(input_usage["audio_seconds"])
            row = conn.execute(
                "UPDATE h3_video_tasks SET usage=%(usage)s WHERE task_id=%(task_id)s RETURNING *",
                {"usage": Jsonb(usage), "task_id": row["task_id"]},
            ).fetchone()
            self._insert_asset(conn, row["task_id"], asset, current)
            row["assets"] = conn.execute(
                "SELECT * FROM h3_task_assets WHERE task_id=%(task_id)s AND deleted_at IS NULL ORDER BY asset_id",
                {"task_id": row["task_id"]},
            ).fetchall()
            self._enqueue_callback(conn, row, "succeeded")
        return True

    def mark_failed(self, task_id: str, claim_token: str | None, code: str, message: str, backend_status: str | None = None) -> bool:
        current = now()
        token_filter = ""
        params: dict[str, Any] = {"task_id": _uuid(task_id), "now": current, "code": code,
                                  "message": str(message)[:4000], "backend_status": backend_status}
        if claim_token:
            token_filter = "AND claim_token=%(token)s AND lease_expires_at > %(now)s"
            params["token"] = _uuid(claim_token)
        with self.connect() as conn:
            row = conn.execute(
                f"""UPDATE h3_video_tasks SET status='failed', error_code=%(code)s,
                       error_message=%(message)s, backend_status=COALESCE(%(backend_status)s,backend_status),
                       completed_at=%(now)s, updated_at=%(now)s,
                       claim_token=NULL, lease_expires_at=NULL
                   WHERE task_id=%(task_id)s {token_filter}
                     AND status IN ('queued','running') RETURNING *""",
                params,
            ).fetchone()
            if row:
                self._enqueue_callback(conn, row, "failed")
        return row is not None

    def replace_backend_job(
        self,
        task_id: str,
        claim_token: str,
        backend_url: str,
        backend_job_id: str,
        backend_status: str,
    ) -> bool:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks
                   SET backend_url=%(url)s, backend_job_id=%(job)s,
                       backend_status=%(backend_status)s,
                       attempt_count=attempt_count+1,
                       next_attempt_at=%(now)s,
                       error_code=NULL, error_message=NULL, updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status='running' AND lease_expires_at > %(now)s
                   RETURNING task_id""",
                {
                    "task_id": _uuid(task_id),
                    "token": _uuid(claim_token),
                    "url": backend_url,
                    "job": backend_job_id,
                    "backend_status": backend_status,
                    "now": current,
                },
            ).fetchone()
        return row is not None

    def mark_backend_unavailable(self, task_id: str, claim_token: str, message: str) -> bool:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks
                   SET attempt_count=attempt_count+1,
                       error_code='BackendUnavailable', error_message=%(message)s,
                       updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status='running' AND lease_expires_at > %(now)s
                   RETURNING task_id""",
                {
                    "task_id": _uuid(task_id),
                    "token": _uuid(claim_token),
                    "message": str(message)[:4000],
                    "now": current,
                },
            ).fetchone()
        return row is not None

    def clear_backend_error(self, task_id: str, claim_token: str, backend_status: str | None = None) -> bool:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks
                   SET error_code=NULL, error_message=NULL,
                       backend_status=COALESCE(%(backend_status)s,backend_status),
                       updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status='running' AND lease_expires_at > %(now)s
                   RETURNING task_id""",
                {
                    "task_id": _uuid(task_id),
                    "token": _uuid(claim_token),
                    "backend_status": backend_status,
                    "now": current,
                },
            ).fetchone()
        return row is not None

    def release_for_retry(self, task_id: str, claim_token: str, message: str, keep_status: bool = True) -> bool:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """UPDATE h3_video_tasks
                   SET status=CASE WHEN %(keep_status)s THEN status ELSE 'queued' END,
                       attempt_count=attempt_count+1,
                       next_attempt_at=%(next)s, error_code='BackendUnavailable',
                       error_message=%(message)s, claim_token=NULL,
                       lease_expires_at=NULL, updated_at=%(now)s
                   WHERE task_id=%(task_id)s AND claim_token=%(token)s
                     AND status IN ('queued','running') AND lease_expires_at > %(now)s
                   RETURNING task_id""",
                {"task_id": _uuid(task_id), "token": _uuid(claim_token), "keep_status": keep_status,
                 "next": current + timedelta(seconds=30), "message": str(message)[:4000], "now": current},
            ).fetchone()
        return row is not None

    def cancel_or_delete(self, task_id: str) -> tuple[str, dict[str, Any]] | None:
        task_uuid = _uuid(task_id)
        if task_uuid is None:
            return None
        current = now()
        with self.connect() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT * FROM h3_video_tasks WHERE task_id=%(task_id)s AND deleted_at IS NULL FOR UPDATE",
                    {"task_id": task_uuid},
                ).fetchone()
                if not row:
                    return None
                if row["status"] == "queued":
                    row = conn.execute(
                        """UPDATE h3_video_tasks SET status='cancelled', completed_at=%(now)s,
                             updated_at=%(now)s, claim_token=NULL, lease_expires_at=NULL
                           WHERE task_id=%(task_id)s RETURNING *""",
                        {"task_id": task_uuid, "now": current},
                    ).fetchone()
                    self._enqueue_callback(conn, row, "cancelled")
                    return "cancelled", self._normalize(row)
                if row["status"] in {"succeeded", "failed"}:
                    row = conn.execute(
                        "UPDATE h3_video_tasks SET deleted_at=%(now)s, updated_at=%(now)s WHERE task_id=%(task_id)s RETURNING *",
                        {"task_id": task_uuid, "now": current},
                    ).fetchone()
                    return "deleted", self._normalize(row)
                return "unsupported", self._normalize(row)

    def list_tasks(self, page: int, page_size: int, status: str | None = None, model: str | None = None,
                   task_type: str | None = None, task_ids: list[str] | None = None) -> dict[str, Any]:
        clauses = ["deleted_at IS NULL", "query_expires_at > %(now)s"]
        params: dict[str, Any] = {"now": now()}
        if status:
            clauses.append("status=%(status)s")
            params["status"] = status
        if model:
            clauses.append("model=%(model)s")
            params["model"] = model
        if task_type and task_type != "generation":
            return {"items": [], "total": 0}
        if task_ids:
            parsed_ids = [item for item in (_uuid(value) for value in task_ids) if item]
            if not parsed_ids:
                return {"items": [], "total": 0}
            clauses.append("task_id=ANY(%(task_ids)s)")
            params["task_ids"] = parsed_ids
        where = " AND ".join(clauses)
        pagination = ""
        if page_size != -1:
            pagination = "LIMIT %(limit)s OFFSET %(offset)s"
            params.update({"limit": page_size, "offset": (page - 1) * page_size})
        with self.connect() as conn:
            total = conn.execute(f"SELECT count(*) AS value FROM h3_video_tasks WHERE {where}", params).fetchone()["value"]
            rows = conn.execute(
                f"SELECT * FROM h3_video_tasks WHERE {where} ORDER BY created_at DESC {pagination}",
                params,
            ).fetchall()
            for row in rows:
                row["assets"] = conn.execute(
                    "SELECT * FROM h3_task_assets WHERE task_id=%(task_id)s AND deleted_at IS NULL ORDER BY asset_id",
                    {"task_id": row["task_id"]},
                ).fetchall()
        return {"items": [self._normalize(row) for row in rows], "total": total}

    def metrics_snapshot(self) -> dict[str, Any]:
        """Prometheus /metrics 用的即时快照。

        `by_status`(→ h3_queue_tasks)**只报本副本自己的 capability**:每个 Pod 只服务
        一个能力池,这样 Prometheus 抓取时打的 `service` 标签就与 capability 1:1 对上,
        可以直接按 `service` 过滤某个池的积压(不必再靠 capability)。代价:同池多副本仍
        报同一个数(用 `max by(service)`/`max by(capability)` 去重),且某池 Pod 全挂时
        它的 h3_queue_tasks 会消失(没人报了)。

        其余 live 指标(waiting_takeover / oldest_* / backend_unavailable)仍是**集群维度**
        (跨全部 capability),保持既有告警语义不变 —— 抓取端用 max()/avg() 去重,别 sum()。
        只有 running_on_instance 是本副本自己的。
        """
        with self.connect() as conn:
            by_status = conn.execute(
                # 只统计本副本的 capability —— 见 docstring。
                """
                SELECT status, required_capability AS capability, COUNT(*) AS count
                FROM h3_video_tasks
                WHERE deleted_at IS NULL AND required_capability = %(capability)s
                GROUP BY status, required_capability
                """,
                {"capability": self.settings.capability},
            ).fetchall()
            live = conn.execute(
                """
                SELECT
                  -- FILTER 只能跟在聚合函数后面,不能套在 EXTRACT(...) 外面
                  EXTRACT(EPOCH FROM (
                    now() - MIN(created_at) FILTER (WHERE status='queued')
                  )) AS oldest_queued_age_seconds,
                  EXTRACT(EPOCH FROM (
                    now() - MIN(started_at) FILTER (WHERE status='running' AND started_at IS NOT NULL)
                  )) AS oldest_running_age_seconds,
                  -- 口径与 dashboard 的 running_waiting_takeover 对齐(上游 !3):
                  -- running 且租约为空或已过期 = 没人在管、等接管。
                  -- 之前这里是 status IN (queued,running) 且 lease 非空且过期,
                  -- 和 dashboard 各算各的,同一件事出两个数字,写告警时得先分辨用哪个。
                  COUNT(*) FILTER (
                    WHERE status='running'
                      AND (lease_expires_at IS NULL OR lease_expires_at <= now())
                  ) AS waiting_takeover,
                  -- 后端当前失联(仍在租约内,worker 还在重试)。持续大于 0 = SGLang 有问题。
                  COUNT(*) FILTER (
                    WHERE status='running'
                      AND error_code='BackendUnavailable'
                      AND lease_expires_at > now()
                  ) AS backend_unavailable,
                  COUNT(*) FILTER (
                    WHERE status='running' AND executor_instance_id = %(instance)s
                  ) AS running_on_instance
                FROM h3_video_tasks
                WHERE deleted_at IS NULL
                """,
                {"instance": self.settings.instance_id},
            ).fetchone()
            completed = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM h3_video_tasks
                WHERE completed_at >= now() - interval '24 hours'
                GROUP BY status
                """
            ).fetchall()
        return {
            "by_status": [dict(row) for row in by_status],
            "completed_24h": [dict(row) for row in completed],
            "oldest_queued_age_seconds": (live or {}).get("oldest_queued_age_seconds"),
            "oldest_running_age_seconds": (live or {}).get("oldest_running_age_seconds"),
            "waiting_takeover": (live or {}).get("waiting_takeover") or 0,
            "backend_unavailable": (live or {}).get("backend_unavailable") or 0,
            "running_on_instance": (live or {}).get("running_on_instance") or 0,
        }

    def dashboard_metrics(
        self,
        start: datetime,
        end: datetime,
        bucket_seconds: int,
        status: str | None = None,
        model: str | None = None,
        instance_id: str | None = None,
        executor_instance_id: str | None = None,
    ) -> dict[str, Any]:
        where, params = self._dashboard_where(status, model, instance_id, executor_instance_id)
        params.update({"start": start, "end": end, "bucket_seconds": bucket_seconds})
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                  ) AS total,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'queued'
                  ) AS pending,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'running'
                  ) AS running,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'running'
                      AND error_code = 'BackendUnavailable'
                      AND lease_expires_at > CURRENT_TIMESTAMP
                  ) AS running_backend_unavailable,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'running'
                      AND (lease_expires_at IS NULL OR lease_expires_at <= CURRENT_TIMESTAMP)
                  ) AS running_waiting_takeover,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'succeeded'
                  ) AS succeeded,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'failed'
                  ) AS failed,
                  COUNT(*) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'cancelled'
                  ) AS canceled,
                  AVG(EXTRACT(EPOCH FROM (scheduled_at - created_at))) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND scheduled_at IS NOT NULL
                  ) AS avg_queue_seconds,
                  percentile_cont(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (scheduled_at - created_at))
                  ) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND scheduled_at IS NOT NULL
                  ) AS p95_queue_seconds,
                  AVG(EXTRACT(EPOCH FROM (completed_at - scheduled_at))) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'succeeded'
                      AND scheduled_at IS NOT NULL AND completed_at IS NOT NULL
                  ) AS avg_processing_seconds,
                  percentile_cont(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (completed_at - scheduled_at))
                  ) FILTER (
                    WHERE created_at >= %(start)s AND created_at < %(end)s
                      AND status = 'succeeded'
                      AND scheduled_at IS NOT NULL AND completed_at IS NOT NULL
                  ) AS p95_processing_seconds,
                  COUNT(*) FILTER (
                    WHERE status = 'succeeded'
                      AND completed_at >= %(start)s AND completed_at < %(end)s
                  ) AS completed_in_window
                FROM h3_video_tasks
                {where}
                """,
                params,
            ).fetchone()
            series_rows = conn.execute(
                f"""
                WITH events AS (
                  SELECT
                    to_timestamp(floor(EXTRACT(EPOCH FROM created_at)
                      / %(bucket_seconds)s) * %(bucket_seconds)s) AS bucket,
                    1 AS submitted,
                    0 AS completed
                  FROM h3_video_tasks
                  {where}
                    AND created_at >= %(start)s AND created_at < %(end)s
                  UNION ALL
                  SELECT
                    to_timestamp(floor(EXTRACT(EPOCH FROM completed_at)
                      / %(bucket_seconds)s) * %(bucket_seconds)s) AS bucket,
                    0 AS submitted,
                    1 AS completed
                  FROM h3_video_tasks
                  {where}
                    AND status = 'succeeded'
                    AND completed_at >= %(start)s AND completed_at < %(end)s
                )
                SELECT bucket, SUM(submitted) AS submitted, SUM(completed) AS completed
                FROM events GROUP BY bucket ORDER BY bucket
                """,
                params,
            ).fetchall()
            option_rows = conn.execute(
                """
                SELECT DISTINCT model, owner_instance_id,
                       COALESCE(executor_instance_id,
                         CASE WHEN scheduled_at IS NOT NULL THEN owner_instance_id END
                       ) AS effective_executor_instance_id
                FROM h3_video_tasks
                WHERE created_at >= %(start)s AND created_at < %(end)s
                ORDER BY model, owner_instance_id, effective_executor_instance_id
                """,
                params,
            ).fetchall()

        metrics = dict(row or {})
        for key in (
            "total", "pending", "running", "running_backend_unavailable",
            "running_waiting_takeover", "succeeded", "failed", "canceled",
            "completed_in_window",
        ):
            metrics[key] = int(metrics.get(key) or 0)
        for key in ("avg_queue_seconds", "p95_queue_seconds", "avg_processing_seconds", "p95_processing_seconds"):
            value = metrics.get(key)
            metrics[key] = float(value) if value is not None else None
        finished = metrics["succeeded"] + metrics["failed"]
        metrics["success_rate"] = metrics["succeeded"] / finished if finished else None
        window_hours = max((end - start).total_seconds() / 3600, 1 / 3600)
        metrics["throughput_per_hour"] = metrics.pop("completed_in_window") / window_hours
        return {
            "metrics": metrics,
            "series": [self._serialize_dashboard_row(item) for item in series_rows],
            "filters": {
                "models": sorted({item["model"] for item in option_rows}),
                "instances": sorted({item["owner_instance_id"] for item in option_rows}),
                "executor_instances": sorted({
                    item["effective_executor_instance_id"]
                    for item in option_rows
                    if item.get("effective_executor_instance_id")
                }),
            },
        }

    def dashboard_tasks(
        self,
        start: datetime,
        end: datetime,
        page: int,
        page_size: int,
        status: str | None = None,
        model: str | None = None,
        instance_id: str | None = None,
        executor_instance_id: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        where, params = self._dashboard_where(status, model, instance_id, executor_instance_id)
        where += " AND created_at >= %(start)s AND created_at < %(end)s"
        params.update({
            "start": start,
            "end": end,
            "limit": page_size,
            "offset": (page - 1) * page_size,
        })
        if query:
            where += " AND (task_id::text ILIKE %(query)s OR prompt ILIKE %(query)s)"
            params["query"] = f"%{query}%"
        with self.connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS value FROM h3_video_tasks {where}", params
            ).fetchone()["value"]
            rows = conn.execute(
                f"""
                SELECT task_id, model, task_mode, prompt, resolution, ratio,
                       duration, seed, status, error_code, error_message, usage,
                       backend_url, backend_job_id, backend_status, owner_instance_id,
                       executor_instance_id, created_at, scheduled_at, started_at,
                       completed_at, deleted_at, updated_at, request_base_url,
                       lease_expires_at,
                       CASE
                         WHEN scheduled_at IS NOT NULL THEN
                           EXTRACT(EPOCH FROM (scheduled_at - created_at))
                         WHEN status = 'queued' THEN
                           EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))
                         ELSE NULL
                       END AS queue_seconds,
                       CASE
                         WHEN scheduled_at IS NULL THEN NULL
                         WHEN completed_at IS NOT NULL THEN
                           EXTRACT(EPOCH FROM (completed_at - scheduled_at))
                         WHEN status = 'running' THEN
                           EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - scheduled_at))
                         ELSE NULL
                       END AS processing_seconds
                FROM h3_video_tasks
                {where}
                ORDER BY created_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            ).fetchall()
            task_ids = [row["task_id"] for row in rows]
            assets_by_task: dict[UUID, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
            if task_ids:
                asset_rows = conn.execute(
                    """SELECT * FROM h3_task_assets
                       WHERE task_id=ANY(%(task_ids)s) AND deleted_at IS NULL
                       ORDER BY asset_id""",
                    {"task_ids": task_ids},
                ).fetchall()
                for asset in asset_rows:
                    assets_by_task[asset["task_id"]].append(asset)
            for row in rows:
                row["assets"] = assets_by_task[row["task_id"]]
        return {
            "items": [self._serialize_dashboard_task(row) for row in rows],
            "total": int(total or 0),
            "page": page,
            "page_size": page_size,
        }

    @staticmethod
    def _dashboard_where(
        status: str | None = None,
        model: str | None = None,
        instance_id: str | None = None,
        executor_instance_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        clauses = ["WHERE TRUE"]
        params: dict[str, Any] = {}
        if status:
            clauses.append("AND status=%(status)s")
            params["status"] = status
        if model:
            clauses.append("AND model=%(model)s")
            params["model"] = model
        if instance_id:
            clauses.append("AND owner_instance_id=%(instance_id)s")
            params["instance_id"] = instance_id
        if executor_instance_id:
            clauses.append(
                "AND COALESCE(executor_instance_id, CASE WHEN scheduled_at IS NOT NULL "
                "THEN owner_instance_id END)=%(executor_instance_id)s"
            )
            params["executor_instance_id"] = executor_instance_id
        return " ".join(clauses), params

    def _serialize_dashboard_task(self, row: dict[str, Any]) -> dict[str, Any]:
        lease_expires_at = row.get("lease_expires_at")
        if row.get("status") != "running":
            execution_health = "not_applicable"
        elif lease_expires_at is None or lease_expires_at <= now():
            execution_health = "waiting_takeover"
        elif row.get("error_code") == "BackendUnavailable":
            execution_health = "backend_unavailable"
        else:
            execution_health = "healthy"
        item = self._serialize_dashboard_row(row)
        item["execution_health"] = execution_health
        item["public"] = self.public_task(self._normalize(row))
        item["task_status"] = DASHBOARD_STATUS_NAMES[item["status"]]
        item["size"] = item["resolution"]
        item["submit_time"] = item["created_at"]
        item["scheduled_time"] = item["scheduled_at"]
        item["end_time"] = item["completed_at"]
        item["code"] = item["error_code"]
        item["message"] = item["error_message"]
        return item

    @staticmethod
    def _serialize_dashboard_row(row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        if item.get("task_id") is not None:
            item["task_id"] = str(item["task_id"])
        if item.get("claim_token") is not None:
            item["claim_token"] = str(item["claim_token"])
        for key, value in tuple(item.items()):
            if isinstance(value, datetime):
                item[key] = value.isoformat()
        for key in ("queue_seconds", "processing_seconds"):
            if item.get(key) is not None:
                item[key] = float(item[key])
        for key in ("submitted", "completed"):
            if item.get(key) is not None:
                item[key] = int(item[key])
        assets = item.get("assets")
        if assets is not None:
            item["assets"] = [asset if isinstance(asset, dict) else dict(asset) for asset in assets]
            for asset in item["assets"]:
                for key, value in tuple(asset.items()):
                    if isinstance(value, datetime):
                        asset[key] = value.isoformat()
                    elif isinstance(value, UUID):
                        asset[key] = str(value)
        return item

    def claim_callback(self) -> dict[str, Any] | None:
        current = now()
        with self.connect() as conn:
            row = conn.execute(
                """WITH candidate AS (
                     SELECT delivery_id FROM h3_callback_deliveries
                     WHERE delivered_at IS NULL AND abandoned_at IS NULL AND next_attempt_at <= %(now)s
                       AND (lease_expires_at IS NULL OR lease_expires_at <= %(now)s)
                     ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
                   ) UPDATE h3_callback_deliveries d
                   SET lease_owner=%(owner)s, lease_expires_at=%(expires)s
                   FROM candidate WHERE d.delivery_id=candidate.delivery_id RETURNING d.*""",
                {"now": current, "owner": self.settings.instance_id, "expires": current + timedelta(seconds=60)},
            ).fetchone()
        return self._normalize(row) if row else None

    def finish_callback(self, delivery_id: int, http_status: int | None, error: str | None) -> None:
        current = now()
        success = http_status is not None and 200 <= http_status < 300
        with self.connect() as conn:
            if success:
                conn.execute(
                    """UPDATE h3_callback_deliveries SET delivered_at=%(now)s,
                       last_http_status=%(status)s, last_error=NULL, lease_owner=NULL,
                       lease_expires_at=NULL WHERE delivery_id=%(id)s""",
                    {"now": current, "status": http_status, "id": delivery_id},
                )
            else:
                conn.execute(
                    """UPDATE h3_callback_deliveries SET attempt_count=attempt_count+1,
                       next_attempt_at=%(next)s, last_http_status=%(status)s,
                       last_error=%(error)s, lease_owner=NULL, lease_expires_at=NULL
                       WHERE delivery_id=%(id)s""",
                    {"next": current + timedelta(seconds=30), "status": http_status,
                     "error": (error or "callback failed")[:2000], "id": delivery_id},
                )

    def abandon_callback(self, delivery_id: int, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE h3_callback_deliveries SET abandoned_at=%(now)s,
                   last_error=%(error)s, lease_owner=NULL, lease_expires_at=NULL
                   WHERE delivery_id=%(id)s""",
                {"now": now(), "error": error[:2000], "id": delivery_id},
            )

    def expired_asset_paths(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT asset_id, relative_path FROM h3_task_assets WHERE deleted_at IS NULL AND expires_at <= %(now)s ORDER BY expires_at LIMIT %(limit)s",
                {"now": now(), "limit": limit},
            ).fetchall()

    def mark_asset_deleted(self, asset_id: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE h3_task_assets SET deleted_at=%(now)s WHERE asset_id=%(id)s", {"now": now(), "id": asset_id})

    def reconcile_expired_tasks(self) -> None:
        current = now()
        with self.connect() as conn:
            if not conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('h3_expired_tasks')) AS locked").fetchone()["locked"]:
                return
            rows = conn.execute(
                """UPDATE h3_video_tasks SET status='failed', error_code='TaskExpired',
                   error_message='task expired before completion', completed_at=%(now)s,
                   updated_at=%(now)s, claim_token=NULL, lease_expires_at=NULL
                   WHERE status IN ('queued','running') AND query_expires_at <= %(now)s
                   RETURNING *""",
                {"now": current},
            ).fetchall()
            for row in rows:
                self._enqueue_callback(conn, row, "failed")

    def purge_expired_metadata(self) -> None:
        cutoff = now() - timedelta(days=DASHBOARD_RETENTION_DAYS)
        with self.connect() as conn:
            if not conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('h3_metadata_purge')) AS locked").fetchone()["locked"]:
                return
            conn.execute(
                """DELETE FROM h3_video_tasks task
                   WHERE task.created_at < %(cutoff)s
                     AND task.status IN ('succeeded','failed','cancelled')
                     AND NOT EXISTS (
                       SELECT 1 FROM h3_task_assets asset
                       WHERE asset.task_id=task.task_id AND asset.deleted_at IS NULL
                     )""",
                {"cutoff": cutoff},
            )

    def _insert_asset(self, conn, task_id: UUID, asset: StoredAsset, created: datetime) -> None:
        conn.execute(
            """INSERT INTO h3_task_assets (
                 task_id, asset_type, role, relative_path, media_type, byte_size,
                 width, height, source_index, metadata, created_at, expires_at
               ) VALUES (%(task_id)s,%(asset_type)s,%(role)s,%(path)s,%(media_type)s,
                         %(byte_size)s,%(width)s,%(height)s,%(source_index)s,%(metadata)s,
                         %(created)s,%(expires)s)
               ON CONFLICT (task_id, asset_type, role, source_index) DO UPDATE SET
                 relative_path=EXCLUDED.relative_path, media_type=EXCLUDED.media_type,
                 byte_size=EXCLUDED.byte_size, width=EXCLUDED.width, height=EXCLUDED.height,
                 metadata=EXCLUDED.metadata,
                 created_at=EXCLUDED.created_at, expires_at=EXCLUDED.expires_at, deleted_at=NULL""",
            {"task_id": task_id, "asset_type": asset.asset_type, "role": asset.role,
             "path": asset.relative_path, "media_type": asset.media_type, "byte_size": asset.byte_size,
             "width": asset.width, "height": asset.height, "source_index": asset.source_index,
             "metadata": Jsonb(asset.metadata), "created": created,
             "expires": created + timedelta(days=self.settings.asset_retention_days)},
        )

    def _enqueue_callback(self, conn, task: dict[str, Any], status: str) -> None:
        callback_url = task.get("callback_url")
        if not callback_url:
            return
        normalized = dict(task)
        normalized["status"] = status
        payload = {"task": self.public_task(self._normalize(normalized))}
        conn.execute(
            """INSERT INTO h3_callback_deliveries (task_id, task_status, callback_url, payload)
               VALUES (%(task_id)s,%(status)s,%(url)s,%(payload)s)
               ON CONFLICT (task_id, task_status) DO NOTHING""",
            {"task_id": task["task_id"], "status": status, "url": callback_url, "payload": Jsonb(payload)},
        )

    def public_task(self, task: dict[str, Any], base_url: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": task["task_id"],
            "model": task["model"],
            "status": task["status"],
            "created_at": int(task["created_at"].timestamp()),
            "updated_at": int(task["updated_at"].timestamp()),
            "resolution": task["resolution"],
            "duration": task["duration"],
            "usage": task.get("usage") or {},
            "ratio": task["ratio"],
            "task_type": "generation",
            "modality": "video",
        }
        if task["status"] == "succeeded":
            content_path = f"/v2/video_generation/{task['task_id']}/content"
            # 配置优先:改了 H3_PUBLIC_BASE_URL,存量任务和 Callback 推送的 URL 也立刻正确,
            # 不用回填 request_base_url(它是建任务时那次请求推导出来的)。
            effective_base_url = (
                self.settings.public_base_url or base_url or task.get("request_base_url") or ""
            ).rstrip("/")
            result["content"] = {"url": f"{effective_base_url}{content_path}"}
        if task["status"] == "failed":
            result["error"] = {"code": task.get("error_code") or "server_error", "message": task.get("error_message") or "task failed"}
        return result

    @staticmethod
    def _normalize(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        output = dict(row)
        for key in ("task_id", "claim_token"):
            if output.get(key) is not None:
                output[key] = str(output[key])
        assets = output.get("assets")
        if assets is not None:
            output["assets"] = [dict(item) for item in assets]
        return output
