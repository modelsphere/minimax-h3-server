from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from .config import Settings
from .models import APIError, MAX_REQUEST_BYTES, error_body, parse_video_request
from .sglang import SGLangClient
from .storage import SharedStorage, StoredAsset, validate_remote_url
from .task_store import DASHBOARD_RETENTION_DAYS, TaskStore
from .worker import CallbackWorker, MaintenanceWorker, TaskWorker


logger = logging.getLogger("h3.server")
DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")
# /metrics 抓后端队列的缓存时长:别让每次抓取都打一次 SGLang
BACKEND_METRICS_TTL_SECONDS = 5.0
# 后端队列里"总要报出来"的状态:即使当前为 0 也输出,避免 series 断档。
# in_progress 是 SGLang 跑起来后的状态,queued 是排队 —— 这两个是看板真正关心的。
KNOWN_BACKEND_STATES = ("queued", "in_progress")
DASHBOARD_STATUSES = {
    "PENDING": "queued",
    "QUEUED": "queued",
    "RUNNING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "CANCELED": "cancelled",
    "CANCELLED": "cancelled",
}


def _request_id() -> str:
    return uuid4().hex


def _error(error: APIError, request_id: str | None = None) -> JSONResponse:
    request_id = request_id or _request_id()
    return JSONResponse(error_body(error, request_id), status_code=error.status_code)


def _request_base_url(request: Request) -> str:
    scheme = request.headers.get(
        "x-forwarded-proto", request.url.scheme
    ).split(",", 1)[0].strip().lower()
    if scheme not in {"http", "https"}:
        scheme = request.url.scheme
    host = request.headers.get(
        "x-forwarded-host", request.headers.get("host", "")
    ).split(",", 1)[0].strip()
    if not host or any(character in host for character in "/\\\r\n"):
        host = request.url.netloc
    return f"{scheme}://{host}"


def _parse_dashboard_time(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise APIError(400, "bad_request_error", f"{field_name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dashboard_filters(start: str | None, end: str | None, status: str | None) -> dict:
    end_time = _parse_dashboard_time(end, "end") if end else datetime.now(timezone.utc)
    start_time = _parse_dashboard_time(start, "start") if start else end_time - timedelta(hours=24)
    if start_time >= end_time:
        raise APIError(400, "bad_request_error", "start must be earlier than end")
    if end_time - start_time > timedelta(days=DASHBOARD_RETENTION_DAYS):
        raise APIError(400, "bad_request_error", "dashboard time range cannot exceed 30 days")
    normalized_status = None
    if status and status.strip():
        normalized_status = DASHBOARD_STATUSES.get(status.strip().upper())
        if normalized_status is None:
            raise APIError(400, "bad_request_error", "status is invalid")
    return {"start": start_time, "end": end_time, "status": normalized_status}


def _dashboard_bucket_seconds(duration: timedelta) -> int:
    if duration <= timedelta(hours=1):
        return 5 * 60
    if duration <= timedelta(days=1):
        return 60 * 60
    if duration <= timedelta(days=7):
        return 6 * 60 * 60
    return 24 * 60 * 60


def _clean_filter(value: str | None) -> str | None:
    value = value.strip() if value else ""
    return value or None


def _asset_is_available(asset: dict) -> bool:
    value = asset.get("expires_at")
    if not value:
        return False
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value > datetime.now(timezone.utc)


def _add_dashboard_asset_links(task: dict, base_url: str) -> None:
    assets = task.pop("assets", [])
    output = next(
        (item for item in assets if item["asset_type"] == "output" and item["role"] == "generated_video"),
        None,
    )
    task["video_url"] = None
    task["video_expires_at"] = output.get("expires_at") if output else None
    if task["status"] == "succeeded":
        task["public"].pop("content", None)
    if task["status"] == "succeeded" and output and _asset_is_available(output):
        task["video_url"] = f"{base_url}/v2/video_generation/{task['task_id']}/content"
        task["public"]["content"] = {"url": task["video_url"]}

    thumbnails = {
        (item["role"], item.get("source_index", 0)): item
        for item in assets
        if item["asset_type"] == "thumbnail" and _asset_is_available(item)
    }
    input_assets = []
    for asset in assets:
        if asset["asset_type"] != "input" or not _asset_is_available(asset):
            continue
        role = asset["role"]
        source_index = asset.get("source_index", 0)
        path = (
            f"{base_url}/api/v1/dashboard/tasks/{task['task_id']}/assets/{role}"
            f"?source_index={source_index}"
        )
        thumbnail = thumbnails.get((role, source_index))
        input_assets.append({
            "role": role,
            "source_index": source_index,
            "media_type": asset["media_type"],
            "metadata": asset.get("metadata") or {},
            "expires_at": asset["expires_at"],
            "url": path,
            "thumbnail_url": f"{path}&thumbnail=true" if thumbnail else None,
        })
    task["input_assets"] = input_assets
    task["input_image_sync_status"] = (
        "SYNCED" if input_assets else "EXPIRED"
        if any(item["asset_type"] == "input" for item in assets) else "NOT_APPLICABLE"
    )
    if input_assets:
        first_input = next(
            (item for item in input_assets if item["media_type"].startswith("image/")),
            input_assets[0],
        )
        task["input_image_url"] = first_input["url"]
        task["input_thumbnail_url"] = first_input["thumbnail_url"] or first_input["url"]
        task["input_image_expires_at"] = first_input["expires_at"]
    else:
        task["input_image_url"] = None
        task["input_thumbnail_url"] = None
        task["input_image_expires_at"] = None


async def _validate_callback(url: str, settings: Settings) -> None:
    validate_remote_url(url, settings.allow_private_urls)
    challenge = uuid4().hex
    try:
        async with httpx.AsyncClient(timeout=3, follow_redirects=False) as client:
            async with client.stream("POST", url, json={"challenge": challenge}) as response:
                if response.status_code != 200:
                    data = None
                else:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 64 * 1024:
                            raise APIError(400, "bad_request_error", "callback_url challenge response is too large")
                    data = json.loads(body)
    except (httpx.HTTPError, ValueError) as exc:
        raise APIError(400, "bad_request_error", f"callback_url challenge failed: {exc}") from exc
    if not isinstance(data, dict) or data.get("challenge") != challenge:
        raise APIError(400, "bad_request_error", "callback_url did not echo the challenge")


def _asset(
    task: dict,
    asset_type: str,
    role: str | None = None,
    source_index: int | None = None,
) -> dict | None:
    for item in task.get("assets", []):
        if (
            item["asset_type"] == asset_type
            and (role is None or item["role"] == role)
            and (source_index is None or item.get("source_index", 0) == source_index)
        ):
            return item
    return None


def _file_response(path: Path, media_type: str, request: Request, filename: str | None = None):
    if not path.is_file():
        return _error(APIError(404, "not_found_error", "asset is no longer available"))
    size = path.stat().st_size
    range_header = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=300"}
    if not range_header:
        return FileResponse(path, media_type=media_type, filename=filename, headers=headers)
    try:
        unit, value = range_header.split("=", 1)
        if unit.strip().lower() != "bytes" or "," in value:
            raise ValueError
        start_text, end_text = value.split("-", 1)
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            start, end = max(size - suffix, 0), size - 1
        if start < 0 or end < start or start >= size:
            raise ValueError
        end = min(end, size - 1)
    except (ValueError, TypeError):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def chunks():
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining:
                data = handle.read(min(1024 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})
    return StreamingResponse(chunks(), status_code=206, media_type=media_type, headers=headers)


def create_app(
    settings: Settings | None = None,
    store: TaskStore | None = None,
    storage: SharedStorage | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or TaskStore(settings)
    storage = storage or SharedStorage(
        settings.storage_root,
        settings.sglang_media_root,
        settings.allow_private_urls,
        settings.media_timeout_seconds,
    )
    def base_url_for(request: Request) -> str:
        """下载 URL 的基址:配了 H3_PUBLIC_BASE_URL 就用它,否则按请求头推导。"""
        return settings.public_base_url or _request_base_url(request)

    stop_event = threading.Event()      # 硬停:连在途任务也放手
    drain_event = threading.Event()     # 排空:只停止领新任务,手上那条跑完
    threads: list[threading.Thread] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate()
        storage.init()
        store.init()
        app.state.ready = True
        task_worker = TaskWorker(settings, store, storage, stop_event, drain_event)
        if settings.start_workers:
            targets = [
                task_worker.run,
                CallbackWorker(settings, store, stop_event).run,
                MaintenanceWorker(store, storage, stop_event).run,
            ]
            for index, target in enumerate(targets):
                thread = threading.Thread(target=target, name=f"h3-worker-{index}", daemon=True)
                thread.start()
                threads.append(thread)
        # SIGTERM 一到就置排空 —— **不能**只依赖下面 finally 里那次。
        # uvicorn 的关闭顺序是"先停止接受新连接、等在途 HTTP 请求结束,再跑 lifespan shutdown";
        # 若有客户端正在下载大视频(StreamingResponse 长连接),lifespan shutdown 会被推迟,
        # 而这段时间里 TaskWorker 还在继续领新任务 —— 又回到"永远排不空"。
        #
        # 这里链式包装:先置排空,再把信号交回 uvicorn 原来的 handler,不打断它的关闭流程。
        # 装在 lifespan startup 里是有意的:uvicorn 在 serve() 开头装自己的 handler,
        # 这里晚于它,才能拿到并链住它(反过来会被它覆盖)。信号只能在主线程装,而
        # lifespan 跑在主线程的事件循环里,满足条件。
        app.state.drain_event = drain_event
        try:
            previous = signal.getsignal(signal.SIGTERM)

            def _on_sigterm(sig, frame):
                drain_event.set()
                logger.info("event=sigterm_received instance=%s -> 停止领新任务", settings.instance_id)
                if callable(previous):
                    previous(sig, frame)

            signal.signal(signal.SIGTERM, _on_sigterm)
            app.state.previous_sigterm = previous
        except (ValueError, OSError):
            # 非主线程 / 不支持信号的环境(测试里常见):退化成只靠 lifespan shutdown
            logger.debug("SIGTERM handler 未安装(非主线程或平台不支持)")
        try:
            yield
        finally:
            app.state.ready = False
            # ── 优雅关闭三步 ──────────────────────────────────────────
            # 1) 停止领新任务。TaskWorker 是从**数据库**拉活的,Pod 进 Terminating 只会把它
            #    从 Service endpoints 摘掉,拦不住它继续 claim —— 不先停就会边等边接新活,
            #    永远等不到空。(SIGTERM handler 通常已经置过了,这里兜住直接调 shutdown 的路径。)
            drain_event.set()
            # 2) 等手上那条跑完。它的记账(轮询状态、下载产物、标 succeeded)只有本进程做得了;
            #    提前放手就得靠别的副本在租约过期后接管,而单副本的池根本没有"别的副本"。
            #    判断依据是 worker 自己的 busy 标志,不查库 —— 库里挂着本实例名的 running 任务
            #    可能是别的实例留下的行,会白等到超时。(历史上 Server 是 StatefulSet、
            #    Pod 名复用,这种"同名旧 Pod 的僵尸"很常见;改 Deployment 后名字不复用,
            #    但判断自己忙不忙看进程内状态仍然是对的做法。)
            deadline = time.monotonic() + settings.shutdown_drain_seconds
            waited = False
            while task_worker.busy.is_set() and time.monotonic() < deadline:
                if not waited:
                    logger.info(
                        "event=shutdown_draining instance=%s capability=%s timeout=%.0fs",
                        settings.instance_id, settings.capability, settings.shutdown_drain_seconds,
                    )
                    waited = True
                await asyncio.sleep(1)
            if waited:
                logger.info(
                    "event=shutdown_drained instance=%s still_busy=%s",
                    settings.instance_id, task_worker.busy.is_set(),
                )
            # 3) 硬停:连在途的也放手(正常情况下这时已经没有在途了)
            stop_event.set()
            for thread in threads:
                thread.join(timeout=5)
            # 还原信号处理:同一个进程里反复建 app 时(测试),别让上一个 app 的 handler 留下来
            prev = getattr(app.state, "previous_sigterm", None)
            if prev is not None:
                try:
                    signal.signal(signal.SIGTERM, prev)
                except (ValueError, OSError):
                    pass

    app = FastAPI(title="MiniMax H3 Self-hosted API", version="0.1.0", lifespan=lifespan)
    app.state.ready = False
    app.state.settings = settings
    app.state.store = store
    app.state.storage = storage

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError):
        return _error(exc)

    @app.post("/v2/video_generation")
    async def create_video(request: Request):
        request_id = _request_id()
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            return _error(APIError(400, "bad_request_error", "request body exceeds 64 MB"), request_id)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _error(APIError(400, "bad_request_error", "request body must be valid JSON"), request_id)
        try:
            parsed = parse_video_request(payload)
            if parsed.required_capability != settings.capability:
                raise APIError(501, "not_implemented_error", f"{parsed.mode} is not enabled on this service")
            if parsed.callback_url:
                await _validate_callback(parsed.callback_url, settings)
            task_id = str(uuid4())
            seed = random.SystemRandom().randrange(0, 2**31)
            assets: list[StoredAsset] = []
            try:
                for media in parsed.media:
                    stored, thumbnail = await asyncio.to_thread(storage.persist_input, task_id, media)
                    assets.append(stored)
                    if thumbnail:
                        assets.append(thumbnail)
                if parsed.mode == "ref2va":
                    storage.validate_reference_totals(assets)
                await asyncio.to_thread(
                    store.insert_task, task_id, payload, parsed, seed, assets, base_url_for(request)
                )
            except Exception:
                for asset in assets:
                    storage.delete(asset.relative_path)
                raise
        except APIError as exc:
            return _error(exc, request_id)
        except Exception:
            logger.exception("failed to create task")
            return _error(APIError(500, "server_error", "failed to create video task"), request_id)
        return {"task_id": task_id}

    @app.get("/v2/query/video_generation/{task_id}")
    def query_task(task_id: str, request: Request):
        task = store.get_task(task_id)
        if task is None:
            return _error(APIError(404, "not_found_error", "task was not found"))
        return {"task": store.public_task(task, base_url_for(request))}

    @app.get("/v2/query/video_generation")
    def list_tasks(
        request: Request,
        page_num: int = Query(1, ge=1),
        page_size: int = Query(20, ge=-1, le=100),
        status: str | None = Query(None, alias="filter.status"),
        model: str | None = Query(None, alias="filter.model"),
        task_type: str | None = Query(None, alias="filter.task_type"),
        task_ids: list[str] | None = Query(None, alias="filter.task_ids"),
    ):
        if page_size == 0:
            return _error(APIError(400, "bad_request_error", "page_size must be -1 or an integer from 1 to 100"))
        if status and status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            return _error(APIError(400, "bad_request_error", "invalid task status filter"))
        result = store.list_tasks(page_num, page_size, status, model, task_type, task_ids)
        base_url = base_url_for(request)
        return {"items": [store.public_task(task, base_url) for task in result["items"]], "total": result["total"]}

    @app.delete("/v2/video_generation/{task_id}")
    def delete_task(task_id: str):
        result = store.cancel_or_delete(task_id)
        if result is None:
            return _error(APIError(404, "not_found_error", "task was not found"))
        action, task = result
        if action == "unsupported":
            return _error(APIError(400, "bad_request_error", f"cannot delete a task in {task['status']} status"))
        if action == "cancelled" and task.get("backend_job_id"):
            SGLangClient(settings, storage, task.get("backend_url")).cancel(task["backend_job_id"])
        return {"task_id": task_id, "action": action, "status": action}

    @app.get("/v2/video_generation/{task_id}/content")
    def task_content(task_id: str, request: Request):
        task = store.get_task(task_id, include_expired=True)
        if task is None or task["status"] != "succeeded":
            return _error(APIError(404, "not_found_error", "video is not available"))
        asset = _asset(task, "output", "generated_video")
        if not asset:
            return _error(APIError(404, "not_found_error", "video is not available"))
        return _file_response(storage.resolve(asset["relative_path"]), asset["media_type"], request, "output.mp4")

    @app.get("/api/v1/dashboard/tasks/{task_id}/assets/{role}")
    def dashboard_asset(
        task_id: str,
        role: str,
        request: Request,
        thumbnail: bool = Query(False),
        source_index: int = Query(0, ge=0),
    ):
        task = store.get_task(task_id, include_expired=True, include_deleted=True)
        if task is None:
            return _error(APIError(404, "not_found_error", "task was not found"))
        asset = _asset(task, "thumbnail" if thumbnail else "input", role, source_index)
        if asset is None:
            return _error(APIError(404, "not_found_error", "asset was not found"))
        return _file_response(storage.resolve(asset["relative_path"]), asset["media_type"], request)

    @app.get("/api/v1/dashboard/tasks")
    def dashboard_tasks(
        request: Request,
        start: str | None = None,
        end: str | None = None,
        status: str | None = None,
        model: str | None = None,
        instance_id: str | None = None,
        executor_instance_id: str | None = None,
        q: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=100),
    ):
        filters = _dashboard_filters(start, end, status)
        result = store.dashboard_tasks(
            start=filters["start"],
            end=filters["end"],
            page=page,
            page_size=page_size,
            status=filters["status"],
            model=_clean_filter(model),
            instance_id=_clean_filter(instance_id),
            executor_instance_id=_clean_filter(executor_instance_id),
            query=_clean_filter(q),
        )
        base_url = base_url_for(request)
        for task in result["items"]:
            _add_dashboard_asset_links(task, base_url)
        return result

    @app.get("/api/v1/dashboard/metrics")
    def dashboard_metrics(
        start: str | None = None,
        end: str | None = None,
        status: str | None = None,
        model: str | None = None,
        instance_id: str | None = None,
        executor_instance_id: str | None = None,
    ):
        filters = _dashboard_filters(start, end, status)
        result = store.dashboard_metrics(
            start=filters["start"],
            end=filters["end"],
            bucket_seconds=_dashboard_bucket_seconds(filters["end"] - filters["start"]),
            status=filters["status"],
            model=_clean_filter(model),
            instance_id=_clean_filter(instance_id),
            executor_instance_id=_clean_filter(executor_instance_id),
        )
        result["window"] = {
            "start": filters["start"].isoformat(),
            "end": filters["end"].isoformat(),
        }
        return result

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz")
    def ready():
        # readiness 决定 Service 端点:数据库不可用时这个实例什么都做不了,
        # 如实报 503 让 k8s 把它摘掉。注意 /healthz(liveness)不能这么做 ——
        # 否则数据库一挂,所有副本会被 kubelet 反复重启。
        if not app.state.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        if not settings.readiness_checks_database:
            return {"status": "ready", "capability": settings.capability}
        db_ok, db_detail = store.health()
        if not db_ok:
            return JSONResponse(
                {"status": "degraded", "database": db_detail}, status_code=503
            )
        return {"status": "ready", "capability": settings.capability}

    def _escape_label(value: Any) -> str:
        """Prometheus 文本格式的标签值转义:反斜杠、双引号、换行三样,顺序不能反
        (先转反斜杠,否则会把自己插入的转义符再转一遍)。

        目前的标签值都是枚举(status/capability)或 pod 名,本来也不含这些字符;
        写全是为了以后有人把 error_code 之类塞进标签时不会静默破坏 exposition 格式。
        """
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )

    def _metric_lines(name: str, help_text: str, kind: str, samples: list[tuple[dict, Any]]) -> list[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
        for labels, value in samples:
            if value is None:
                continue
            if labels:
                rendered = ",".join(
                    f'{key}="{_escape_label(val)}"' for key, val in sorted(labels.items())
                )
                lines.append(f"{name}{{{rendered}}} {value}")
            else:
                lines.append(f"{name} {value}")
        return lines

    def _backend_snapshot() -> dict[str, Any]:
        """后端(SGLang)队列。SGLang 没有 /metrics,只能数 GET /v1/videos。

        每次抓取都打一次后端会给它添负担,所以缓存 BACKEND_METRICS_TTL 秒。
        后端不可达时返回 up=0,而不是让整个 /metrics 挂掉。
        """
        cached = getattr(app.state, "backend_metrics", None)
        now_ts = time.monotonic()
        if cached and now_ts - cached["at"] < BACKEND_METRICS_TTL_SECONDS:
            return cached["value"]
        value: dict[str, Any] = {"up": 0, "states": {}}
        try:
            # 用独立的短超时(见 Settings.metrics_backend_timeout_seconds):
            # 必须明显短于 Prometheus 的 scrapeTimeout,否则后端假死时连数据库那部分指标也丢。
            jobs = SGLangClient(settings, storage,
                                timeout=settings.metrics_backend_timeout_seconds).list_jobs()
            # 后端可用时**总是**给出已知状态的计数,空队列输出 0 而不是不输出 ——
            # 否则"有任务在跑但没有排队"这个正常场景下,queued 这条 series 直接消失,
            # 看板和告警的 PromQL 返回空结果(断档),看起来像采集挂了。
            states: dict[str, int] = {s: 0 for s in KNOWN_BACKEND_STATES}
            for job in jobs:
                st = str(job.get("status") or "unknown")
                states[st] = states.get(st, 0) + 1
            value = {"up": 1, "states": states}
        except Exception:
            logger.warning("backend metrics scrape failed", exc_info=True)
        app.state.backend_metrics = {"at": now_ts, "value": value}
        return value

    @app.get("/metrics")
    def metrics():
        end = datetime.now(timezone.utc)
        data = store.dashboard_metrics(end - timedelta(hours=24), end, 3600)["metrics"]
        lines = ["# TYPE h3_tasks gauge"]
        for status in ("queued", "running", "succeeded", "failed", "cancelled"):
            metric_key = {"queued": "pending", "cancelled": "canceled"}.get(status, status)
            lines.append(f'h3_tasks{{status="{status}"}} {data.get(metric_key, 0)}')
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


    # 说明:/metrics 保持原样(它的 h3_tasks 是"24 小时窗口内创建且当前为该状态"的计数,
    # 已有看板/告警依赖它,不能改名也不能加标签)。排队/运行的即时指标一律放到
    # /metrics/queue,用 h3_queue_* 这套新名字,两边互不影响。
    @app.get("/metrics/queue")
    def metrics_queue():
        snapshot = store.metrics_snapshot()
        backend = _backend_snapshot()
        lines: list[str] = []

        lines += _metric_lines(
            "h3_queue_tasks",
            "本副本 capability 下各状态的任务数(同池多副本报同一份值,service/capability 1:1,查询用 max by(service) 去重、别 sum())",
            "gauge",
            [
                ({"status": row["status"], "capability": row["capability"]}, row["count"])
                for row in snapshot["by_status"]
            ],
        )
        lines += _metric_lines(
            "h3_queue_oldest_queued_age_seconds",
            "最老的排队任务已经等了多久(秒);没有排队任务时不上报",
            "gauge",
            [({}, snapshot["oldest_queued_age_seconds"])],
        )
        lines += _metric_lines(
            "h3_queue_oldest_running_age_seconds",
            "最老的运行中任务已经跑了多久(秒);卡住的任务会让它一直涨",
            "gauge",
            [({}, snapshot["oldest_running_age_seconds"])],
        )
        lines += _metric_lines(
            "h3_queue_waiting_takeover",
            "running 但租约已过期/为空、等待其它副本接管的任务数(持续大于 0 = 该 capability 的副本全挂或全卡)",
            "gauge",
            [({}, snapshot["waiting_takeover"])],
        )
        lines += _metric_lines(
            "h3_queue_backend_unavailable",
            "running 且当前与后端(SGLang)失联、仍在租约内重试的任务数(持续大于 0 = SGLang 有问题)",
            "gauge",
            [({}, snapshot["backend_unavailable"])],
        )
        lines += _metric_lines(
            "h3_queue_running_on_instance",
            "本副本自己正在执行的任务数(每副本各报各的;标签叫 h3_instance 是为了避开 Prometheus 自带的 instance)",
            "gauge",
            [({"h3_instance": settings.instance_id}, snapshot["running_on_instance"])],
        )
        lines += _metric_lines(
            "h3_queue_completed_24h",
            "过去 24 小时内完成的任务数,按终态分",
            "gauge",
            [({"status": row["status"]}, row["count"]) for row in snapshot["completed_24h"]],
        )
        lines += _metric_lines(
            "h3_queue_backend_up",
            "本副本对应的 SGLang 后端是否可达(sidecar 下一个 Server 只有一个后端,不打 backend 标签)",
            "gauge",
            [({"capability": settings.capability}, backend["up"])],
        )
        lines += _metric_lines(
            "h3_queue_backend_jobs",
            "本 Pod 里 SGLang 上的作业数,按后端自己的状态分(state=queued 即后端排队深度)。"
            "带 capability 是为了能按能力池筛选 —— 池信息只在 Pod 名里,否则只能 pod=~\".*fl2va.*\" 这种正则",
            "gauge",
            [
                ({"capability": settings.capability, "state": state}, count)
                for state, count in sorted(backend["states"].items())
            ],
        )
        lines += _metric_lines(
            "h3_queue_build_info",
            "构建/部署信息,值恒为 1",
            "gauge",
            [({"capability": settings.capability, "h3_instance": settings.instance_id}, 1)],
        )
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


def _configure_logging() -> None:
    """让 h3.* 的日志真的输出。

    在此之前代码里从没配过 logging:root logger 没有任何 handler,于是 Python 的
    lastResort 机制只放行 **WARNING 及以上**,而且没有时间戳。后果是
    `logger.info` 全部被丢掉 —— 包括 api.md 里专门承诺"可检索"的那套结构化事件日志
    (task_lease_renewed / task_takeover_started / task_backend_recovered / 排空关闭 …)。
    线上验证过:server 容器的日志里只有 uvicorn 的访问日志,一条业务事件都没有。

    级别用 H3_LOG_LEVEL 调(默认 INFO)。已经有 handler 时不覆盖 —— 尊重外部注入的配置。
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=os.getenv("H3_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    _configure_logging()
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port)


if __name__ == "__main__":
    main()
