import base64
import io
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from h3_service.config import Settings
from h3_service.server import create_app
from h3_service.sglang import SGLangClient
from h3_service.storage import SharedStorage
from h3_service.task_store import TaskStore


def image_data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (320, 256), "teal").save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class MemoryStore:
    def __init__(self, settings):
        self.settings = settings
        self.tasks = {}

    def init(self):
        pass

    def insert_task(self, task_id, payload, parsed, seed, assets, request_base_url):
        created = datetime.now(timezone.utc)
        task = {
            "task_id": task_id, "model": parsed.model, "task_mode": parsed.mode,
            "required_capability": parsed.required_capability, "content": parsed.content,
            "prompt": parsed.prompt, "resolution": parsed.resolution, "ratio": parsed.ratio,
            "duration": parsed.duration, "seed": seed, "status": "queued", "usage": {},
            "created_at": created, "updated_at": created, "query_expires_at": created + timedelta(days=7),
            "deleted_at": None, "backend_job_id": None, "assets": [asset.__dict__ for asset in assets],
            "request_base_url": request_base_url,
        }
        self.tasks[task_id] = task
        return task

    def get_task(self, task_id, include_expired=False, include_deleted=False):
        task = self.tasks.get(task_id)
        if task and (include_deleted or not task["deleted_at"]):
            return task
        return None

    def public_task(self, task, base_url=None):
        result = {
            "id": task["task_id"], "model": task["model"], "status": task["status"],
            "created_at": int(task["created_at"].timestamp()), "updated_at": int(task["updated_at"].timestamp()),
            "resolution": task["resolution"], "duration": task["duration"], "usage": task["usage"],
            "ratio": task["ratio"], "task_type": "generation", "modality": "video",
        }
        if task["status"] == "succeeded":
            result["content"] = {
                "url": f"{base_url or task['request_base_url']}/v2/video_generation/{task['task_id']}/content"
            }
        return result

    def list_tasks(self, page, page_size, status, model, task_type, task_ids):
        items = list(self.tasks.values())
        if page_size == -1:
            return {"items": items, "total": len(items)}
        return {"items": items[(page - 1) * page_size:page * page_size], "total": len(items)}

    def cancel_or_delete(self, task_id):
        task = self.tasks.get(task_id)
        if not task:
            return None
        task["status"] = "cancelled"
        task["updated_at"] = datetime.now(timezone.utc)
        return "cancelled", task

    def dashboard_tasks(self, start, end, page, page_size, **_filters):
        status_names = {
            "queued": "PENDING", "running": "RUNNING", "succeeded": "SUCCEEDED",
            "failed": "FAILED", "cancelled": "CANCELED",
        }
        items = []
        for source in self.tasks.values():
            item = dict(source)
            item.update({
                "task_status": status_names[item["status"]],
                "size": item["resolution"],
                "submit_time": item["created_at"].isoformat(),
                "scheduled_time": None,
                "end_time": None,
                "queue_seconds": None,
                "processing_seconds": None,
                "code": None,
                "message": None,
                "public": self.public_task(item),
            })
            items.append(item)
        return {"items": items, "total": len(items), "page": page, "page_size": page_size}

    def health(self, ttl=5.0, timeout=2.0):
        if getattr(self, "db_down", False):
            return False, "OperationalError: connection refused"
        return True, "ok"

    def metrics_snapshot(self):
        by_status = {}
        for task in self.tasks.values():
            key = (task["status"], task["required_capability"])
            by_status[key] = by_status.get(key, 0) + 1
        return {
            "by_status": [
                {"status": status, "capability": capability, "count": count}
                for (status, capability), count in by_status.items()
            ],
            "completed_24h": [],
            "oldest_queued_age_seconds": 12.5 if any(t["status"] == "queued" for t in self.tasks.values()) else None,
            "oldest_running_age_seconds": None,
            "waiting_takeover": 0,
            "backend_unavailable": 0,
            "running_on_instance": 0,
        }

    def dashboard_metrics(self, start, end, bucket_seconds, **_filters):
        counts = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0, "canceled": 0}
        mapping = {"queued": "pending", "cancelled": "canceled"}
        for task in self.tasks.values():
            key = mapping.get(task["status"], task["status"])
            counts[key] += 1
        return {
            "metrics": counts | {
                "total": len(self.tasks), "success_rate": None,
                "avg_queue_seconds": None, "p95_queue_seconds": None,
                "avg_processing_seconds": None, "p95_processing_seconds": None,
                "throughput_per_hour": 0,
            },
            "series": [],
            "filters": {"models": ["MiniMax-H3"], "instances": [], "executor_instances": []},
        }


def test_official_create_query_list_and_cancel(tmp_path):
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang",
        start_workers=False,
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    app = create_app(settings, store, storage)
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A fox in snow"}],
        "resolution": "768P", "duration": 5, "ratio": "16:9",
    }
    with TestClient(app) as client:
        created = client.post("/v2/video_generation", json=payload)
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        queried = client.get(f"/v2/query/video_generation/{task_id}")
        assert queried.json()["task"]["status"] == "queued"
        assert client.get("/v2/query/video_generation").json()["total"] == 1
        cancelled = client.delete(f"/v2/video_generation/{task_id}")
        assert cancelled.json()["action"] == "cancelled"


def test_list_tasks_accepts_page_size_minus_one_for_all_records(tmp_path):
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
    )
    store = MemoryStore(settings)
    app = create_app(settings, store, SharedStorage(tmp_path, "/data/minimax-h3", False, 2))
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A fox in snow"}],
        "resolution": "768P", "duration": 5, "ratio": "16:9",
    }

    with TestClient(app) as client:
        client.post("/v2/video_generation", json=payload)
        client.post("/v2/video_generation", json=payload)
        response = client.get("/v2/query/video_generation?page_size=-1")
        invalid = client.get("/v2/query/video_generation?page_size=0")

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert len(response.json()["items"]) == 2
    assert invalid.status_code == 400


def test_dashboard_matches_reference_controls_and_api_shape(tmp_path):
    settings = Settings(
        database_url="postgresql://unused",
        storage_root=tmp_path,
        sglang_url="http://sglang",
        start_workers=False,
    )
    store = MemoryStore(settings)
    app = create_app(settings, store, SharedStorage(tmp_path, "/data/minimax-h3", False, 2))

    with TestClient(app) as client:
        page = client.get("/dashboard")
        metrics = client.get("/api/v1/dashboard/metrics")
        tasks = client.get("/api/v1/dashboard/tasks?page=1&page_size=50")
        prometheus = client.get("/metrics")
        too_long = client.get(
            "/api/v1/dashboard/metrics",
            params={"start": "2026-07-01T00:00:00Z", "end": "2026-08-01T00:00:00Z"},
        )
        invalid_status = client.get("/api/v1/dashboard/metrics", params={"status": "unknown"})

    assert page.status_code == 200
    for control_id in (
        "range-buttons", "status-filter", "model-filter", "request-instance-filter",
        "executor-instance-filter", "search", "refresh-button", "prev-page",
        "next-page", "drawer-close", "m-running-health",
    ):
        assert f'id="{control_id}"' in page.text
    assert "RUNNING · 后端失联" in page.text
    assert "RUNNING · 待接管" in page.text
    assert set(metrics.json()) == {"metrics", "series", "filters", "window"}
    assert tasks.json() == {"items": [], "total": 0, "page": 1, "page_size": 50}
    assert prometheus.status_code == 200
    assert 'h3_tasks{status="queued"} 0' in prometheus.text
    assert too_long.status_code == 400
    assert invalid_status.status_code == 400


def test_content_urls_use_current_request_origin(tmp_path):
    settings = Settings(
        database_url="postgresql://unused",
        storage_root=tmp_path,
        sglang_url="http://sglang",
        start_workers=False,
    )
    store = MemoryStore(settings)
    app = create_app(settings, store, SharedStorage(tmp_path, "/data/minimax-h3", False, 2))
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A fox in snow"}],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
    }
    forwarded = {"x-forwarded-proto": "http", "x-forwarded-host": "203.0.113.10:8080"}
    with TestClient(app) as client:
        task_id = client.post("/v2/video_generation", json=payload, headers=forwarded).json()["task_id"]
        assert store.tasks[task_id]["request_base_url"] == "http://203.0.113.10:8080"
        store.tasks[task_id]["status"] = "succeeded"
        store.tasks[task_id]["assets"].append({
            "asset_type": "output",
            "role": "generated_video",
            "media_type": "video/mp4",
            "expires_at": datetime.now(timezone.utc) + timedelta(days=30),
        })

        router_url = f"http://203.0.113.10:8080/v2/video_generation/{task_id}/content"
        queried = client.get(f"/v2/query/video_generation/{task_id}", headers=forwarded)
        assert queried.json()["task"]["content"]["url"] == router_url
        listed = client.get("/v2/query/video_generation", headers=forwarded)
        assert listed.json()["items"][0]["content"]["url"] == router_url
        dashboard = client.get("/api/v1/dashboard/tasks", headers=forwarded)
        assert dashboard.json()["items"][0]["public"]["content"]["url"] == router_url

        direct = client.get(f"/v2/query/video_generation/{task_id}")
        assert direct.json()["task"]["content"]["url"] == (
            f"http://testserver/v2/video_generation/{task_id}/content"
        )


def test_callback_payload_url_falls_back_to_creation_origin(tmp_path):
    settings = Settings(
        database_url="postgresql://unused",
        storage_root=tmp_path,
        sglang_url="http://sglang",
        start_workers=False,
    )
    created = datetime.now(timezone.utc)
    task = {
        "task_id": "00000000-0000-0000-0000-000000000001",
        "model": "MiniMax-H3",
        "status": "succeeded",
        "created_at": created,
        "updated_at": created,
        "resolution": "768P",
        "duration": 5,
        "usage": {},
        "ratio": "16:9",
        "request_base_url": "http://203.0.113.10:8080",
    }

    public = TaskStore(settings).public_task(task)

    assert public["content"]["url"] == (
        "http://203.0.113.10:8080/v2/video_generation/"
        "00000000-0000-0000-0000-000000000001/content"
    )


def test_ref_is_rejected_by_fl2va_service(tmp_path):
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
    )
    store = MemoryStore(settings)
    app = create_app(settings, store, SharedStorage(tmp_path, "/data/minimax-h3", False, 2))
    payload = {
        "model": "MiniMax-H3",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://example.com/ref.jpg"}},
        ],
        "resolution": "768P", "duration": 5, "ratio": "adaptive",
    }
    with TestClient(app) as client:
        response = client.post("/v2/video_generation", json=payload)
    assert response.status_code == 501
    assert store.tasks == {}


@pytest.mark.parametrize("frame_roles", [
    [], ["first_frame"], ["last_frame"], ["last_frame", "first_frame"],
])
@pytest.mark.parametrize("turbo", [False, True])
def test_ref_is_created_by_ref2va_service(tmp_path, frame_roles, turbo):
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang",
        capability="ref2va", start_workers=False,
        ref2va_num_inference_steps=9 if turbo else None,
        ref2va_flow_shift=6.0 if turbo else None,
        ref2va_audio_flow_shift=3.0 if turbo else None,
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    app = create_app(settings, store, storage)
    payload = {
        "model": "MiniMax-H3",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "role": "reference_image", "image_url": {"url": image_data_url()}},
            *[
                {"type": "image_url", "role": role, "image_url": {"url": image_data_url()}}
                for role in frame_roles
            ],
        ],
        "resolution": "768P", "duration": 5, "ratio": "adaptive",
    }
    with TestClient(app) as client:
        response = client.post("/v2/video_generation", json=payload)
    assert response.status_code == 200
    task = store.tasks[response.json()["task_id"]]
    assert task["task_mode"] == "ref2va"
    assert task["required_capability"] == "ref2va"
    assert [asset["asset_type"] for asset in task["assets"]] == ["input", "thumbnail"] * (1 + len(frame_roles))
    inputs = [asset for asset in task["assets"] if asset["asset_type"] == "input"]
    assert [asset["role"] for asset in inputs] == ["reference_image", *frame_roles]
    assert [asset["source_index"] for asset in inputs] == list(range(1, 2 + len(frame_roles)))
    assert all(storage.resolve(asset["relative_path"]).is_file() for asset in inputs)
    backend_payload = SGLangClient(settings, storage).build_payload(task)
    assert backend_payload["task"] == "ref2va"
    if turbo:
        assert backend_payload["num_inference_steps"] == 9
        assert backend_payload["flow_shift"] == 6.0
        assert backend_payload["audio_flow_shift"] == 3.0
        assert backend_payload["quality"] == "lossless"
    else:
        assert not {"num_inference_steps", "flow_shift", "audio_flow_shift", "quality"} & backend_payload.keys()
    assert len(backend_payload["conditions"]) == 1 + len(frame_roles)
    assert [item["frame_index"] for item in backend_payload["conditions"] if item["role"] == "keyframe"] == [
        index for role, index in [("first_frame", 0), ("last_frame", -1)] if role in frame_roles
    ]
    assert backend_payload["conditions"][-1]["role"] == "reference"


def test_readyz_reports_503_when_database_is_down(tmp_path):
    """readiness 要反映数据库健康 —— 库挂了还报就绪,k8s 会继续往这个实例转发。
    但 liveness(/healthz)必须保持 200,否则库一挂所有副本被反复重启。"""
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    app = create_app(settings, store, storage)

    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200

        store.db_down = True
        resp = client.get("/readyz")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
        assert "OperationalError" in resp.json()["database"]
        # liveness 不受影响
        assert client.get("/healthz").status_code == 200

        store.db_down = False
        assert client.get("/readyz").status_code == 200


def test_readyz_database_check_can_be_disabled(tmp_path):
    """readiness 查库是可关的:数据库抖动就让所有副本 NotReady,
    某些环境宁可让请求报 500 也不想端点全空。"""
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
        readiness_checks_database=False,
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    app = create_app(settings, store, storage)

    with TestClient(app) as client:
        store.db_down = True
        assert client.get("/readyz").status_code == 200      # 关掉后不受数据库影响
        assert client.get("/healthz").status_code == 200


def _metrics_app(tmp_path):
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    return create_app(settings, store, storage)


def test_metrics_exposes_queue_and_running(tmp_path, monkeypatch):
    """/metrics 要给出排队/运行的即时数,且后端不可达时不能把整个接口带崩。"""
    from h3_service import server as server_module

    app = _metrics_app(tmp_path)
    monkeypatch.setattr(
        server_module.SGLangClient, "list_jobs",
        lambda self: (_ for _ in ()).throw(RuntimeError("backend down")),
    )
    with TestClient(app) as client:
        created = client.post("/v2/video_generation", json={
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": "A fox in snow"}],
            "resolution": "768P", "duration": 5, "ratio": "16:9",
        })
        assert created.status_code == 200
        body = client.get("/metrics/queue").text

    assert "# TYPE h3_queue_tasks gauge" in body
    assert 'h3_queue_tasks{capability="fl2va",status="queued"} 1' in body
    assert "h3_queue_oldest_queued_age_seconds 12.5" in body
    assert "h3_queue_waiting_takeover 0" in body
    assert "h3_queue_backend_unavailable 0" in body
    # 后端挂了 -> up=0,而不是 500
    assert 'h3_queue_backend_up{capability="fl2va"} 0' in body
    assert "h3_queue_build_info" in body
    # 标签不能叫 instance —— 会跟 Prometheus 自带的撞,被改名成 exported_instance
    assert "h3_instance=" in body and "{instance=" not in body


def test_metrics_reports_backend_queue(tmp_path, monkeypatch):
    """后端可达时,把 SGLang 的作业状态数出来 —— queued 就是后端排队深度。"""
    from h3_service import server as server_module

    app = _metrics_app(tmp_path)
    monkeypatch.setattr(
        server_module.SGLangClient, "list_jobs",
        lambda self: [{"status": "running"}, {"status": "queued"}, {"status": "queued"}],
    )
    with TestClient(app) as client:
        body = client.get("/metrics/queue").text

    assert [line for line in body.splitlines() if 'state="queued"' in line][0].endswith(" 2")
    assert [line for line in body.splitlines() if 'state="running"' in line][0].endswith(" 1")
    assert 'h3_queue_backend_up{capability="fl2va"} 1' in body


def test_legacy_metrics_endpoint_is_unchanged(tmp_path):
    """/metrics 是既有看板/告警在抓的,格式和 series 不能动 —— 新指标另开 /metrics/queue。"""
    app = _metrics_app(tmp_path)
    with TestClient(app) as client:
        body = client.get("/metrics").text

    assert body.startswith("# TYPE h3_tasks gauge")
    for status in ("queued", "running", "succeeded", "failed", "cancelled"):
        assert f'h3_tasks{{status="{status}"}} ' in body
    # 旧端点里不能出现新指标,也不能给 h3_tasks 加标签
    assert "h3_queue_" not in body
    assert "capability=" not in body


def test_public_base_url_overrides_request_headers(tmp_path):
    """配了 H3_PUBLIC_BASE_URL 就一律用它拼下载 URL。

    经外层 API 网关时,请求头里的 Host 是我们的内部地址,按请求头推导出来的
    URL 客户端够不到;挂在网关子路径下时,推导逻辑只出 scheme://host 还会丢掉前缀。
    """
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
        public_base_url="https://gw.example.com/h3",
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    app = create_app(settings, store, storage)

    payload = {"model": "MiniMax-H3", "resolution": "768P", "duration": 5, "ratio": "16:9",
               "content": [{"type": "text", "text": "A fox in snow"}]}
    with TestClient(app) as client:
        task_id = client.post("/v2/video_generation", json=payload).json()["task_id"]
        store.tasks[task_id]["status"] = "succeeded"
        store.tasks[task_id]["output_relative_path"] = "out.mp4"

        # 故意给一个会把推导带偏的转发头
        got = client.get(f"/v2/query/video_generation/{task_id}",
                         headers={"X-Forwarded-Host": "internal.svc:8000",
                                  "X-Forwarded-Proto": "http"}).json()["task"]
        assert got["content"]["url"] == f"https://gw.example.com/h3/v2/video_generation/{task_id}/content"

        listed = client.get("/v2/query/video_generation").json()["items"][0]
        assert listed["content"]["url"].startswith("https://gw.example.com/h3/")


def test_content_url_falls_back_to_request_when_public_base_url_unset(tmp_path):
    """没配 H3_PUBLIC_BASE_URL 时行为不变:仍按 X-Forwarded-* / Host 推导(回归保护)。"""
    settings = Settings(
        database_url="postgresql://unused", storage_root=tmp_path,
        sglang_url="http://sglang", start_workers=False,
    )
    store = MemoryStore(settings)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    app = create_app(settings, store, storage)

    payload = {"model": "MiniMax-H3", "resolution": "768P", "duration": 5, "ratio": "16:9",
               "content": [{"type": "text", "text": "A fox in snow"}]}
    with TestClient(app) as client:
        task_id = client.post("/v2/video_generation", json=payload).json()["task_id"]
        store.tasks[task_id]["status"] = "succeeded"
        store.tasks[task_id]["output_relative_path"] = "out.mp4"
        got = client.get(f"/v2/query/video_generation/{task_id}",
                         headers={"X-Forwarded-Host": "h3.example.com",
                                  "X-Forwarded-Proto": "https"}).json()["task"]
        assert got["content"]["url"] == f"https://h3.example.com/v2/video_generation/{task_id}/content"
