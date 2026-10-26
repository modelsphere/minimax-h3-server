from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from .config import Settings
from .storage import SharedStorage


SGLANG_MODEL = "/model"


class BackendError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        ambiguous: bool = False,
        missing: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.ambiguous = ambiguous
        self.missing = missing


class SGLangClient:
    def __init__(self, settings: Settings, storage: SharedStorage, base_url: str | None = None,
                 timeout: float | None = None):
        self.settings = settings
        self.storage = storage
        self.base_url = (base_url or settings.sglang_url).rstrip("/")
        # timeout 可覆盖:/metrics 抓取要用比 Prometheus scrapeTimeout 短得多的值,
        # 否则后端连上却不响应时,Prometheus 先放弃,整个端点的指标都拿不到。
        self.timeout = settings.backend_timeout_seconds if timeout is None else timeout

    def is_ready(self) -> bool:
        """SGLang 是否已完成启动、可以接收生成请求。

        这个检查只用于 Worker 领任务前的启动门禁。不可达、超时和非 200 都表示
        "继续等待",不是需要计入任务重试次数的异常。
        """
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def build_payload(self, task: dict[str, Any]) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = []
        if task["task_mode"] in {"fl2va", "ref2va"}:
            for asset in task.get("assets", []):
                if asset["asset_type"] != "input" or asset["role"] not in {"first_frame", "last_frame"}:
                    continue
                conditions.append({
                    "type": "image",
                    "uri": self.storage.sglang_uri(asset["relative_path"]),
                    "role": "keyframe",
                    "frame_index": 0 if asset["role"] == "first_frame" else -1,
                })
            conditions.sort(key=lambda item: item["frame_index"], reverse=True)
        if task["task_mode"] == "ref2va":
            ref_types = {
                "reference_image": "image",
                "reference_video": "video",
                "reference_audio": "audio",
            }
            assets = sorted(
                (
                    asset for asset in task.get("assets", [])
                    if asset["asset_type"] == "input" and asset["role"] in ref_types
                ),
                key=lambda asset: (asset.get("source_index", 0), asset.get("asset_id", 0)),
            )
            for asset in assets:
                conditions.append({
                    "type": ref_types[asset["role"]],
                    "uri": self.storage.sglang_uri(asset["relative_path"]),
                    "role": "reference",
                })

        target = {
            "short_edge": 768,
            "aspect_ratio": "auto" if task["ratio"] == "adaptive" else task["ratio"],
            "duration_seconds": float(task["duration"]),
        }
        payload = {
            "model": SGLANG_MODEL,
            "prompt": task["prompt"],
            "task": task["task_mode"],
            "conditions": conditions,
            "target": target,
            "seed": task["seed"],
        }
        if task["task_mode"] == "ref2va":
            sampling = self.settings.ref2va_sampling
            if sampling:
                payload.update(sampling)
                payload["quality"] = "lossless"
            return payload
        payload.update({
            "seconds": task["duration"],
            "num_outputs_per_prompt": 1,
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "quality": "lossless",
        })
        return payload

    def submit(self, task: dict[str, Any]) -> dict[str, Any]:
        try:
            response = httpx.post(f"{self.base_url}/v1/videos", json=self.build_payload(task), timeout=self.timeout)
        except httpx.ConnectError as exc:
            raise BackendError(f"cannot connect to SGLang: {exc}", retryable=True) from exc
        except httpx.TimeoutException as exc:
            raise BackendError(f"SGLang submission timed out: {exc}", ambiguous=True) from exc
        except httpx.TransportError as exc:
            raise BackendError(f"SGLang submission connection failed: {exc}", ambiguous=True) from exc
        if response.status_code >= 500:
            raise BackendError(f"SGLang returned HTTP {response.status_code}: {response.text[:1000]}", retryable=True)
        if response.status_code >= 400:
            raise BackendError(f"SGLang rejected the task with HTTP {response.status_code}: {response.text[:1000]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise BackendError("SGLang returned invalid JSON") from exc
        job_id = data.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BackendError("SGLang response did not contain a video id")
        return data

    def list_jobs(self) -> list[dict[str, Any]]:
        """后端当前的作业列表(GET /v1/videos)。

        SGLang 这个 build 没有 /metrics,队列情况只能从作业列表里数
        (实测:并发为 1,后到的作业一直是 status=queued 直到前一个跑完)。
        """
        response = httpx.get(f"{self.base_url}/v1/videos", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else payload
        return [job for job in (data or []) if isinstance(job, dict)]

    def status(self, job_id: str) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/v1/videos/{job_id}", timeout=self.timeout)
        except httpx.TransportError as exc:
            raise BackendError(f"cannot query SGLang task: {exc}", retryable=True) from exc
        if response.status_code >= 500:
            raise BackendError(f"SGLang query returned HTTP {response.status_code}", retryable=True)
        if response.status_code == 404:
            raise BackendError("SGLang task no longer exists", missing=True)
        if response.status_code >= 400:
            raise BackendError(f"SGLang query returned HTTP {response.status_code}: {response.text[:1000]}")
        try:
            return response.json()
        except ValueError as exc:
            raise BackendError("SGLang status response was invalid JSON", retryable=True) from exc

    def content_chunks(self, job_id: str) -> Iterator[bytes]:
        try:
            with httpx.stream("GET", f"{self.base_url}/v1/videos/{job_id}/content", timeout=self.timeout) as response:
                if response.status_code != 200:
                    raise BackendError(f"SGLang content returned HTTP {response.status_code}", retryable=response.status_code >= 500)
                yield from response.iter_bytes()
        except httpx.TransportError as exc:
            raise BackendError(f"cannot download SGLang output: {exc}", retryable=True) from exc

    def cancel(self, job_id: str) -> None:
        try:
            httpx.delete(f"{self.base_url}/v1/videos/{job_id}", timeout=self.timeout)
        except httpx.HTTPError:
            pass
