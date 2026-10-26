from __future__ import annotations

import math
import os
import socket
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse


# Deployment guards use this marker to reject images with implicit Turbo defaults.
REF2VA_TURBO_CONFIG_VERSION = 1


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    storage_root: Path
    sglang_url: str
    sglang_media_root: str = "/data/minimax-h3"
    capability: str = "fl2va"
    instance_id: str = socket.gethostname()
    host: str = "0.0.0.0"
    port: int = 8000
    lease_seconds: int = 300
    poll_seconds: float = 2.0
    queue_timeout_seconds: int = 86400
    max_attempts: int = 5
    query_retention_days: int = 7
    asset_retention_days: int = 30
    allow_private_urls: bool = False
    media_timeout_seconds: float = 30.0
    backend_timeout_seconds: float = 30.0
    # Worker 领任务前探测 SGLang /health 的超时。它只用于启动门禁,
    # 应明显短于真正提交/查询任务时的 backend_timeout_seconds。
    backend_ready_timeout_seconds: float = 2.0
    # 抓 /metrics/queue 时问后端队列用的超时,**必须明显短于 Prometheus 的 scrapeTimeout**
    # (chart 默认 10s)。沿用 backend_timeout(30s)的话:SGLang 连上了但不响应时,
    # Prometheus 会在我们的异常处理跑到之前就放弃整个请求 —— 于是不但拿不到
    # "后端不可用"这个信号,连同一个端点上的数据库队列指标也一起丢了。
    metrics_backend_timeout_seconds: float = 3.0
    # 收到 SIGTERM 后,最多花多久等手上那条任务跑完(超时就放手,交给租约接管)。
    # 必须小于 k8s 的 terminationGracePeriodSeconds,否则等不到自己收尾就被 SIGKILL。
    shutdown_drain_seconds: float = 600.0
    callback_attempts: int = 3
    start_workers: bool = True
    # readiness 是否检查数据库。开着(默认)的收益不在"数据库整个挂掉"
    # (那时所有副本都不可用,500 还是 503 没差别),而在两个场景:
    #   ① 只有部分副本连不上库(NetworkPolicy/DNS/Secret 挂载出错)时,
    #      流量自动避让到健康副本;
    #   ② H3_DATABASE_URL 配错时,新副本卡在 NotReady、滚动更新停住,
    #      老副本还活着 —— 而不是一路"就绪"把所有副本换成坏的。
    # 代价:数据库抖动(如主从切换)会让所有副本短暂 NotReady。
    readiness_checks_database: bool = True
    # 对外基址,例如 https://h3.example.com 或 https://gw.example.com/h3。
    # 设了就一律用它拼下载 URL,不再从请求头推导 —— 经外层 API 网关时,
    # 请求头里的 Host 是我们的内部地址,推导出来的 URL 客户端够不到;
    # 而且 H3 挂在网关子路径下时,推导逻辑只出 scheme://host,会丢掉前缀。
    public_base_url: str = ""
    # Unset means base Ref2VA: leave sampling and quality to the backend.
    # Keep raw environment values until Ref2VA validation; FL2VA ignores them.
    ref2va_num_inference_steps: int | float | str | None = None
    ref2va_flow_shift: float | str | None = None
    ref2va_audio_flow_shift: float | str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("H3_DATABASE_URL", "").strip(),
            storage_root=Path(os.getenv("H3_STORAGE_ROOT", "/mnt/h3")).resolve(),
            sglang_url=os.getenv("H3_SGLANG_URL", "http://127.0.0.1:30010").rstrip("/"),
            sglang_media_root=os.getenv("H3_SGLANG_MEDIA_ROOT", "/data/minimax-h3").rstrip("/"),
            capability=os.getenv("H3_SERVICE_CAPABILITY", "fl2va").lower(),
            instance_id=os.getenv("H3_INSTANCE_ID") or socket.gethostname(),
            host=os.getenv("H3_HOST", "0.0.0.0"),
            port=int(os.getenv("H3_PORT", "8000")),
            lease_seconds=int(os.getenv("H3_TASK_LEASE_SECONDS", "300")),
            poll_seconds=float(os.getenv("H3_BACKEND_POLL_SECONDS", "2")),
            queue_timeout_seconds=int(os.getenv("H3_QUEUE_TIMEOUT_SECONDS", "86400")),
            max_attempts=int(os.getenv("H3_MAX_ATTEMPTS", "5")),
            query_retention_days=int(os.getenv("H3_QUERY_RETENTION_DAYS", "7")),
            asset_retention_days=int(os.getenv("H3_ASSET_RETENTION_DAYS", "30")),
            allow_private_urls=_bool_env("H3_ALLOW_PRIVATE_URLS"),
            media_timeout_seconds=float(os.getenv("H3_MEDIA_TIMEOUT_SECONDS", "30")),
            backend_timeout_seconds=float(os.getenv("H3_BACKEND_TIMEOUT_SECONDS", "30")),
            backend_ready_timeout_seconds=float(os.getenv("H3_BACKEND_READY_TIMEOUT_SECONDS", "2")),
            metrics_backend_timeout_seconds=float(os.getenv("H3_METRICS_BACKEND_TIMEOUT_SECONDS", "3")),
            shutdown_drain_seconds=float(os.getenv("H3_SHUTDOWN_DRAIN_SECONDS", "600")),
            callback_attempts=int(os.getenv("H3_CALLBACK_ATTEMPTS", "3")),
            start_workers=_bool_env("H3_START_WORKERS", True),
            readiness_checks_database=_bool_env("H3_READINESS_CHECKS_DATABASE", True),
            public_base_url=os.getenv("H3_PUBLIC_BASE_URL", "").strip().rstrip("/"),
            ref2va_num_inference_steps=os.getenv("H3_REF2VA_NUM_INFERENCE_STEPS"),
            ref2va_flow_shift=os.getenv("H3_REF2VA_FLOW_SHIFT"),
            ref2va_audio_flow_shift=os.getenv("H3_REF2VA_AUDIO_FLOW_SHIFT"),
        )

    @property
    def ref2va_sampling(self) -> dict[str, int | float]:
        if self.capability != "ref2va":
            return {}
        values = {
            "num_inference_steps": self.ref2va_num_inference_steps,
            "flow_shift": self.ref2va_flow_shift,
            "audio_flow_shift": self.ref2va_audio_flow_shift,
        }
        values = {key: value.strip() if isinstance(value, str) else value for key, value in values.items()}
        present = [value is not None and value != "" for value in values.values()]
        if not any(present):
            return {}
        if not all(present):
            raise RuntimeError(
                "H3_REF2VA_NUM_INFERENCE_STEPS, H3_REF2VA_FLOW_SHIFT and "
                "H3_REF2VA_AUDIO_FLOW_SHIFT must be set together or all left unset"
            )
        steps = values["num_inference_steps"]
        try:
            number = Decimal(str(steps))
            if isinstance(steps, bool) or not number.is_finite() or number < 2 or number != number.to_integral_value():
                raise ValueError
            normalized: dict[str, int | float] = {"num_inference_steps": int(number)}
        except (InvalidOperation, ValueError, TypeError, OverflowError):
            raise RuntimeError("H3_REF2VA_NUM_INFERENCE_STEPS must be a finite integer >= 2") from None
        for key in ("flow_shift", "audio_flow_shift"):
            try:
                value = float(values[key])
                if isinstance(values[key], bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                raise RuntimeError(f"H3_REF2VA_{key.upper()} must be finite and > 0") from None
            normalized[key] = value
        return normalized

    def validate(self) -> None:
        if not self.database_url:
            raise RuntimeError("H3_DATABASE_URL is required")
        parsed = urlparse(self.sglang_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("H3_SGLANG_URL must be an absolute http(s) URL")
        if not self.sglang_media_root.startswith("/"):
            raise RuntimeError("H3_SGLANG_MEDIA_ROOT must be an absolute container path")
        if self.public_base_url and not self.public_base_url.startswith(("http://", "https://")):
            raise RuntimeError("H3_PUBLIC_BASE_URL must start with http:// or https://")
        if self.capability not in {"fl2va", "ref2va"}:
            raise RuntimeError("H3_SERVICE_CAPABILITY must be fl2va or ref2va")
        if self.capability == "ref2va":
            self.ref2va_sampling
        if min(self.lease_seconds, self.queue_timeout_seconds, self.max_attempts) <= 0:
            raise RuntimeError("lease, task timeout, and max attempts must be positive")
        if min(self.poll_seconds, self.backend_ready_timeout_seconds) <= 0:
            raise RuntimeError("backend poll interval and readiness timeout must be positive")
        if min(
            self.query_retention_days,
            self.asset_retention_days,
        ) <= 0:
            raise RuntimeError("retention days must be positive")
