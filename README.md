# minimax-h3-server

A self-hosted, MiniMax H3-compatible video generation API for the open-weight
MiniMax H3 models, served through [SGLang](https://github.com/sgl-project/sglang).

This is an independent project. It is not affiliated with or endorsed by MiniMax;
"MiniMax" and "MiniMax H3" are used only to describe API and model compatibility.

This repository provides two independently deployable processes:

- `h3-service`: a complete MiniMax H3-compatible API, PostgreSQL task queue,
  SGLang worker, callback delivery, mounted-disk asset storage and dashboard.
- `h3-router`: an optional stateless request router. It has no database, mount,
  cache or file-writing code. A single `h3-service` works without it.

The current release implements text-to-video and first/last-frame image-to-video
through an FL2VA SGLang deployment, plus reference image/video/audio generation
through a Ref2VA SGLang deployment.

## Architecture

```text
client ───────────────────────────────► h3-service (FL2VA)
  │                                      ├─ PostgreSQL queue/leases
  │                                      ├─ mounted disk
  │                                      └─ SGLang FL2VA
  │
  └─ optional h3-router ─┬─ T2V/I2V ──► h3-service (FL2VA)
                        └─ Ref ───────► h3-service (Ref2VA) ─► SGLang Ref2VA
```

The router sends T2V/I2V creation traffic to `H3_FL2VA_SERVICE_URLS` and Ref
creation traffic to `H3_REF2VA_SERVICE_URLS`. Query, delete, download and
dashboard traffic use the combined FL2VA and Ref2VA pool because all H3 service
instances share the same PostgreSQL database and mounted asset namespace.
Router round-robin state exists only in memory and is safe to lose. When FL2VA
and Ref2VA services run on different machines, they must share PostgreSQL and
the mounted asset namespace so any H3 service can answer task operations.

## Images

| Component | Image |
| --- | --- |
| H3 Server | `4pdosc/minimax-h3-server` |
| H3 Router | `4pdosc/minimax-h3-router` |
| SGLang (FL2VA) | `lmsysorg/sglang:v0.5.18-cu129` |
| SGLang (Ref2VA) | `lmsysorg/sglang:v0.5.19` |

Ref2VA requests that mix reference media with first/last frames need SGLang
v0.5.19 or newer.

## Kubernetes

The Helm chart lives in [modelsphere/helm-charts](https://github.com/modelsphere/helm-charts)
as `minimax-h3`:

```bash
helm repo add modelsphere https://modelsphere.github.io/helm-charts
helm install h3 modelsphere/minimax-h3 -n minimax-h3 --create-namespace
```

See the chart README for values, storage and upgrade notes.

## Run

Python 3.11+ is required. For local development:

```bash
./postgresql_demo/build.sh
./postgresql_demo/docker.sh
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
export H3_DATABASE_URL=postgresql://h3:h3@127.0.0.1:5432/minimaxh3
export H3_SGLANG_URL=http://127.0.0.1:30010
.venv/bin/h3-service
```

`postgresql_demo/` is an optional single-node PostgreSQL 16 deployment. It is
intended only for a simple manual installation and does not include HA features.

Open `http://127.0.0.1:8000/dashboard` for the dashboard.

Build the images on the build machine:

```bash
./h3_service/build.sh
./h3_router/build.sh
./postgresql_demo/build.sh
```

`build.sh` tags local images (`minimax-h3-server:dev`, `minimax-h3-router:dev`);
set `H3_SERVER_IMAGE` / `H3_ROUTER_IMAGE` to tag them for your registry. Builds use
PyPI by default; pass `--build-arg PIP_INDEX_URL=<mirror>` to use a mirror.

Run the services on their respective deployment machines. The launchers default
to the published images and local addresses; override them with environment
variables (`H3_DATABASE_URL`, `H3_SGLANG_URL`, `H3_DATA_DIR`,
`H3_FL2VA_MODEL_DIR`, `H3_REF2VA_MODEL_DIR`, `H3_FL2VA_SERVICE_URLS`,
`H3_REF2VA_SERVICE_URLS`, `H3_SERVER_IMAGE`, `H3_ROUTER_IMAGE`, `H3_SGLANG_IMAGE`):

```bash
./postgresql_demo/docker.sh
./h3_service/sglang.sh
./h3_service/docker.sh
./h3_router/docker.sh
```

The API and SGLang containers must mount the same host directory at the paths
specified by `H3_STORAGE_ROOT` and `H3_SGLANG_MEDIA_ROOT`. `docker.sh` starts the
API container with this layout. `h3_service/sglang.sh` starts FL2VA with 4 GPUs,
TP 2 and Ulysses 2; `h3_service/sglang_ref2va.sh` starts Ref2VA with 4 GPUs,
TP 4 and Ulysses 1. The same H3 Server image and `docker.sh` are used for both
capabilities: set `H3_SERVICE_CAPABILITY=ref2va` and point `H3_SGLANG_URL` to the
Ref2VA SGLang address on the Ref machine. Download URLs are built from the
current request host and forwarded headers, so no public URL setting is needed.
No FTP service or upload is used.

Ref2VA now defaults to the LightX2V 8-step 768p Turbo LoRA. Download the exact
file on every GPU node that can serve Ref2VA before starting the offline container:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors \
  --local-dir /srv/models/Minimax-h3-Turbo
./h3_service/sglang_ref2va.sh
H3_SERVICE_CAPABILITY=ref2va H3_SGLANG_URL=http://127.0.0.1:30010 \
  ./h3_service/docker.sh
```

Both launchers source `h3_service/ref2va_profile.sh`. The shared switch
`H3_REF2VA_TURBO_ENABLED` defaults to `true` and controls the LoRA arguments,
mount and API sampling settings together. Use the same environment when starting
both containers. The SGLang script mounts that directory read-only at `/lora`, uses scale `1.0`,
alpha `8`, and `--lora-merge-mode auto`, and checks that the file exists before
replacing the container. Override the host directory with `H3_REF2VA_LORA_DIR`.
The original Ref2VA base checkpoint remains mounted at `/model`.
Request-based warmup uses `--warmup-steps 2`: H3 requires both endpoints of the
sigma interval, so SGLang v0.5.19's default one-point warmup is invalid.

The H3 Server sends `num_inference_steps=9`, `flow_shift=6.0`,
`audio_flow_shift=3.0`, and `quality="lossless"` for Ref2VA. In SGLang v0.5.19,
nine sigma points mean eight denoiser evaluations. `lossless` disables additional
sampling approximations; the output still reflects the distilled LoRA.
The three settings are configurable through `H3_REF2VA_NUM_INFERENCE_STEPS`,
`H3_REF2VA_FLOW_SHIFT`, and `H3_REF2VA_AUDIO_FLOW_SHIFT`; FL2VA keeps its existing
50-step settings. When starting the Server directly, leave all three unset (or
blank) for base mode, or set all three together. Steps accept integer-valued
numeric strings such as `9.0`; invalid Ref2VA values do not affect FL2VA startup.
Keep the LoRA file and sampling settings consistent across all
Ref2VA replicas and recheck the step convention when upgrading SGLang.

To return to base Ref2VA, set the shared switch to `false` for both launchers:

```bash
export H3_REF2VA_TURBO_ENABLED=false
./h3_service/sglang_ref2va.sh
H3_SERVICE_CAPABILITY=ref2va H3_SGLANG_URL=http://127.0.0.1:30010 \
  ./h3_service/docker.sh
```

This removes the LoRA arguments and mount and omits `num_inference_steps`,
`flow_shift`, `audio_flow_shift` and `quality` from Ref2VA requests, restoring
the original backend defaults. Warmup remains at two steps in both modes.
Pause new Ref2VA submissions and drain queued and running tasks before changing
all replicas together. Request bodies and the public model name `MiniMax-H3`
remain the same.

`H3_SERVER_IMAGE` overrides the image for both the build and API launch scripts.
The API launcher checks the image's Ref2VA config protocol before removing the
current container, in both Turbo and base mode; Server images built before this
protocol existed are rejected. For Kubernetes, see the chart README in
[modelsphere/helm-charts](https://github.com/modelsphere/helm-charts).

The router uses a separate minimal image without PostgreSQL/Pillow dependencies.
Its container is read-only and receives no database or storage settings. For
multiple backends, set comma-separated URLs in `H3_FL2VA_SERVICE_URLS` and
`H3_REF2VA_SERVICE_URLS` when running `h3_router/docker.sh`.

Environment variables and the full API are described in [api.md](api.md)
(in Chinese).

## API examples

T2V:

```bash
curl -X POST http://127.0.0.1:8000/v2/video_generation \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-H3",
    "content": [{"type": "text", "text": "A fox walks through snow"}],
    "resolution": "768P",
    "duration": 5,
    "ratio": "16:9"
  }'
```

I2V accepts `image_url` content with `role: first_frame`, `last_frame`, or both.
When the role is omitted it defaults to `first_frame`. Images may be HTTP(S) or
data URLs. Only 768P and durations from 4 to 15 seconds are accepted.

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "The camera slowly pushes forward"},
    {"type": "image_url", "role": "first_frame", "image_url": {"url": "https://example.com/first.jpg"}},
    {"type": "image_url", "role": "last_frame", "image_url": {"url": "https://example.com/last.jpg"}}
  ],
  "resolution": "768P",
  "duration": 5,
  "ratio": "adaptive"
}
```

Ref2VA accepts reference images, videos and audio using the roles
`reference_image`, `reference_video` and `reference_audio`. A reference audio
must be accompanied by at least one reference image or video. The Router sends
these requests to the Ref2VA H3 Server pool.

Ref2VA also accepts `first_frame` and/or `last_frame` images alongside reference
media (one image per frame role). These requests remain in the Ref2VA pool and
keep the requested ratio. This combination requires SGLang v0.5.19 or newer.
Reference-media limits exclude the optional first/last frames;
keyframes do not replace the visual reference required by `reference_audio`.

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "Keep the subject from <Picture 1> and transition between the supplied frames"},
    {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://example.com/reference.jpg"}},
    {"type": "image_url", "role": "first_frame", "image_url": {"url": "https://example.com/first.jpg"}},
    {"type": "image_url", "role": "last_frame", "image_url": {"url": "https://example.com/last.jpg"}}
  ],
  "resolution": "768P",
  "duration": 5,
  "ratio": "16:9"
}
```

Task operations:

```text
GET    /v2/query/video_generation/{task_id}
GET    /v2/query/video_generation
DELETE /v2/video_generation/{task_id}
GET    /v2/video_generation/{task_id}/content
```

The content endpoint supports HTTP byte ranges. This version deliberately has no
Bearer authentication; place the service behind your own authenticated gateway
before exposing it publicly.

## Callback and URL security

If `callback_url` is supplied, task creation first sends `{"challenge":"..."}`.
The endpoint must return HTTP 200 with the same JSON field. State notifications
then contain `{"task": ...}` for queued/running/terminal transitions and are
retried up to `H3_CALLBACK_ATTEMPTS` times.

By default, image and callback URLs that resolve to loopback, private, link-local,
multicast or reserved addresses are rejected, credentials in URLs are rejected,
redirects are revalidated, downloads are bounded, and callback redirects are not
followed. This reduces SSRF risk. Set `H3_ALLOW_PRIVATE_URLS=true` only in a
trusted network where private image/callback endpoints are required.

## Persistence and retention

The task store claims work with PostgreSQL `FOR UPDATE SKIP LOCKED` and renewable
leases. In addition, it:

- keeps assets as `h3_task_assets.relative_path` on a shared mount;
- splits task, one-to-many assets and callback deliveries into separate tables;
- stores official request content, mode/capability and SGLang backend job identity;
- uses lowercase official task states and soft deletion;
- adds callback leases/idempotency and advisory-locked cleanup;
- separates the 7-day query window from 30-day assets/dashboard metadata.

`h3_video_tasks`, `h3_task_assets` and `h3_callback_deliveries` are created
idempotently at startup. Official query APIs stop returning
a task after 7 days. Inputs, thumbnails and outputs are removed from the mounted
disk after 30 days; terminal metadata is then purged after the dashboard window.
The query and asset windows are configurable; dashboard metadata retention is
fixed at 30 days.

## Health and operations

```text
GET /healthz       liveness
GET /readyz        database/storage initialization completed
GET /metrics       Prometheus gauges by task status (24h window, legacy)
GET /metrics/queue Prometheus live queue/backend gauges (see Metrics)
GET /dashboard     operator dashboard
```

Workers persist the SGLang job id before polling. PostgreSQL leases and heartbeats
allow another service instance to resume polling after a crash without submitting
a second job. A submission timeout is treated as ambiguous and fails the task to
avoid accidental duplicate GPU work.

On the target server, a useful first-run check is:

```bash
curl -f http://127.0.0.1:8000/readyz
curl -f http://127.0.0.1:8000/healthz
psql "$H3_DATABASE_URL" -c '\dt h3_*'
```

## Metrics

Two Prometheus text endpoints, split on purpose:

- **`GET /metrics`** — legacy, kept byte-stable for existing dashboards/alerts.
  A single gauge `h3_tasks{status=...}` counting tasks **created in the last 24h
  and currently in that state** (`status`: `queued`, `running`, `succeeded`,
  `failed`, `cancelled`). Do not rename or relabel it.
- **`GET /metrics/queue`** — the live queue/backend view. This is what the Helm chart's
  `ServiceMonitor` scrapes (`serviceMonitor.path`, default `/metrics/queue`,
  `interval` 30s / `scrapeTimeout` 10s). New `h3_queue_*` names, safe to extend.

### `/metrics/queue` series

All gauges. Values are **cluster-wide** (every Server replica shares one database
and reports identical numbers) **except** the per-replica ones flagged below.
Aggregate cluster-wide series with `max()`/`avg()`, **never `sum()`** (summing
multiplies by the replica count).

| Metric | Labels | Meaning |
| --- | --- | --- |
| `h3_queue_tasks` | `status`, `capability` | Tasks currently in the DB per status per capability pool. |
| `h3_queue_oldest_queued_age_seconds` | — | Age of the oldest queued task. **Not emitted** when nothing is queued. |
| `h3_queue_oldest_running_age_seconds` | — | Age of the oldest running task; keeps climbing for a stuck task. |
| `h3_queue_waiting_takeover` | — | Running tasks whose lease is empty/expired, waiting for another replica to take over. Sustained `> 0` = every replica for that capability is down or stuck. |
| `h3_queue_backend_unavailable` | — | Running tasks currently disconnected from SGLang but still retrying within lease. Sustained `> 0` = SGLang problem. |
| `h3_queue_running_on_instance` | `h3_instance` | **Per-replica.** Tasks this replica is executing. Label is `h3_instance` (not `instance`) to avoid clashing with Prometheus's own target label. |
| `h3_queue_completed_24h` | `status` | Tasks completed in the last 24h by terminal status. |
| `h3_queue_backend_up` | `capability` | **Per-replica.** Whether this replica's sidecar SGLang is reachable (`1`/`0`). |
| `h3_queue_backend_jobs` | `capability`, `state` | **Per-replica.** Jobs on this pod's SGLang by the backend's own state; `state="queued"` is the backend queue depth. |
| `h3_queue_build_info` | `capability`, `h3_instance` | **Per-replica.** Build/deploy info; value is always `1`. |

The backend series (`h3_queue_backend_up` / `h3_queue_backend_jobs`) are derived by
polling SGLang's `GET /v1/videos` (SGLang exposes no `/metrics`), cached briefly
per pod. The poll uses `server.config.metricsBackendTimeoutSeconds` (default 3s),
which must stay well under `serviceMonitor.scrapeTimeout` — otherwise a hung
backend makes Prometheus give up and the database-derived series are lost too.

Example scrape:

```bash
curl -s http://127.0.0.1:8000/metrics/queue
# h3_queue_tasks{capability="fl2va",status="succeeded"} 12
# h3_queue_backend_up{capability="fl2va"} 1
# h3_queue_backend_jobs{capability="fl2va",state="queued"} 0
# ...
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
