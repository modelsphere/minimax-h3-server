# MiniMax H3 API 部署与接口说明

本文档说明 H3 Server、H3 Router 的部署关系、环境变量和对外接口。

## 1. 服务关系

```text
客户端
  │
  ▼
H3 Router :8080（无状态，只转发请求）
  ├── T2V / I2V ──► H3 Server A :8000 ──► SGLang A :30010
  ├── T2V / I2V ──► H3 Server B :8000 ──► SGLang B :30010
  └── Ref2VA ─────► H3 Ref Server :8000 ──► Ref2VA SGLang :30110

H3 Server A/B/Ref ──► 同一个 PostgreSQL
H3 Server A/B/Ref ──► 同一个共享挂载盘
SGLang A/B/Ref    ──► 同一个共享挂载盘
```

- Router 不连接数据库，也不保存文件。
- H3 Server 负责 API、任务调度、Callback 和数据清理，不会启动 SGLang。
- SGLang 需要独立部署，每个 H3 Server 通过 `H3_SGLANG_URL` 连接对应的 SGLang。
- 多个 H3 Server 必须共享 PostgreSQL 和资产目录。
- 跨节点部署时，资产目录应使用支持 `ReadWriteMany` 的存储，例如 NFS 或 CephFS。

## 2. 镜像与端口

| 服务 | 镜像 | 默认端口 | 启动命令 |
|---|---|---:|---|
| H3 Server | `4pdosc/minimax-h3-server` | 8000 | `python -m h3_service.server` |
| H3 Router | `4pdosc/minimax-h3-router` | 8080 | `python -m h3_router.server` |
| SGLang | `lmsysorg/sglang:v0.5.19`(FL2VA / Ref2VA，首尾帧与参考素材混用需 v0.5.19+) | 30010 | 使用下方 GPU 参数启动 |

SGLang 启动参数至少包含：

```text
--num-gpus 4 --tp-size 2 --ulysses-degree 2
```

Ref2VA SGLang 使用：

```text
--num-gpus 4 --tp-size 4 --ulysses-degree 1 --warmup-mode request
```

## 3. H3 Server 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `H3_DATABASE_URL` | 无，必填 | PostgreSQL 连接串，例如 `postgresql://h3:h3@postgresql:5432/minimaxh3`。 |
| `H3_STORAGE_ROOT` | `/mnt/h3` | H3 Server 中的共享资产目录。建议部署时设置为 `/data/minimax-h3`。 |
| `H3_SGLANG_URL` | `http://127.0.0.1:30010` | 当前 H3 Server 对应的 SGLang 地址。 |
| `H3_SGLANG_MEDIA_ROOT` | `/data/minimax-h3` | SGLang 能看到的共享资产路径。 |
| `H3_SERVICE_CAPABILITY` | `fl2va` | 当前 H3 Server 连接的模型类型。T2V/I2V Server 使用 `fl2va`，Ref Server 使用 `ref2va`。 |
| `H3_REF2VA_NUM_INFERENCE_STEPS` | `9` | Ref2VA 专用采样步数，整数且至少为 2；v0.5.19 中对应 8 次去噪计算。 |
| `H3_REF2VA_FLOW_SHIFT` | `6.0` | Ref2VA video flow shift，必须为有限正数。 |
| `H3_REF2VA_AUDIO_FLOW_SHIFT` | `3.0` | Ref2VA audio flow shift，必须为有限正数。 |
| `H3_INSTANCE_ID` | 容器 hostname | Dashboard 中显示的请求接入/任务执行实例，K8s 建议设为 Pod 名。 |
| `H3_HOST` | `0.0.0.0` | API 监听地址。 |
| `H3_PORT` | `8000` | API 监听端口。 |
| `H3_TASK_LEASE_SECONDS` | `300` | Running 任务的端到端健康租约，单位秒。Server 存活且持续收到 SGLang 响应时才续约。 |
| `H3_BACKEND_POLL_SECONDS` | `2` | 查询 SGLang 任务状态的间隔，单位秒。 |
| `H3_QUEUE_TIMEOUT_SECONDS` | `86400` | 任务从创建到进入终态的最长时间，单位秒；排队超时返回 `QueueTimeout`，运行超时返回 `TaskTimeout`。 |
| `H3_MAX_ATTEMPTS` | `5` | 向 SGLang 提交生成请求的最大次数，包含首次提交。状态查询失败不计入。 |
| `H3_QUERY_RETENTION_DAYS` | `7` | API 可以查询任务的保留天数。 |
| `H3_ASSET_RETENTION_DAYS` | `30` | 输入素材、缩略图和生成视频的保留天数。 |
| `H3_ALLOW_PRIVATE_URLS` | `false` | 是否允许输入素材或 Callback 使用内网 URL。 |
| `H3_MEDIA_TIMEOUT_SECONDS` | `30` | 下载远程输入素材的超时秒数。 |
| `H3_BACKEND_TIMEOUT_SECONDS` | `30` | 调用 SGLang 的 HTTP 超时秒数。 |
| `H3_BACKEND_READY_TIMEOUT_SECONDS` | `2` | Worker 领任务前调用 SGLang `/health` 的超时秒数；未就绪时任务保持 queued。 |
| `H3_CALLBACK_ATTEMPTS` | `3` | Callback 最大尝试次数。 |
| `H3_START_WORKERS` | `true` | 是否启动任务、Callback 和清理 Worker。 |

每个 H3 Server 固定启动一个任务 Worker，Dashboard 元数据固定保留 30 天。下载 URL 根据请求的 `Host`、`X-Forwarded-Host` 和 `X-Forwarded-Proto` 动态生成。

Running 任务每 60 秒检查一次续约条件。SGLang 正常返回任务状态或持续传输视频数据时续约；H3 Server 或 SGLang 失联后停止续约，默认约 5 分钟后允许其他相同 `H3_SERVICE_CAPABILITY` 的 Worker 接管。接管方会先查询原 SGLang job，无法恢复时再提交到自己的 `H3_SGLANG_URL`。任务在故障转移期间仍对外返回 `running`。

首次提交和故障接管后的重新提交共用 `H3_MAX_ATTEMPTS` 限制。每次实际调用 SGLang 生成接口前都会原子占用一次提交次数，达到上限后任务变为 `failed`，错误码为 `TooManyAttempts`。`H3_QUEUE_TIMEOUT_SECONDS` 同时为排队、运行、重提和结果下载路径提供总生命周期兜底。

每个副本的 `H3_INSTANCE_ID` 应保持唯一，Kubernetes 中建议使用 Pod 名。相同 capability 的副本必须共享 PostgreSQL 和 `H3_STORAGE_ROOT`，否则不能安全接管任务。

续约和接管日志使用可检索的 `event` 字段，主要包括：

```text
task_lease_renewed
task_lease_renewal_suspended
task_backend_recovered
task_lease_expired
task_expired_lease_observed
task_takeover_started
task_takeover_resumed
task_takeover_resubmit_started
task_takeover_resubmit_succeeded
task_takeover_deferred
task_backend_submission_reserved
task_attempt_limit_exceeded
task_timeout_exceeded
```

正常续约日志为 DEBUG；续约停止、租约过期、提交次数耗尽和任务超时为 WARNING；接管开始、恢复原 job、提交次数预占和转投成功为 INFO。所有相关日志都包含 `task_id`、执行实例、租约时间及 backend job 信息，但不会打印 Prompt、媒体内容或数据库凭证。

K8s 中关键环境变量示例：

```yaml
env:
  - name: H3_DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: minimax-h3-db
        key: url
  - name: H3_STORAGE_ROOT
    value: /data/minimax-h3
  - name: H3_SGLANG_URL
    value: http://sglang-fl2va-a:30010
  - name: H3_SGLANG_MODEL
    value: /model
  - name: H3_SGLANG_MEDIA_ROOT
    value: /data/minimax-h3
  - name: H3_SERVICE_CAPABILITY
    value: fl2va
  - name: H3_INSTANCE_ID
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
```

Ref H3 Server 使用同一个镜像，关键配置改为：

```yaml
env:
  - name: H3_SGLANG_URL
    value: http://sglang-ref2va:30010
  - name: H3_SERVICE_CAPABILITY
    value: ref2va
```

## 4. Router 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `H3_FL2VA_SERVICE_URLS` | 空，必填 | T2V/I2V 后端地址，多个地址使用英文逗号分隔。 |
| `H3_REF2VA_SERVICE_URLS` | 空 | Ref2VA H3 Server 地址，多个地址使用英文逗号分隔；为空时 Ref 请求返回 HTTP 501。 |
| `H3_ROUTER_HOST` | `0.0.0.0` | Router 监听地址。 |
| `H3_ROUTER_PORT` | `8080` | Router 监听端口。 |
| `H3_ROUTER_CONNECT_TIMEOUT_SECONDS` | `10` | 连接单个 H3 Server 的超时秒数。 |

查询、删除、下载和 Dashboard 请求会在 `H3_FL2VA_SERVICE_URLS` 和 `H3_REF2VA_SERVICE_URLS` 的所有地址中轮询转发，重复地址会自动去重。

```yaml
env:
  - name: H3_FL2VA_SERVICE_URLS
    value: http://h3-server-a:8000,http://h3-server-b:8000
  - name: H3_REF2VA_SERVICE_URLS
    value: http://h3-ref-server-a:8000,http://h3-ref-server-b:8000
```

Router 在多个后端之间使用内存轮询。连接失败或连接超时时会尝试下一个地址；后端已经返回 HTTP 500/502/503 时会直接转发响应，不会自动切换。Router 当前没有主动健康检查或熔断。

## 5. PostgreSQL 与共享盘

当前的数据库参数：

```text
database: minimaxh3
user:     h3
password: h3
port:     5432
```

H3 Server 启动时会自动创建或升级业务表。数据库用户需要拥有 `minimaxh3` 数据库中创建表、修改表和读写数据的权限。实际部署时建议通过 Kubernetes Secret 注入连接串。

H3 Server 和对应的 SGLang 必须挂载同一个资产卷，并看到一致的路径：

```text
H3_STORAGE_ROOT=/data/minimax-h3
H3_SGLANG_MEDIA_ROOT=/data/minimax-h3
```

## 6. 对外接口

客户端统一访问 Router，例如：

```text
http://localhost:8080
```

### 6.1 创建 T2V 任务

```bash
curl -X POST 'http://localhost:8080/v2/video_generation' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-H3",
    "content": [
      {"type": "text", "text": "一只水獭在河边缓慢行走，镜头向前推进"}
    ],
    "resolution": "768P",
    "duration": 6,
    "ratio": "16:9"
  }'
```

### 6.2 创建 I2V 任务

图片支持 HTTP/HTTPS URL 或 `data:image/...;base64,...`：

```bash
curl -X POST 'http://localhost:8080/v2/video_generation' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-H3",
    "content": [
      {"type": "text", "text": "镜头缓慢向前推进，画面自然运动"},
      {
        "type": "image_url",
        "role": "first_frame",
        "image_url": {"url": "https://example.com/first-frame.png"}
      }
    ],
    "resolution": "768P",
    "duration": 6,
    "ratio": "adaptive"
  }'
```

创建成功：

```json
{"task_id":"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
```

主要参数限制：

- `model`：固定为 `MiniMax-H3`。
- `content`：至少包含一个非空的 `text`，合并后的文本最长 7000 字符。
- `resolution`：当前只支持 `768P`。
- `duration`：4～15 秒的整数。
- T2V 的 `ratio`：支持 `21:9`、`16:9`、`4:3`、`1:1`、`3:4`、`9:16`。
- I2V 的 `ratio`：使用 `adaptive`。
- I2V 图片角色：`first_frame` 或 `last_frame`，每种最多一张；不传角色时默认为首帧。
- Ref 支持最多 9 张参考图片、3 个参考视频和 3 个参考音频，合计最多 12 个参考素材；可额外指定一张首帧和一张尾帧。
- Ref 视频和音频单个时长为 2～15 秒，同类素材总时长不超过 15 秒。
- `reference_audio` 不能单独使用，至少需要一个 `reference_image` 或 `reference_video`。

### 6.2.1 创建 Ref2VA 任务

Ref 图片、视频和音频支持 HTTP/HTTPS URL 或 `data:...;base64,...`。

Ref2VA 部署默认加载 `lightx2v/Minimax-h3-Turbo` 的
`minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`，使用 scale `1.0`、alpha `8`
和动态 LoRA。H3 Server 显式传入采样参数 `9 / 6.0 / 3.0`（步数 / video shift / audio shift），
在 SGLang v0.5.19 中执行 8 次去噪计算。客户端仍传 `model: "MiniMax-H3"`，
无需增加 LoRA 字段；FL2VA 继续使用原有采样参数。
权重须提前下载并只读挂载到 `/lora`，文件缺失时无法启动，不会自动回退基座。
下载、启动和回退操作见仓库 README 及 Helm Chart README。

Ref2VA 可同时指定 `first_frame`、`last_frame`（每种最多一张，也可只指定其中一种），
请求仍路由到 Ref2VA 池，`ratio` 按请求值传入。首尾帧不会替代 `reference_audio`
要求的参考图片或参考视频。混合条件需要 Ref2VA 后端使用 **SGLang v0.5.19 或更新版本**。
仅有首尾帧、没有参考素材的请求仍走 FL2VA。

```bash
curl -X POST 'http://localhost:8080/v2/video_generation' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-H3",
    "content": [
      {"type": "text", "text": "水獭叼走水豚头上的橘子并吃掉"},
      {
        "type": "image_url",
        "role": "reference_image",
        "image_url": {"url": "https://example.com/input.jpg"}
      },
      {
        "type": "video_url",
        "role": "reference_video",
        "video_url": {"url": "https://example.com/input.mp4"}
      },
      {
        "type": "image_url",
        "role": "first_frame",
        "image_url": {"url": "https://example.com/first-frame.png"}
      },
      {
        "type": "image_url",
        "role": "last_frame",
        "image_url": {"url": "https://example.com/last-frame.png"}
      }
    ],
    "resolution": "768P",
    "duration": 5,
    "ratio": "adaptive"
  }'
```

Ref 图片支持 JPEG、PNG、WebP、HEIC、HEIF；视频支持 MP4/MOV（H.264/H.265）；音频支持 WAV、MP3。

### 6.3 查询单个任务

```bash
curl 'http://localhost:8080/v2/query/video_generation/<task_id>'
```

任务状态包括：`queued`、`running`、`succeeded`、`failed`、`cancelled`。成功后响应中的 `task.content.url` 是视频下载地址。

### 6.4 查询任务列表

```bash
curl 'http://localhost:8080/v2/query/video_generation?page_num=1&page_size=20&filter.status=succeeded'
```

支持参数：`page_num`、`page_size`、`filter.status`、`filter.model`、`filter.task_type` 和可重复传入的 `filter.task_ids`。`page_size` 最大为 100。

### 6.5 删除或取消任务

```bash
curl -X DELETE 'http://localhost:8080/v2/video_generation/<task_id>'
```

- `queued` 任务会被取消。
- `succeeded` 或 `failed` 任务会被删除。
- 当前不支持删除正在运行的任务，会返回 HTTP 400。

### 6.6 下载视频

```bash
curl -L 'http://localhost:8080/v2/video_generation/<task_id>/content' \
  -o output.mp4
```

下载接口支持 HTTP `Range`。对外代理或 Ingress 需要保留 `Range`、`X-Forwarded-Host` 和 `X-Forwarded-Proto` 请求头。

## 7. 其他接口

| 接口 | 服务 | 用途 |
|---|---|---|
| `GET /healthz` | Router、H3 Server | 存活检查。Router 只检查自身，不探测后端。 |
| `GET /readyz` | H3 Server | 就绪检查。 |
| `GET /metrics` | H3 Server | Prometheus 指标。 |
| `GET /dashboard` | Router、H3 Server | 任务 Dashboard；对外建议访问 Router。 |

## 8. Callback

请求可以携带 `callback_url`。创建任务时服务会先向该地址发送 challenge，Callback 服务需要原样返回：

```json
{"challenge":"收到的 challenge 值"}
```

校验通过后，服务会推送 `queued`、`running` 和最终状态：

```json
{"task":{"id":"...","status":"running"}}
```

默认拒绝内网 Callback URL；仅在可信网络中设置 `H3_ALLOW_PRIVATE_URLS=true`。

## 9. 部署注意事项

- Router 对外暴露，H3 Server、SGLang 和 PostgreSQL 建议只在集群内访问。
- Ingress 如果接收 Base64 图片、视频或音频，请把请求体上限设置为至少 64 MiB。
- Ingress 必须正确传递 `X-Forwarded-Host` 和 `X-Forwarded-Proto`，否则生成的下载地址可能不正确。
- H3 Server 和 SGLang 的共享盘挂载路径必须与环境变量一致。
- 不要把多个互不共享任务状态的 SGLang 实例放在同一个随机负载均衡 Service 后面；一个 H3 Server 应使用一个稳定的 SGLang 地址。
- API 查询数据默认保留 7 天，资产默认保留 30 天，旧任务数据仍可兼容当前表结构。
