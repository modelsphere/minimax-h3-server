# H3 Server 不跑模型(推理在 SGLang 那个 sidecar 里),只做 API + 任务库 + 转发,
# 依赖 av / pillow / pillow-heif / psycopg / fastapi / uvicorn 全部有 manylinux wheel,
# 不需要 CUDA 运行时。原来的 nvcr-pytorch:25.06-py3 基座 27GB,换 slim 后 353MB。
#
# 27GB 不只是占盘:kubelet 首次拉常 context canceled 进 ImagePullBackOff,
# 每次上新节点都要重拉几十 GB,滚动更新也慢。
# 换基座后实测:PyAV 载入 561 个 codec、pillow-heif 注册成功、PIL 真解码、
# 完整测试套件 50 passed / 1 skipped(连真 postgres 跑)。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./requirements.txt
# --progress-bar off: pip's progress bar starts a refresh thread, which fails with
# "can't start new thread" on CI runners that cap the thread count.
# PIP_INDEX_URL picks a mirror, e.g. --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt -i "$PIP_INDEX_URL"
COPY . ./h3_service

EXPOSE 8000
CMD ["python", "-m", "h3_service.server"]
