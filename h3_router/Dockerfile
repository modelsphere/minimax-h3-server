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
COPY . ./h3_router

EXPOSE 8080
CMD ["python", "-m", "h3_router.server"]
