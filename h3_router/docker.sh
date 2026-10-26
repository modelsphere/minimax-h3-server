#!/usr/bin/env bash
set -e

# 多个 H3 Server 地址使用逗号分隔。部署 Ref Server 后填写 H3_REF2VA_SERVICE_URLS。
docker rm -f minimax-h3-router 2>/dev/null || true
docker run -d \
  --name minimax-h3-router \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp \
  -e "H3_FL2VA_SERVICE_URLS=${H3_FL2VA_SERVICE_URLS:-http://127.0.0.1:8000}" \
  -e "H3_REF2VA_SERVICE_URLS=${H3_REF2VA_SERVICE_URLS:-}" \
  -p 8080:8080 \
  "${H3_ROUTER_IMAGE:-4pdosc/minimax-h3-router:latest}"
