#!/usr/bin/env bash
set -e

# 多个 H3 Server 地址使用逗号分隔。部署 Ref Server 后填写 H3_REF2VA_SERVICE_URLS。
docker_network="${H3_DOCKER_NETWORK:-minimax-h3}"
docker network inspect "$docker_network" >/dev/null 2>&1 || docker network create "$docker_network" >/dev/null
docker rm -f minimax-h3-router 2>/dev/null || true
docker run -d \
  --name minimax-h3-router \
  --network "$docker_network" \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp \
  -e "H3_FL2VA_SERVICE_URLS=${H3_FL2VA_SERVICE_URLS:-http://minimax-h3-api:8000}" \
  -e "H3_REF2VA_SERVICE_URLS=${H3_REF2VA_SERVICE_URLS:-}" \
  -p 8080:8080 \
  "${H3_ROUTER_IMAGE:-4pdosc/minimax-h3-router:latest}"
