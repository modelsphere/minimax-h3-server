#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
docker_network="${H3_DOCKER_NETWORK:-minimax-h3}"
docker network inspect "$docker_network" >/dev/null 2>&1 || docker network create "$docker_network" >/dev/null
mkdir -p data
docker rm -f minimax-h3-postgres 2>/dev/null || true
docker run -d \
  --name minimax-h3-postgres \
  --network "$docker_network" \
  --restart unless-stopped \
  -e POSTGRES_DB=minimaxh3 \
  -e POSTGRES_USER=h3 \
  -e POSTGRES_PASSWORD=h3 \
  -p 5432:5432 \
  -v "$(pwd)/data:/var/lib/postgresql/data" \
  minimax-h3-postgres
