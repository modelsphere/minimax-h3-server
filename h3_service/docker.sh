#!/usr/bin/env bash
set -e

source "$(dirname "${BASH_SOURCE[0]}")/ref2va_profile.sh"
server_image="${H3_SERVER_IMAGE:-4pdosc/minimax-h3-server:latest}"
data_dir="${H3_DATA_DIR:-/srv/minimax-h3}"
service_capability=$(printf '%s' "${H3_SERVICE_CAPABILITY:-fl2va}" | tr '[:upper:]' '[:lower:]')
case "$service_capability" in
  fl2va|ref2va) ;;
  *) echo "H3_SERVICE_CAPABILITY must be fl2va or ref2va" >&2; exit 1 ;;
esac
if [[ "$service_capability" == ref2va ]]; then
  ref2va_configure
  ref2va_check_server_image "$server_image"
fi

# 数据库、SGLang 地址和宿主机存储目录用环境变量覆盖。
mkdir -p "$data_dir"
docker rm -f minimax-h3-api 2>/dev/null || true
# --restart unless-stopped \
docker run -d \
  --name minimax-h3-api \
  -e "H3_DATABASE_URL=${H3_DATABASE_URL:-postgresql://h3:h3@127.0.0.1:5432/minimaxh3}" \
  -e H3_STORAGE_ROOT=/data/minimax-h3 \
  -e "H3_SGLANG_URL=${H3_SGLANG_URL:-http://127.0.0.1:30010}" \
  -e H3_SGLANG_MEDIA_ROOT=/data/minimax-h3 \
  -e "H3_SERVICE_CAPABILITY=$service_capability" \
  "${ref2va_sampling_env_args[@]}" \
  -e H3_QUERY_RETENTION_DAYS=7 \
  -e H3_ASSET_RETENTION_DAYS=30 \
  -e H3_ALLOW_PRIVATE_URLS=false \
  -p 8000:8000 \
  -v "$data_dir:/data/minimax-h3" \
  "$server_image"
