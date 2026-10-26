#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
docker build -t "${H3_SERVER_IMAGE:-minimax-h3-server:dev}" "$@" .
