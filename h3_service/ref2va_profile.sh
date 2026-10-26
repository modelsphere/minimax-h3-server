#!/usr/bin/env bash
# Shared by the API and SGLang launchers. Use the same environment for both.
ref2va_lora_mount_args=()
ref2va_lora_args=()
ref2va_sampling_env_args=()

ref2va_configure() {
  case "${H3_REF2VA_TURBO_ENABLED:-true}" in
    true|1) ref2va_turbo_enabled=true ;;
    false|0) ref2va_turbo_enabled=false ;;
    *) echo "H3_REF2VA_TURBO_ENABLED must be true, false, 1 or 0" >&2; return 1 ;;
  esac
  if [[ "$ref2va_turbo_enabled" == false ]]; then
    return
  fi
  ref2va_lora_dir="${H3_REF2VA_LORA_DIR:-/srv/models/Minimax-h3-Turbo}"
  ref2va_lora_file="${H3_REF2VA_LORA_WEIGHT_NAME:-minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors}"
  ref2va_lora_mount_args=(-v "$ref2va_lora_dir:/lora:ro")
  ref2va_lora_args=(
    --lora-path /lora --lora-weight-name "$ref2va_lora_file"
    --lora-nickname h3-ref2va-turbo --lora-scale "${H3_REF2VA_LORA_SCALE:-1.0}"
    --lora-alpha "${H3_REF2VA_LORA_ALPHA:-8}" --lora-merge-mode auto
  )
  ref2va_sampling_env_args=(
    -e "H3_REF2VA_NUM_INFERENCE_STEPS=${H3_REF2VA_NUM_INFERENCE_STEPS:-9}"
    -e "H3_REF2VA_FLOW_SHIFT=${H3_REF2VA_FLOW_SHIFT:-6.0}"
    -e "H3_REF2VA_AUDIO_FLOW_SHIFT=${H3_REF2VA_AUDIO_FLOW_SHIFT:-3.0}"
  )
}

ref2va_check_server_image() {
  # Keep the protocol check in sync with the Helm chart's files/ref2va-config-check.py
  # (modelsphere/helm-charts, charts/minimax-h3).
  # Run before removing the existing API, in both Turbo and base mode.
  local check_code
  check_code=$(cat <<'PY'
import sys
try:
    from h3_service.config import REF2VA_TURBO_CONFIG_VERSION
except ImportError:
    REF2VA_TURBO_CONFIG_VERSION = None
if REF2VA_TURBO_CONFIG_VERSION != 1:
    sys.exit(
        "Incompatible H3 Server image: Ref2VA requires Turbo config protocol 1 "
        "for both Turbo and base mode. Rebuild the Server from this release "
        "and update server.image.tag / H3_SERVER_IMAGE."
    )
PY
)
  docker run --rm --network none --entrypoint python "$1" -c "$check_code"
}
