#!/usr/bin/env bash
set -euo pipefail

POLICY_NAME="${POLICY_NAME:-openpi}"
HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
PORT="${POLICY_SERVER_PORT:-8000}"

ARGS=(
  "--policy" "${POLICY_NAME}"
  "--host" "${HOST}"
  "--port" "${PORT}"
)

if [[ -n "${POLICY_SUBMISSION_CONFIG:-}" ]]; then
  ARGS+=("--submission-config" "${POLICY_SUBMISSION_CONFIG}")
fi

if [[ -n "${POLICY_DEFAULT_PROMPT:-}" ]]; then
  ARGS+=("--default-prompt" "${POLICY_DEFAULT_PROMPT}")
fi

if [[ -n "${POLICY_RECORD_DIR:-}" ]]; then
  ARGS+=("--record-dir" "${POLICY_RECORD_DIR}")
fi

if [[ "${POLICY_NAME}" == "openpi" ]]; then
  : "${POLICY_CHECKPOINT_DIR:?POLICY_CHECKPOINT_DIR is required for openpi policy}"
  : "${POLICY_CONFIG_NAME:?POLICY_CONFIG_NAME is required for openpi policy}"

  ARGS+=(
    "--checkpoint-dir" "${POLICY_CHECKPOINT_DIR}"
    "--config-name" "${POLICY_CONFIG_NAME}"
  )

  if [[ -n "${POLICY_PYTORCH_DEVICE:-}" ]]; then
    ARGS+=("--pytorch-device" "${POLICY_PYTORCH_DEVICE}")
  fi
elif [[ "${POLICY_NAME}" == "unskild_gr00t" ]]; then
  if [[ -n "${POLICY_CHECKPOINT_URI:-}" ]]; then
    # Backward-compatible fallback for existing env usage.
    ARGS+=("--checkpoint-path" "${POLICY_CHECKPOINT_URI}")
  elif [[ -n "${POLICY_CHECKPOINT_PATH:-}" ]]; then
    ARGS+=("--checkpoint-path" "${POLICY_CHECKPOINT_PATH}")
  elif [[ -z "${POLICY_SUBMISSION_CONFIG:-}" && -n "${POLICY_CHECKPOINT_DIR:-}" ]]; then
    ARGS+=("--checkpoint-path" "${POLICY_CHECKPOINT_DIR}")
  fi

  if [[ -n "${POLICY_DEVICE:-}" ]]; then
    ARGS+=("--device" "${POLICY_DEVICE}")
  fi
else
  echo "[ERROR] Unsupported POLICY_NAME=${POLICY_NAME}. Allowed: openpi, unskild_gr00t" >&2
  exit 1
fi

exec /workspace/.venv/bin/python /workspace/server/serve_hsr_policy_ws.py "${ARGS[@]}"
