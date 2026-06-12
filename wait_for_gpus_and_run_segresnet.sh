#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/run_segresnet_5fold.sh}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
REQUIRED_GPUS="${REQUIRED_GPUS:-5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20480}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "Training script is missing or not executable: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

if ! [[ "${REQUIRED_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REQUIRED_GPUS must be a positive integer." >&2
  exit 2
fi

if ! [[ "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_MIB must be a non-negative integer." >&2
  exit 2
fi

if ! [[ "${POLL_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_INTERVAL_SECONDS must be a positive integer." >&2
  exit 2
fi

while true; do
  echo
  echo "[$(date '+%F %T')] Scanning GPUs: need ${REQUIRED_GPUS} with free memory > ${MIN_FREE_MIB} MiB."

  if ! gpu_status="$(
    "${NVIDIA_SMI}" \
      --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )"; then
    echo "nvidia-smi failed; retrying in ${POLL_INTERVAL_SECONDS} seconds." >&2
    sleep "${POLL_INTERVAL_SECONDS}"
    continue
  fi

  printf '%s\n' "${gpu_status}" |
    awk -F',' '
      {
        for (i = 1; i <= NF; i++) {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
        }
        printf "GPU %-2s | %-28s | total %5s MiB | used %5s MiB | free %5s MiB | util %3s%%\n",
          $1, $2, $3, $4, $5, $6
      }
    '

  mapfile -t selected_gpus < <(
    printf '%s\n' "${gpu_status}" |
      awk -F',' -v minimum="${MIN_FREE_MIB}" '
        {
          index = $1
          free = $5
          gsub(/[[:space:]]/, "", index)
          gsub(/[[:space:]]/, "", free)
          if (free > minimum) {
            print free, index
          }
        }
      ' |
      sort -k1,1nr -k2,2n |
      head -n "${REQUIRED_GPUS}" |
      awk '{print $2}'
  )

  if ((${#selected_gpus[@]} >= REQUIRED_GPUS)); then
    selected_csv="$(IFS=,; echo "${selected_gpus[*]}")"
    export CUDA_VISIBLE_DEVICES="${selected_csv}"

    echo "[$(date '+%F %T')] Found ${REQUIRED_GPUS} suitable GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "Stopping GPU scans and starting: ${TRAIN_SCRIPT}"
    exec "${TRAIN_SCRIPT}"
  fi

  echo "[$(date '+%F %T')] Only ${#selected_gpus[@]} suitable GPU(s); checking again in ${POLL_INTERVAL_SECONDS} seconds."
  sleep "${POLL_INTERVAL_SECONDS}"
done
