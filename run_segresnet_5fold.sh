#!/usr/bin/env bash
set -Eeuo pipefail

AUTOSEG_ROOT="${AUTOSEG_ROOT:-/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/kidney_lesion_modellllllllll/autoseg}"
WORK_DIR="${WORK_DIR:-${AUTOSEG_ROOT}/segresnet3D_5fold}"
PYTHON="${PYTHON:-/ssd/chenxi/anaconda3/envs/nnunet/bin/python}"
TORCHRUN="${TORCHRUN:-/ssd/chenxi/anaconda3/envs/nnunet/bin/torchrun}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29541}"
TMPDIR="${SEGRESNET_TMPDIR:-/dev/shm/user-segresnet-tmp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}"
export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR

mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#visible_gpus[@]}"

if ((NPROC_PER_NODE == 0)); then
  echo "CUDA_VISIBLE_DEVICES does not contain any GPU indices." >&2
  exit 2
fi

checkpoint_epoch() {
  "${PYTHON}" - "$1" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)))
PY
}

configured_epochs() {
  "${PYTHON}" - "$1" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
print(int(config["num_epochs"]))
PY
}

run_fold() {
  local fold="$1"
  local bundle="${WORK_DIR}/segresnet_${fold}"
  local config="${bundle}/configs/hyper_parameters.yaml"
  local train_script="${bundle}/scripts/train.py"
  local checkpoint="${bundle}/model/model_final.pt"
  local master_port=$((MASTER_PORT_BASE + fold))
  local num_epochs
  local saved_epoch
  local -a resume_args=()

  if [[ ! -f "${config}" || ! -f "${train_script}" ]]; then
    echo "Fold ${fold} bundle is incomplete: ${bundle}" >&2
    return 1
  fi

  num_epochs="$(configured_epochs "${config}")"

  if [[ -f "${checkpoint}" ]]; then
    saved_epoch="$(checkpoint_epoch "${checkpoint}")"
    if ((saved_epoch >= num_epochs - 1)); then
      echo "[$(date '+%F %T')] Fold ${fold} already completed at epoch ${saved_epoch}; skipping."
      return 0
    fi

    echo "[$(date '+%F %T')] Fold ${fold} resuming from epoch ${saved_epoch}; next epoch is $((saved_epoch + 1))."
    resume_args=(
      "--pretrained_ckpt_name=${checkpoint}"
      "--continue=True"
    )
  else
    echo "[$(date '+%F %T')] Fold ${fold} starting from epoch 0."
  fi

  "${TORCHRUN}" \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${master_port}" \
    "${train_script}" run \
    "--config_file=${config}" \
    "${resume_args[@]}"

  echo "[$(date '+%F %T')] Fold ${fold} finished successfully."
}

on_error() {
  local exit_code="$?"
  echo "[$(date '+%F %T')] Training stopped with exit code ${exit_code}. Later folds were not started." >&2
  exit "${exit_code}"
}
trap on_error ERR

cd "${AUTOSEG_ROOT}"

echo "[$(date '+%F %T')] Starting sequential SegResNet 5-fold training."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; processes=${NPROC_PER_NODE}"
echo "MLFLOW_ALLOW_FILE_STORE=${MLFLOW_ALLOW_FILE_STORE}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "TMPDIR=${TMPDIR}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

for fold in 0 1 2 3 4; do
  run_fold "${fold}"
done

echo "[$(date '+%F %T')] All five folds completed successfully."
