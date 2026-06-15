#!/usr/bin/env bash
set -Eeuo pipefail

WORK_DIR="${WORK_DIR:-/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/kidney_lesion_modellllllllll/autoseg/segresnet3D_5fold}"
PYTHON="${PYTHON:-/home/chenxi/anaconda3/envs/nnunet/bin/python}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/肾肿瘤CT/动脉图像_平扫图像_平扫mask_HU值加窗/cz1_cz2/cz2医院_CE}"
DATALIST="${DATALIST:-${WORK_DIR}/cz2_ce_positive_auto3dseg_datalist.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/predictions_cz2_ce_positive_5fold}"
LOG_ROOT="${LOG_ROOT:-${WORK_DIR}/inference_logs_cz2_ce_positive}"
TRAINING_DATALIST="${TRAINING_DATALIST:-${WORK_DIR}/cz1_cy_auto3dseg_datalist_all_testing.json}"
GPU="${GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TMP_ROOT="${TMP_ROOT:-/dev/shm/user-segresnet-cz2-ce-positive}"
FORCE="${FORCE:-0}"

export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable is missing: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${DATALIST}" ]]; then
  echo "External positive datalist is missing: ${DATALIST}" >&2
  exit 2
fi

expected_cases="$(
  "${PYTHON}" - "${DATALIST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    data = json.load(stream)

testing = data.get("testing", [])
if len(testing) != int(data.get("numTesting", -1)):
    raise ValueError("numTesting does not match the testing list")
print(len(testing))
PY
)"

if ((expected_cases == 0)); then
  echo "External datalist contains no testing cases." >&2
  exit 2
fi

for fold in 0 1 2 3 4; do
  bundle="${WORK_DIR}/segresnet_${fold}"
  for required_file in \
    "${bundle}/configs/hyper_parameters.yaml" \
    "${bundle}/scripts/infer.py" \
    "${bundle}/model/model.pt"; do
    if [[ ! -f "${required_file}" ]]; then
      echo "Required fold ${fold} file is missing: ${required_file}" >&2
      exit 2
    fi
  done
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
chmod 700 "${TMP_ROOT}"

while pgrep -f "segresnet_[0-4]/scripts/infer.py.*${TRAINING_DATALIST}" >/dev/null; do
  counts=()
  for fold in 0 1 2 3 4; do
    count="$(find "${WORK_DIR}/predictions_all_5fold/fold${fold}" -type f -name '*.nii.gz' 2>/dev/null | wc -l)"
    counts+=("fold${fold}=${count}")
  done
  echo "[$(date '+%F %T')] Training-set inference is still running (${counts[*]})."
  echo "Waiting ${POLL_SECONDS} seconds before checking again."
  sleep "${POLL_SECONDS}"
done

while true; do
  free_mib="$(
    nvidia-smi \
      --id="${GPU}" \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits |
      tr -d '[:space:]'
  )"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && ((free_mib >= MIN_FREE_MIB)); then
    break
  fi
  echo "[$(date '+%F %T')] GPU ${GPU} has ${free_mib:-unknown} MiB free; need ${MIN_FREE_MIB} MiB."
  sleep "${POLL_SECONDS}"
done

echo "[$(date '+%F %T')] Starting CZ2 CE positive-only five-fold inference."
echo "Cases=${expected_cases}; physical GPU=${GPU}; folds run sequentially."

for fold in 0 1 2 3 4; do
  bundle="${WORK_DIR}/segresnet_${fold}"
  output_dir="${OUTPUT_ROOT}/fold${fold}"
  log_file="${LOG_ROOT}/fold${fold}.log"
  console_log="${LOG_ROOT}/fold${fold}.console.log"
  tmp_dir="${TMP_ROOT}/fold${fold}"

  mkdir -p "${output_dir}" "${tmp_dir}"
  existing_count="$(find "${output_dir}" -type f -name '*.nii.gz' 2>/dev/null | wc -l)"
  if ((existing_count == expected_cases)); then
    echo "[$(date '+%F %T')] Fold ${fold} already has ${existing_count} predictions; skipping."
    continue
  fi
  if ((existing_count > 0)) && [[ "${FORCE}" != "1" ]]; then
    echo "Fold ${fold} has ${existing_count}/${expected_cases} prediction files in ${output_dir}." >&2
    echo "Rerun with FORCE=1 to overwrite matching predictions." >&2
    exit 2
  fi

  chmod 700 "${tmp_dir}"

  echo "[$(date '+%F %T')] Running fold ${fold} on physical GPU ${GPU}."
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export TMPDIR="${tmp_dir}"
    exec "${PYTHON}" "${bundle}/scripts/infer.py" run \
      "--config_file=${bundle}/configs/hyper_parameters.yaml" \
      "--data_file_base_dir=${EXTERNAL_ROOT}" \
      "--data_list_file_path=${DATALIST}" \
      "--pretrained_ckpt_name=${bundle}/model/model.pt" \
      "--infer#data_list_key=testing" \
      "--infer#output_path=${output_dir}" \
      "--auto_scale_allowed=False" \
      "--num_workers=2" \
      "--log_output_file=${log_file}"
  ) >"${console_log}" 2>&1

  output_count="$(find "${output_dir}" -type f -name '*.nii.gz' | wc -l)"
  if ((output_count != expected_cases)); then
    echo "Fold ${fold} produced ${output_count}/${expected_cases} predictions." >&2
    echo "See ${console_log}" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] Fold ${fold} completed with ${output_count} predictions."
done

echo "[$(date '+%F %T')] All five folds completed CZ2 CE positive-only inference."
