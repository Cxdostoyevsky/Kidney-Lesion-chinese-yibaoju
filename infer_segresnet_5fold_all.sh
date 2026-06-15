#!/usr/bin/env bash
set -Eeuo pipefail

AUTOSEG_ROOT="${AUTOSEG_ROOT:-/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/kidney_lesion_modellllllllll/autoseg}"
WORK_DIR="${WORK_DIR:-${AUTOSEG_ROOT}/segresnet3D_5fold}"
PYTHON="${PYTHON:-/ssd/chenxi/anaconda3/envs/nnunet/bin/python}"
SOURCE_DATALIST="${SOURCE_DATALIST:-${WORK_DIR}/cz1_cy_auto3dseg_datalist_tumor_only_5fold.json}"
INFER_DATALIST="${INFER_DATALIST:-${WORK_DIR}/cz1_cy_auto3dseg_datalist_all_testing.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/predictions_all_5fold}"
LOG_ROOT="${LOG_ROOT:-${WORK_DIR}/inference_logs}"
GPU_LIST="${GPU_LIST:-1,2,3,4,5}"
TMP_ROOT="${TMP_ROOT:-/dev/shm/user-segresnet-infer}"
FORCE="${FORCE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable is missing: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SOURCE_DATALIST}" ]]; then
  echo "Source datalist is missing: ${SOURCE_DATALIST}" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<< "${GPU_LIST}"
if ((${#gpus[@]} < 5)); then
  echo "GPU_LIST must provide at least five GPU indices, one for each fold." >&2
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

"${PYTHON}" - "${SOURCE_DATALIST}" "${INFER_DATALIST}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with source_path.open(encoding="utf-8") as stream:
    datalist = json.load(stream)

training = datalist.get("training", [])
if not training:
    raise ValueError(f"No training samples found in {source_path}")

testing = []
for sample in training:
    if "image" not in sample:
        raise ValueError(f"Training sample has no image entry: {sample}")
    testing.append({"image": sample["image"]})

output = {key: value for key, value in datalist.items() if key != "testing"}
output["testing"] = testing
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")

print(f"Prepared {len(testing)} testing samples in {output_path}")
PY

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "PREPARE_ONLY=1; inference was not started."
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
chmod 700 "${TMP_ROOT}"

declare -a pids=()
declare -a folds=()

for fold in 0 1 2 3 4; do
  gpu="${gpus[$fold]//[[:space:]]/}"
  bundle="${WORK_DIR}/segresnet_${fold}"
  output_dir="${OUTPUT_ROOT}/fold${fold}"
  log_file="${LOG_ROOT}/fold${fold}.log"
  tmp_dir="${TMP_ROOT}/fold${fold}"

  if [[ -d "${output_dir}" ]] && find "${output_dir}" -type f -print -quit | grep -q .; then
    if [[ "${FORCE}" != "1" ]]; then
      echo "Fold ${fold} output is not empty: ${output_dir}" >&2
      echo "Move it aside, or rerun with FORCE=1 to allow overwriting matching files." >&2
      exit 2
    fi
  fi

  mkdir -p "${output_dir}" "${tmp_dir}"
  chmod 700 "${tmp_dir}"

  echo "[$(date '+%F %T')] Starting fold ${fold} on physical GPU ${gpu}."
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TMPDIR="${tmp_dir}"
    exec "${PYTHON}" "${bundle}/scripts/infer.py" run \
      "--config_file=${bundle}/configs/hyper_parameters.yaml" \
      "--data_list_file_path=${INFER_DATALIST}" \
      "--pretrained_ckpt_name=${bundle}/model/model.pt" \
      "--infer#data_list_key=testing" \
      "--infer#output_path=${output_dir}" \
      "--auto_scale_allowed=False" \
      "--num_workers=2" \
      "--log_output_file=${log_file}"
  ) >"${log_file}.console" 2>&1 &

  pid="$!"
  pids+=("${pid}")
  folds+=("${fold}")
  echo "  PID ${pid}, output ${output_dir}, log ${log_file}.console"
done

failed=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  fold="${folds[$index]}"
  if wait "${pid}"; then
    count="$(find "${OUTPUT_ROOT}/fold${fold}" -type f -name '*.nii.gz' | wc -l)"
    echo "[$(date '+%F %T')] Fold ${fold} completed with ${count} prediction files."
  else
    status="$?"
    echo "[$(date '+%F %T')] Fold ${fold} failed with exit code ${status}." >&2
    echo "See ${LOG_ROOT}/fold${fold}.log.console" >&2
    failed=1
  fi
done

if ((failed)); then
  exit 1
fi

echo "[$(date '+%F %T')] All five folds completed inference."
