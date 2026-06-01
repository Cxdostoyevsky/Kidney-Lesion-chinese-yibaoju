#!/usr/bin/env bash
set -euo pipefail

remote="chenxi@101.6.64.59:/home/chenxi/Kidney_Lesion/"
ssh_command="ssh -p 8834"

args=(
  --archive
  --verbose
  --exclude=.git/
  --exclude=.DS_Store
  --exclude=.env
  --exclude=.env.*
  --exclude=__pycache__/
  --exclude=.pytest_cache/
  --exclude=.mypy_cache/
  --exclude=.ruff_cache/
  --exclude=.venv/
  --exclude=venv/
  --exclude=outputs/
  --exclude=logs/
  --exclude=checkpoints/
  --exclude=*.pt
  --exclude=*.pth
  --exclude=*.ckpt
)

if [[ "${1:-}" == "--apply" ]]; then
  echo "Uploading local files to ${remote}"
elif [[ $# -eq 0 ]]; then
  args+=(--dry-run)
  echo "Preview only. Run ./sync_to_server.sh --apply to upload these files."
else
  echo "Usage: $0 [--apply]" >&2
  exit 2
fi

rsync "${args[@]}" -e "${ssh_command}" ./ "${remote}"
