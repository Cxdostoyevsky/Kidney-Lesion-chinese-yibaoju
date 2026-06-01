# Project Workflow

## Server

- SSH: `ssh -p 8834 chenxi@101.6.64.59`
- Local project root: this directory
- Remote project root: `/home/chenxi/Kidney_Lesion`

## GitHub

- Repository: `https://github.com/Cxdostoyevsky/Kidney-Lesion-chinese-yibaoju.git`
- Use Git commits to record code changes locally.
- Use `git pull --rebase` before starting work when the local branch is already
  tracking the GitHub branch.
- Use `git push` after committing reviewed code changes.

## Sync Rules

- Edit and verify code locally before uploading it to the server.
- Use `./sync_from_server.sh` to preview files that would be downloaded from the
  server before starting work on server-side changes.
- Use `./sync_from_server.sh --apply` only after reviewing the preview.
- Use `./sync_to_server.sh` to preview the files that would be uploaded.
- Use `./sync_to_server.sh --apply` only after reviewing the preview.
- Do not delete remote files automatically. If deletion is needed, review and run
  the corresponding `rsync --delete` command manually.
- Do not commit or upload passwords, private keys, tokens, datasets, model
  weights, runtime output, or local environment files unless explicitly needed.
- Use Git locally to record code changes before uploading them.
