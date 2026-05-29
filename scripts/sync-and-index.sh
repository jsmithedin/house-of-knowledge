#!/bin/bash
set -euo pipefail

VAULT_PATH="${VAULT_PATH:-$HOME/path/to/obsidian-vault}"
NUC_HOST="${NUC_HOST:-nuc.local}"
NUC_APP_DIR="${NUC_APP_DIR:-/app}"

echo "Syncing vault to NUC..."
rsync -avz --delete \
  "$VAULT_PATH/" \
  "$NUC_HOST:$NUC_APP_DIR/data/obsidian/"

echo "Running indexer on NUC..."
ssh "$NUC_HOST" "cd $NUC_APP_DIR && docker compose exec -T app python scripts/index_notes.py"

echo "Done."
