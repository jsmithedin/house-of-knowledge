# house-of-knowledge

RAG app over Obsidian D&D session notes. See `docs/superpowers/specs/2026-05-29-dnd-rag-design.md` for design.

## Quick start (local dev)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in AWS credentials
pytest -v
python app/main.py
```

## Deploy (NUC)

```bash
cp .env.example .env  # fill in AWS credentials
docker compose up -d --build
```

## Index notes

From Mac after a session:

```bash
VAULT_PATH=~/path/to/vault NUC_HOST=nuc.local ./scripts/sync-and-index.sh
```
