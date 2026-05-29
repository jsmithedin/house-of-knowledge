# DnD RAG — Design Spec

**Project:** house-of-knowledge  
**Date:** 2026-05-29  
**Status:** Approved (brainstorming)

## Goal

Build a RAG application over Obsidian D&D session notes, served to a small group (DM + 4 players) via a web UI. The app answers lore questions, supports DM prep, and helps players catch up on past sessions.

**Published wiki:** https://wordlewarriors.github.io/normal-door-opening/

## Decisions Summary

| Decision | Choice |
|----------|--------|
| Primary use cases | Mixed — fact recall, DM prep, player catch-up |
| Spoiler control | None — all authenticated users see full vault |
| Citations | Always shown, linking to published wiki (path-based URLs) |
| Metadata filters | Independent dropdowns; set filters combine with AND |
| Chat history | Sliding window (default 8 messages) |
| Re-indexing | Mac rsync + SSH one-liner triggers `docker exec` indexer |
| Vault scale | Medium — 50–200 session notes, tens of MB markdown |
| Bedrock model | Configurable via `BEDROCK_MODEL_ID` env var (default: Nova Lite) |
| Architecture | Monolithic single-container (Approach 1) |

## Architecture

```
Mac (Obsidian)                    NUC (Docker)
─────────────                     ──────────────
rsync + ssh one-liner  ──────►   data/obsidian/  (vault mirror)
                                  data/chromadb/  (persistent)
                                       │
                                  ┌────▼────┐
                                  │  app    │
                                  │ Gradio  │◄── Cloudflare Tunnel
                                  │ BGE-M3  │    + Access (Google OAuth)
                                  │ Chroma  │
                                  └────┬────┘
                                       │
                                  AWS Bedrock (eu-west-2)
                                  Nova Lite / Haiku via env var
```

### Components

| Component | Role |
|-----------|------|
| `docker-compose.yml` | Single `app` service, two volume mounts, env vars for Bedrock + wiki base URL |
| `Dockerfile` | Python 3.11 slim, sentence-transformers + BGE-M3 pre-downloaded at build |
| `app/main.py` | Gradio UI, RAG pipeline, Chroma query, Bedrock invoke, citation rendering |
| `scripts/index_notes.py` | Parse frontmatter, chunk by H2/H3, embed, upsert to ChromaDB |
| `scripts/sync-and-index.sh` | Mac-side: rsync vault → NUC, SSH trigger `docker exec ... index_notes.py` |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEDROCK_MODEL_ID` | `amazon.nova-lite-v1:0` | Bedrock generation model (swap to Haiku via env) |
| `AWS_REGION` | `eu-west-2` | Bedrock region |
| `AWS_ACCESS_KEY_ID` | — | IAM user credentials (in `.env`, not git) |
| `AWS_SECRET_ACCESS_KEY` | — | IAM user credentials (in `.env`, not git) |
| `WIKI_BASE_URL` | `https://wordlewarriors.github.io/normal-door-opening` | Citation link prefix |
| `CHAT_HISTORY_WINDOW` | `8` | Max messages in sliding window |

### Auth Boundary

Cloudflare Access handles identity (Google OAuth, email allowlist of 6 addresses). The Gradio app has no login of its own and trusts all authenticated users equally.

## Indexing & Data Model

### Frontmatter Schema

Each session note must have YAML frontmatter:

```yaml
---
session: 42
arc: "Ivaran Sylhorn's Conspiracy"
date: 2025-02-04
tags:
  - npc/ivaran
  - location/manor
---
```

`index_notes.py` reads these fields directly. Notes without frontmatter are skipped with a warning logged.

### Chunking Strategy

- Split on `##` (H2) and `###` (H3) headings
- Each chunk = heading text + body until the next heading of equal or higher level
- Chunk ID = `{relative_path}#{heading_slug}` (stable across re-indexes)
- Parent note metadata (session, arc, date, tags, file path) attached to every chunk

### ChromaDB Metadata per Chunk

| Field | Example | Used for |
|-------|---------|----------|
| `source_path` | `sessions/2025-02-04-unmasking.md` | Citation URL construction |
| `heading` | `Unmasking the Ascension Ritual` | Citation display |
| `session` | `42` | Filter dropdown |
| `arc` | `Ivaran Sylhorn's Conspiracy` | Filter dropdown |
| `date` | `2025-02-04` | Sorting, display |
| `tags` | `npc/ivaran,location/manor` | Filter dropdown (comma-separated) |

### Citation URLs

Built at query time from indexed metadata:

```
{WIKI_BASE_URL}/{source_path_without_.md}
```

Example: `sessions/2025-02-04-unmasking.md` → `https://wordlewarriors.github.io/normal-door-opening/sessions/2025-02-04-unmasking`

Rendered as markdown links: **[Session 42 — Unmasking the Ascension Ritual](url)**

### Indexing Behavior

- **Upsert by chunk ID** — re-running updates changed chunks, leaves untouched ones alone
- **Deletion** — if a note file is removed from the vault, its chunks are deleted from Chroma on next index pass
- **Scope** — only `.md` files under `data/obsidian/`; attachments/images ignored
- **Trigger** — Mac-side `sync-and-index.sh`: rsync vault → NUC, then `ssh nuc 'docker exec ... python scripts/index_notes.py'`

### Filter Dropdown Population

At app startup, query ChromaDB for distinct values of `arc`, `session`, and `tags`. Refreshed on container restart (acceptable for v1 — indexing happens post-session, not mid-game).

## RAG Query Pipeline

### Query Flow

```
User message + optional filters (arc, session, tag) + k
        │
        ▼
Build ChromaDB where-clause from set filters (AND)
        │
        ▼
Embed query with BGE-M3 (same model as indexing)
        │
        ▼
ChromaDB similarity search → top-k chunks
        │
        ▼
Assemble prompt:
  - System: campaign assistant instructions
  - Context: retrieved chunks with metadata
  - History: last N messages (sliding window)
  - User: current question
        │
        ▼
Bedrock InvokeModel (model from BEDROCK_MODEL_ID)
        │
        ▼
Response + inline citations appended
```

### System Prompt

> You are a D&D campaign lore assistant for the "Normal Door Opening" campaign. Answer questions using only the provided session notes. If the notes don't contain enough information, say so — don't invent lore. Be concise but complete. When referencing events, include session numbers and dates where available.

Tone: helpful lore oracle, not a rules lawyer. No D&D 5e rules lookups unless they're in the notes.

### Metadata Filtering

Filters apply only when set; unset dropdowns are ignored. Set filters combine with AND:

- arc = "Ivaran Sylhorn's Conspiracy" + tag = "npc/ivaran" → chunks matching both
- session = "42" alone → all chunks from session 42

ChromaDB `where` clause built dynamically. Tag filter uses `$contains` on the comma-separated tags field.

### Retrieval Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| k (chunks) | 5 | 1–15 (slider in UI) |
| Chat history window | 8 messages | Configurable via env var |

### Citation Rendering

Every response ends with a **Sources** section listing all retrieved chunks:

```markdown
---
**Sources:**
- [Session 42 — Unmasking the Ascension Ritual](https://wordlewarriors.github.io/.../sessions/2025-02-04-unmasking)
- [Session 38 — Exploring Ivaran's Manor](https://wordlewarriors.github.io/.../sessions/2024-12-03-manor)
```

Links are always shown — even if the model paraphrased rather than quoting directly.

### Error Handling

| Failure | Behavior |
|---------|----------|
| Bedrock timeout/error | User-facing: "Generation failed — try again." Logged server-side. |
| No chunks retrieved | "No matching notes found. Try broadening your filters or rephrasing." Skip Bedrock call. |
| Empty vault / no index | Startup banner: "No notes indexed yet. Run sync-and-index.sh." |
| Invalid Bedrock model ID | Fail at startup with clear log message. |

## Gradio UI

### Layout

Single-page chat app:

- **Header:** "Normal Door Opening — Campaign Lore"
- **Filter bar:** Arc dropdown, Session dropdown, Tag dropdown, k slider (1–15, default 5)
- **Chat area:** Gradio `Chatbot` with markdown rendering for citation links
- **Input:** Text box + Send button
- **Clear button:** Resets chat history; filters persist

All filter dropdowns default to "All" (no filter applied).

### Startup Behavior

1. Load BGE-M3 model (~30–60s on first start)
2. Connect to ChromaDB persistent volume
3. Query distinct metadata → populate dropdowns
4. If collection is empty, show banner: "No notes indexed yet. Run sync-and-index.sh from your Mac."
5. Bind to `0.0.0.0:7860` for Cloudflare Tunnel

## Infrastructure & Operations

### Docker

```yaml
services:
  app:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data/obsidian:/app/data/obsidian:ro
      - ./data/chromadb:/app/data/chromadb
    env_file: .env
    restart: unless-stopped
```

**Dockerfile:** Python 3.11-slim, BGE-M3 pre-downloaded at build, dependencies: `gradio`, `chromadb`, `sentence-transformers`, `boto3`, `pyyaml`, `python-frontmatter`.

### Memory Budget (8 GB NUC)

| Component | ~RAM |
|-----------|------|
| OS + Docker overhead | 1.5 GB |
| BGE-M3 model | 1.5 GB |
| ChromaDB (50–200 notes) | < 0.5 GB |
| Gradio + Python runtime | 0.5 GB |
| **Headroom** | **~4 GB** |

### AWS Bedrock

| Setting | Value |
|---------|-------|
| Region | `eu-west-2` |
| Model | `BEDROCK_MODEL_ID` env var (default: Nova Lite) |
| IAM | Dedicated user, `bedrock:InvokeModel` on chosen model ARN only |
| Budget alert | $2/month |

Estimated cost: Nova Lite well under $1/month; Haiku ~$1–3/month at expected usage (~5 users, ~20 queries/session, weekly sessions).

### Cloudflare

| Component | Config |
|-----------|--------|
| Tunnel | `cloudflared` on NUC → `dnd.[domain]` → `localhost:7860` |
| DNS | NS record in Route 53 for subdomain, delegated to Cloudflare |
| Access | Google OAuth, email allowlist (6 addresses), no self-registration |

### Mac Sync Script

```bash
#!/bin/bash
VAULT_PATH="$HOME/path/to/obsidian-vault"
NUC_HOST="nuc.local"

rsync -avz --delete "$VAULT_PATH/" "$NUC_HOST:/app/data/obsidian/"
ssh "$NUC_HOST" 'cd /app && docker compose exec -T app python scripts/index_notes.py'
```

Run once after each session. `--delete` keeps NUC mirror in sync when notes are removed.

## File Structure

```
house-of-knowledge/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
├── app/
│   └── main.py
├── scripts/
│   ├── index_notes.py
│   └── sync-and-index.sh
└── data/                 # gitignored, lives on NUC
    ├── obsidian/
    └── chromadb/
```

## Out of Scope (v1)

- Spoiler / per-user content filtering
- Re-ranking or hybrid search
- Auto-watch indexing
- Filter refresh without container restart
- D&D rules lookups
- Multi-vault support

## Approaches Considered

| Approach | Verdict |
|----------|---------|
| **1. Monolithic container** | **Selected** — simplest ops, one model load, matches NUC constraints |
| 2. Split app + indexer containers | Rejected — doubles BGE-M3 memory (~3 GB), overkill for vault size |
| 3. Enhanced retrieval (re-ranker) | Deferred to v2 — metadata filters sufficient for v1, CPU cost on NUC |
