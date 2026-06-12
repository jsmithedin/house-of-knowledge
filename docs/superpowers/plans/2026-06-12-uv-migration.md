# UV Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two pip requirements files with a single uv project (`pyproject.toml` + `uv.lock`), with eval deps as an optional extras group, and update the Dockerfile and both READMEs to match.

**Architecture:** One `pyproject.toml` at the repo root holds all dependencies. Main deps install with `uv sync`; eval deps install with `uv sync --extra eval`. The Dockerfile copies the uv binary from the official image and installs with `uv sync --frozen --no-dev`. All run commands in documentation gain a `uv run` prefix.

**Tech Stack:** uv 0.7.x, Python 3.11, Streamlit, existing test suite (pytest via `uv run pytest`)

---

## File map

| Action | File |
|--------|------|
| Create | `pyproject.toml` |
| Create (generated) | `uv.lock` |
| Delete | `requirements.txt` |
| Delete | `eval/requirements-eval.txt` |
| Modify | `Dockerfile` |
| Modify | `README.md` |
| Modify | `eval/README.md` |

---

### Task 1: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

Create `pyproject.toml` at the repo root with this exact content:

```toml
[project]
name = "house-of-knowledge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "streamlit==1.44.1",
    "chromadb==0.6.3",
    "sentence-transformers==3.4.1",
    "boto3==1.37.29",
    "python-frontmatter==1.1.0",
    "pyyaml==6.0.2",
]

[project.optional-dependencies]
eval = [
    "tabulate==0.9.0",
]

[dependency-groups]
dev = [
    "pytest==8.3.5",
]
```

- [ ] **Step 2: Generate the lockfile**

```bash
uv lock
```

Expected: `uv.lock` created in the repo root, no errors.

- [ ] **Step 3: Verify main install**

```bash
uv sync
```

Expected: all main deps installed into `.venv/`, no errors.

- [ ] **Step 4: Verify tests pass**

```bash
uv run pytest -v
```

Expected: same pass/fail as before the migration. No import errors.

- [ ] **Step 5: Verify eval extras install**

```bash
uv sync --extra eval
```

Expected: `tabulate` added to `.venv/`, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add pyproject.toml and uv.lock"
```

---

### Task 2: Remove old requirements files

**Files:**
- Delete: `requirements.txt`
- Delete: `eval/requirements-eval.txt`

- [ ] **Step 1: Delete both files**

```bash
git rm requirements.txt eval/requirements-eval.txt
```

- [ ] **Step 2: Verify install still works from lockfile only**

```bash
uv sync
uv run pytest -v
```

Expected: same test results as Task 1 Step 4. No references to the deleted files are hit.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove requirements.txt and eval/requirements-eval.txt"
```

---

### Task 3: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Replace Dockerfile content**

Replace the entire `Dockerfile` with:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download BGE-M3 at build time
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 7860
CMD ["uv", "run", "streamlit", "run", "app/main.py",
     "--server.port=7860",
     "--server.address=0.0.0.0",
     "--server.headless=true"]
```

Key changes from the original:
- `COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv` — fetches the uv binary
- `COPY pyproject.toml uv.lock ./` — copies project manifest and lockfile before app code (enables Docker layer caching)
- `RUN uv sync --frozen --no-dev` — installs from lockfile, excludes pytest
- `RUN uv run python -c ...` — pre-downloads BGE-M3 using uv-managed Python
- `CMD ["uv", "run", "streamlit", ...]` — runs streamlit via uv

- [ ] **Step 2: Verify Dockerfile syntax (if Docker is available)**

```bash
docker build --no-cache -t house-of-knowledge:test . 2>&1 | tail -5
```

Expected: `Successfully built <id>` (BGE-M3 download will take several minutes on first run). If Docker is not available locally, skip this step — the Dockerfile will be verified on next NUC deploy.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: migrate Dockerfile to uv"
```

---

### Task 4: Update README.md quick start

**Files:**
- Modify: `README.md` (Quick start section, lines ~148–155)

- [ ] **Step 1: Replace the quick start code block**

Find this block in the `## Quick start (local dev)` section:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in AWS credentials
pytest -v
streamlit run app/main.py --server.port=7860
```

Replace with:

```bash
uv sync
cp .env.example .env  # fill in AWS credentials
uv run pytest -v
uv run streamlit run app/main.py --server.port=7860
```

- [ ] **Step 2: Verify the file looks correct**

```bash
grep -n "uv sync\|pip install\|venv" README.md
```

Expected: only `uv sync` lines appear — no remaining `pip install` or `venv` references.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README quick start for uv"
```

---

### Task 5: Update eval/README.md

**Files:**
- Modify: `eval/README.md` (Prerequisites section ~line 11, workflow table ~lines 48–57, smoke test ~lines 110–113)

- [ ] **Step 1: Replace the prerequisites install command**

Find in the `## 2. Prerequisites` section:

```bash
pip install -r eval/requirements-eval.txt
```

Replace with:

```bash
uv sync --extra eval
```

- [ ] **Step 2: Replace the workflow table**

Find the workflow table (all `python -m eval.*` and `streamlit run eval/scoring_app.py` commands) and replace with:

| Stage | Command | Output |
|-------|---------|--------|
| 1 (💰 costs money) | `uv run python -m eval.runner --run-id YYYY-MM-DD-name` | `eval/runs/<id>/results.jsonl` |
| 2 (free) | `uv run python -m eval.score_retrieval --run-id YYYY-MM-DD-name` | `eval/runs/<id>/retrieval_scores.jsonl` |
| 3 (free) | `uv run python -m eval.make_blind --run-id YYYY-MM-DD-name` | `eval/runs/<id>/blind/pack.jsonl`, `eval/runs/<id>/blind/private/mapping.json` |
| 4 (free) | `uv run streamlit run eval/scoring_app.py -- --run-id YYYY-MM-DD-name` | `eval/runs/<id>/blind/scores.jsonl` |
| 5 (free) | `uv run python -m eval.unmask --run-id YYYY-MM-DD-name` | `eval/runs/<id>/scored.jsonl` |
| 6 (💰 costs money) | `uv run python -m eval.judge --run-id YYYY-MM-DD-name` | `eval/runs/<id>/judge_scores.jsonl` |
| 7 (free) | `uv run python -m eval.report --run-id YYYY-MM-DD-name` | `eval/runs/<id>/report.md` |

- [ ] **Step 3: Update the blind-scoring session run command (§5)**

Find in `## 5. The blind-scoring session`:

```bash
streamlit run eval/scoring_app.py -- --run-id <id>
```

Replace with:

```bash
uv run streamlit run eval/scoring_app.py -- --run-id <id>
```

- [ ] **Step 4: Update the smoke test example (§7)**

Find in `## 7. Resuming and re-running`:

```bash
python -m eval.runner --run-id 2026-06-15-smoke --limit 2
```

Replace with:

```bash
uv run python -m eval.runner --run-id 2026-06-15-smoke --limit 2
```

- [ ] **Step 5: Verify no bare `python -m` or `streamlit run` remain**

```bash
grep -n "^python\|^streamlit" eval/README.md
```

Expected: no output — all commands now start with `uv run`.

- [ ] **Step 6: Commit**

```bash
git add eval/README.md
git commit -m "docs: update eval/README.md commands for uv"
```

---

### Task 6: Final verification

- [ ] **Step 1: Clean install from scratch**

```bash
rm -rf .venv
uv sync
uv run pytest -v
```

Expected: `.venv` recreated, all tests pass.

- [ ] **Step 2: Verify eval extras**

```bash
uv sync --extra eval
uv run python -c "import tabulate; print('tabulate ok')"
```

Expected: `tabulate ok` printed, no import errors.

- [ ] **Step 3: Verify dev deps are isolated from main install**

```bash
uv sync --no-dev
uv run python -c "import pytest" 2>&1
```

Expected: `ModuleNotFoundError: No module named 'pytest'` — confirms pytest is dev-only and excluded from production install.

- [ ] **Step 4: Confirm no requirements.txt references remain in tracked files**

```bash
git grep "requirements.txt\|requirements-eval"
```

Expected: no output.
