#!/usr/bin/env python3
# scripts/index_notes.py
import logging
import sys
from pathlib import Path

import frontmatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_markdown
from app.config import Settings
from app.embedder import Embedder
from app.store import NoteStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def index_vault(
    obsidian_dir: str,
    chroma_dir: str,
    embedder,
    collection_name: str = "campaign_notes",
) -> dict:
    store = NoteStore(chroma_dir=chroma_dir, collection_name=collection_name)
    obs_path = Path(obsidian_dir)

    all_ids: set[str] = set()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    skipped = 0
    indexed_files = 0

    for md_file in sorted(obs_path.rglob("*.md")):
        post = frontmatter.load(md_file)
        if not post.metadata:
            log.warning("Skipping %s — no frontmatter", md_file)
            skipped += 1
            continue

        rel_path = str(md_file.relative_to(obs_path))
        tags = post.metadata.get("tags", [])
        tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)

        chunks = chunk_markdown(post.content, source_path=rel_path)
        if not chunks:
            log.warning("Skipping %s — no H2/H3 headings", md_file)
            skipped += 1
            continue

        indexed_files += 1
        for chunk in chunks:
            all_ids.add(chunk.chunk_id)
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "source_path": rel_path,
                "heading": chunk.heading,
                "session": str(post.metadata.get("session", "")),
                "arc": str(post.metadata.get("arc", "")),
                "date": str(post.metadata.get("date", "")),
                "tags": tags_str,
            })

    if ids:
        embeddings = embedder.embed_documents(documents)
        store.upsert_chunks(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    stale = store.get_all_ids() - all_ids
    store.delete_ids(list(stale))
    if stale:
        log.info("Deleted %d stale chunks", len(stale))

    return {
        "indexed_files": indexed_files,
        "indexed_chunks": len(ids),
        "skipped_files": skipped,
        "deleted_chunks": len(stale),
    }


def main():
    settings = Settings()
    embedder = Embedder()
    stats = index_vault(
        obsidian_dir=settings.obsidian_dir,
        chroma_dir=settings.chroma_dir,
        embedder=embedder,
        collection_name=settings.collection_name,
    )
    log.info("Index complete: %s", stats)


if __name__ == "__main__":
    main()
