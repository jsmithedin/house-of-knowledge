import chromadb


class NoteStore:
    def __init__(self, chroma_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def get_all_ids(self) -> set[str]:
        if self._collection.count() == 0:
            return set()
        result = self._collection.get(include=[])
        return set(result["ids"])

    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def get_distinct_metadata(self) -> dict[str, list[str]]:
        if self._collection.count() == 0:
            return {"arcs": [], "sessions": [], "tags": []}
        result = self._collection.get(include=["metadatas"])
        arcs: set[str] = set()
        sessions: set[str] = set()
        tags: set[str] = set()
        for meta in result["metadatas"]:
            if meta.get("arc"):
                arcs.add(meta["arc"])
            if meta.get("session"):
                sessions.add(str(meta["session"]))
            for tag in meta.get("tags", "").split(","):
                tag = tag.strip()
                if tag:
                    tags.add(tag)
        return {
            "arcs": sorted(arcs),
            "sessions": sorted(sessions, key=lambda s: int(s) if s.isdigit() else s),
            "tags": sorted(tags),
        }
