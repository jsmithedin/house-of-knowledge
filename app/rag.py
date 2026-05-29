import logging

from app.bedrock import BedrockClient, BedrockError
from app.citations import format_sources_section
from app.filters import build_where_clause
from app.store import NoteStore
from app.usage import UsageStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    'You are a D&D campaign lore assistant for the "Normal Door Opening" campaign. '
    "Answer questions using only the provided session notes. "
    "If the notes don't contain enough information, say so — don't invent lore. "
    "Be concise but complete. When referencing events, include session numbers and dates where available."
)


class RagPipeline:
    def __init__(
        self,
        store: NoteStore,
        embedder,
        bedrock: BedrockClient,
        wiki_base_url: str,
        chat_history_window: int,
        usage_store: UsageStore | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.bedrock = bedrock
        self.wiki_base_url = wiki_base_url
        self.chat_history_window = chat_history_window
        self.usage_store = usage_store

    def query(
        self,
        message: str,
        arc: str | None,
        session: str | None,
        tag: str | None,
        k: int,
        history: list[dict],
    ) -> tuple[str, str]:
        where = build_where_clause(arc, session, tag)
        embedding = self.embedder.embed_query(message)
        results = self.store.query(query_embedding=embedding, n_results=k, where=where)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            return (
                "No matching notes found. Try broadening your filters or rephrasing.",
                "",
            )

        context_parts = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(
                f"[Session {meta.get('session', '?')} — {meta.get('heading', '?')} "
                f"({meta.get('date', '?')})]\n{doc}"
            )
        context = "\n\n---\n\n".join(context_parts)

        history_text = ""
        window = history[-self.chat_history_window :]
        for msg in window:
            role = msg["role"].capitalize()
            history_text += f"{role}: {msg['content']}\n"

        user_message = ""
        if history_text:
            user_message += f"Conversation so far:\n{history_text}\n"
        user_message += f"Retrieved session notes:\n{context}\n\nQuestion: {message}"

        try:
            result = self.bedrock.invoke(SYSTEM_PROMPT, user_message)
        except BedrockError:
            return ("Generation failed — try again.", "")

        if self.usage_store is not None:
            try:
                self.usage_store.record_event(
                    self.bedrock.model_id,
                    result.input_tokens,
                    result.output_tokens,
                )
            except Exception:
                log.exception("Failed to record token usage")

        sources = format_sources_section(self.wiki_base_url, metadatas)
        return (result.text, sources)
