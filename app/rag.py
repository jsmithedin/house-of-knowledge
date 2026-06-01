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

AGENTIC_SYSTEM_PROMPT = (
    'You are a D&D campaign lore assistant for the "Normal Door Opening" campaign. '
    "Use the search_knowledge_base tool to look up information from the session notes "
    "before answering. Search as many times as needed to give a complete answer. "
    "If the notes don't contain the information after searching, say so — don't invent lore. "
    "Be concise but complete. When referencing events, include session numbers and dates where available."
)

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the D&D campaign session notes. Call this whenever you need information "
        "to answer the question. You can call it multiple times with different queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the session notes.",
            },
            "k": {
                "type": "integer",
                "description": "Number of note chunks to retrieve (1–10).",
            },
        },
        "required": ["query"],
    },
}


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
        agentic: bool = False,
    ) -> tuple[str, str]:
        where = build_where_clause(arc, session, tag)

        history_text = ""
        window = history[-self.chat_history_window:]
        for msg in window:
            role = msg["role"].capitalize()
            history_text += f"{role}: {msg['content']}\n"

        if agentic:
            return self._query_agentic(message, history_text, where, k)
        return self._query_standard(message, history_text, where, k)

    def _query_standard(
        self,
        message: str,
        history_text: str,
        where: dict | None,
        k: int,
    ) -> tuple[str, str]:
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

        user_message = ""
        if history_text:
            user_message += f"Conversation so far:\n{history_text}\n"
        user_message += f"Retrieved session notes:\n{context}\n\nQuestion: {message}"

        try:
            result = self.bedrock.invoke(SYSTEM_PROMPT, user_message)
        except BedrockError:
            return ("Generation failed — try again.", "")

        self._record_usage(result.input_tokens, result.output_tokens)
        sources = format_sources_section(self.wiki_base_url, metadatas)
        return (result.text, sources)

    def _query_agentic(
        self,
        message: str,
        history_text: str,
        where: dict | None,
        k: int,
    ) -> tuple[str, str]:
        all_metadatas: list[dict] = []

        def execute_tool(name: str, tool_input: dict) -> str:
            if name != "search_knowledge_base":
                return "Unknown tool."
            query = tool_input["query"]
            n = min(max(int(tool_input.get("k", k)), 1), 10)
            embedding = self.embedder.embed_query(query)
            results = self.store.query(query_embedding=embedding, n_results=n, where=where)
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            all_metadatas.extend(metas)
            if not docs:
                return "No results found for that query."
            parts = []
            for doc, meta in zip(docs, metas):
                parts.append(
                    f"[Session {meta.get('session', '?')} — {meta.get('heading', '?')} "
                    f"({meta.get('date', '?')})]\n{doc}"
                )
            return "\n\n---\n\n".join(parts)

        user_content = ""
        if history_text:
            user_content += f"Conversation so far:\n{history_text}\n"
        user_content += f"Question: {message}"

        initial_messages = [
            {"role": "user", "content": [{"type": "text", "text": user_content}]}
        ]

        try:
            result = self.bedrock.invoke_with_tools(
                system_prompt=AGENTIC_SYSTEM_PROMPT,
                initial_messages=initial_messages,
                tools=[SEARCH_TOOL],
                tool_executor=execute_tool,
            )
        except BedrockError:
            return ("Generation failed — try again.", "")

        self._record_usage(result.input_tokens, result.output_tokens)
        sources = format_sources_section(self.wiki_base_url, all_metadatas)
        return (result.text, sources)

    def _record_usage(self, input_tokens: int, output_tokens: int) -> None:
        if self.usage_store is None:
            return
        try:
            self.usage_store.record_event(self.bedrock.model_id, input_tokens, output_tokens)
        except Exception:
            log.exception("Failed to record token usage")
