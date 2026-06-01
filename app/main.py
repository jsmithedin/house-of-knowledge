import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.bedrock import BedrockClient
from app.config import HAIKU_MODEL_ID, MODEL_LABELS, NOVA_LITE_MODEL_ID, Settings
from app.embedder import Embedder
from app.indexing import index_vault
from app.citations import resolve_wikilinks
from app.rag import RagPipeline
from app.store import NoteStore
from app.usage import UsageStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REINDEX_PORT = 7861
EMPTY_BANNER_MSG = "⚠️ No notes indexed yet. Run sync-and-index.sh from your Mac."


class _ReindexHandler(BaseHTTPRequestHandler):
    settings: Settings
    embedder: Embedder

    def do_POST(self):
        if self.path != "/reindex":
            self.send_error(404)
            return
        stats = index_vault(
            obsidian_dir=self.settings.obsidian_dir,
            chroma_dir=self.settings.chroma_dir,
            embedder=self.embedder,
            collection_name=self.settings.collection_name,
        )
        log.info("Reindex complete: %s", stats)
        body = json.dumps(stats).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _start_reindex_server(settings: Settings, embedder: Embedder) -> None:
    handler = type(
        "ReindexHandler",
        (_ReindexHandler,),
        {"settings": settings, "embedder": embedder},
    )
    server = HTTPServer(("0.0.0.0", REINDEX_PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Reindex server listening on port %s", REINDEX_PORT)


@st.cache_resource
def _load_pipeline() -> tuple[RagPipeline, NoteStore, dict[str, BedrockClient]]:
    settings = Settings()
    embedder = Embedder()
    store = NoteStore(chroma_dir=settings.chroma_dir, collection_name=settings.collection_name)
    bedrock_clients = {
        NOVA_LITE_MODEL_ID: BedrockClient(model_id=NOVA_LITE_MODEL_ID, region=settings.aws_region),
        HAIKU_MODEL_ID: BedrockClient(model_id=HAIKU_MODEL_ID, region=settings.aws_region),
    }
    default_bedrock = bedrock_clients.get(
        settings.bedrock_model_id, bedrock_clients[NOVA_LITE_MODEL_ID]
    )
    usage_store = UsageStore(settings.usage_db_path)
    usage_store.rollup_stale_months()
    pipeline = RagPipeline(
        store=store,
        embedder=embedder,
        bedrock=default_bedrock,
        wiki_base_url=settings.wiki_base_url,
        chat_history_window=settings.chat_history_window,
        usage_store=usage_store,
    )
    _start_reindex_server(settings, embedder)
    return pipeline, store, bedrock_clients


def _respond(
    message: str,
    history: list[dict],
    arc: str,
    session: str,
    tag: str,
    k: int,
    agentic: bool,
    bedrock: BedrockClient,
) -> str:
    answer, sources = pipeline.query(
        message=message,
        arc=arc,
        session=session,
        tag=tag,
        k=k,
        history=history,
        agentic=agentic,
        bedrock=bedrock,
    )
    answer = resolve_wikilinks(answer, pipeline.wiki_base_url)
    return f"{answer}\n\n{sources}" if sources else answer


st.set_page_config(
    page_title="Normal Door Opening — Campaign Lore",
    page_icon="🎲",
    layout="wide",
)

pipeline, store, bedrock_clients = _load_pipeline()

_MODEL_OPTIONS = [NOVA_LITE_MODEL_ID, HAIKU_MODEL_ID]
_default_model_idx = (
    _MODEL_OPTIONS.index(pipeline.bedrock.model_id)
    if pipeline.bedrock.model_id in _MODEL_OPTIONS
    else 0
)

st.title("🎲 Normal Door Opening — Campaign Lore")

if store.count == 0:
    st.warning(EMPTY_BANNER_MSG)

distinct = store.get_distinct_metadata()
col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 2, 2])
with col1:
    arc = st.selectbox("Arc", ["All"] + distinct["arcs"], index=0)
with col2:
    session = st.selectbox("Session", ["All"] + distinct["sessions"], index=0)
with col3:
    tag = st.selectbox("Tag", ["All"] + distinct["tags"], index=0)
with col4:
    k = st.slider("k (chunks)", min_value=1, max_value=15, value=5)
with col5:
    rag_mode = st.radio("RAG mode", ["Standard", "Agentic"], horizontal=True)
with col6:
    model_label = st.radio(
        "Model",
        [MODEL_LABELS[mid] for mid in _MODEL_OPTIONS],
        index=_default_model_idx,
        horizontal=True,
    )
model_id = next(mid for mid, label in MODEL_LABELS.items() if label == model_label)
selected_bedrock = bedrock_clients[model_id]

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Searching notes..."):
            full_answer = _respond(
                prompt,
                history,
                arc,
                session,
                tag,
                int(k),
                agentic=(rag_mode == "Agentic"),
                bedrock=selected_bedrock,
            )
        st.markdown(full_answer)
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
