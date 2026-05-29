import logging

import gradio as gr

from app.bedrock import BedrockClient
from app.config import Settings
from app.embedder import Embedder
from app.rag import RagPipeline
from app.store import NoteStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

settings = Settings()
embedder = Embedder()
store = NoteStore(chroma_dir=settings.chroma_dir, collection_name=settings.collection_name)
bedrock = BedrockClient(model_id=settings.bedrock_model_id, region=settings.aws_region)
pipeline = RagPipeline(
    store=store,
    embedder=embedder,
    bedrock=bedrock,
    wiki_base_url=settings.wiki_base_url,
    chat_history_window=settings.chat_history_window,
)

distinct = store.get_distinct_metadata()
ARC_CHOICES = ["All"] + distinct["arcs"]
SESSION_CHOICES = ["All"] + distinct["sessions"]
TAG_CHOICES = ["All"] + distinct["tags"]

EMPTY_BANNER = (
    "⚠️ No notes indexed yet. Run sync-and-index.sh from your Mac."
    if store.count == 0
    else None
)


def respond(message, history, arc, session, tag, k):
    if not message.strip():
        return history, ""
    chat_history = [{"role": h["role"], "content": h["content"]} for h in history]
    answer, sources = pipeline.query(
        message=message,
        arc=arc,
        session=session,
        tag=tag,
        k=int(k),
        history=chat_history,
    )
    full_answer = f"{answer}\n\n{sources}" if sources else answer
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_answer})
    return history, ""


with gr.Blocks(title="Normal Door Opening — Campaign Lore") as demo:
    gr.Markdown("# 🎲 Normal Door Opening — Campaign Lore")
    if EMPTY_BANNER:
        gr.Markdown(EMPTY_BANNER)

    with gr.Row():
        arc_dd = gr.Dropdown(choices=ARC_CHOICES, value="All", label="Arc")
        session_dd = gr.Dropdown(choices=SESSION_CHOICES, value="All", label="Session")
        tag_dd = gr.Dropdown(choices=TAG_CHOICES, value="All", label="Tag")
        k_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="k (chunks)")

    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(placeholder="Type your question...", label="Question")
    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    submit.click(respond, [msg, chatbot, arc_dd, session_dd, tag_dd, k_slider], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, arc_dd, session_dd, tag_dd, k_slider], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
