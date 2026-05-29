import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    )
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "eu-west-2"))
    wiki_base_url: str = field(
        default_factory=lambda: os.getenv(
            "WIKI_BASE_URL", "https://wordlewarriors.github.io/normal-door-opening"
        ).rstrip("/")
    )
    chat_history_window: int = field(
        default_factory=lambda: int(os.getenv("CHAT_HISTORY_WINDOW", "8"))
    )
    obsidian_dir: str = field(default_factory=lambda: os.getenv("OBSIDIAN_DIR", "data/obsidian"))
    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "data/chromadb"))
    collection_name: str = "campaign_notes"
