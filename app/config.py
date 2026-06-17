import os
from dataclasses import dataclass, field


NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"
HAIKU_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"

MODEL_LABELS: dict[str, str] = {
    NOVA_LITE_MODEL_ID: "Nova Lite",
    HAIKU_MODEL_ID: "Haiku 4.5",
}


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float


# USD per 1M tokens — verify against current AWS Bedrock pricing when updating.
MODEL_PRICING: dict[str, ModelPricing] = {
    NOVA_LITE_MODEL_ID: ModelPricing(input_per_1m=0.06, output_per_1m=0.24),
    HAIKU_MODEL_ID: ModelPricing(input_per_1m=1.00, output_per_1m=5.00),
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0": ModelPricing(
        input_per_1m=1.00, output_per_1m=5.00
    ),
}


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        return 0.0
    return (input_tokens / 1_000_000) * pricing.input_per_1m + (
        output_tokens / 1_000_000
    ) * pricing.output_per_1m


@dataclass(frozen=True)
class Settings:
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv("BEDROCK_MODEL_ID", NOVA_LITE_MODEL_ID)
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
    usage_db_path: str = field(
        default_factory=lambda: os.getenv("USAGE_DB_PATH", "data/usage.sqlite")
    )
    collection_name: str = "campaign_notes"
    langfuse_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "http://192.168.1.231:3000")
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
