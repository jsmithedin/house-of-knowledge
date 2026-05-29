import os

from app.config import MODEL_PRICING, Settings, estimate_cost_usd


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("WIKI_BASE_URL", raising=False)
    s = Settings()
    assert s.bedrock_model_id == "amazon.nova-lite-v1:0"
    assert s.wiki_base_url == "https://wordlewarriors.github.io/normal-door-opening"
    assert s.chat_history_window == 8


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_WINDOW", "12")
    monkeypatch.setenv("WIKI_BASE_URL", "https://example.com/wiki")
    s = Settings()
    assert s.chat_history_window == 12
    assert s.wiki_base_url == "https://example.com/wiki"


def test_usage_db_path_default(monkeypatch):
    monkeypatch.delenv("USAGE_DB_PATH", raising=False)
    s = Settings()
    assert s.usage_db_path == "data/usage.sqlite"


def test_usage_db_path_from_env(monkeypatch):
    monkeypatch.setenv("USAGE_DB_PATH", "/tmp/test-usage.sqlite")
    s = Settings()
    assert s.usage_db_path == "/tmp/test-usage.sqlite"


def test_estimate_cost_nova_lite():
    usd = estimate_cost_usd("amazon.nova-lite-v1:0", input_tokens=1_000_000, output_tokens=1_000_000)
    pricing = MODEL_PRICING["amazon.nova-lite-v1:0"]
    expected = pricing.input_per_1m + pricing.output_per_1m
    assert usd == expected


def test_estimate_cost_unknown_model():
    assert estimate_cost_usd("unknown.model", 1000, 1000) == 0.0
