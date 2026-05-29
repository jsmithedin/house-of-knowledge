import os
from app.config import Settings


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
