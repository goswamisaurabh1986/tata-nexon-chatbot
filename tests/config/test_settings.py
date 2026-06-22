import pytest
from pydantic import ValidationError


def test_settings_provide_domain_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from src.config.settings import Settings

    settings = Settings()

    assert settings.app.app_name == "Tata Nexon Chatbot"
    assert settings.llm.chat_model == "gpt-4o-mini"
    assert settings.llm.embedding_model == "text-embedding-3-small"
    assert settings.llm.embedding_batch_size == 32
    assert settings.retrieval.top_k == 5
    assert settings.ingestion.chunk_size == 1000
    assert settings.memory.sqlite_path == "chatbot_memory.db"


def test_settings_load_environment_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("AGENT_TOP_K", "3")
    monkeypatch.setenv("INGESTION_CHUNK_SIZE", "512")
    monkeypatch.setenv("CHATBOT_USER_ID", "krishiv")

    from src.config.settings import Settings

    settings = Settings()

    assert settings.llm.chat_model == "gpt-4.1-mini"
    assert settings.llm.embedding_model == "text-embedding-3-large"
    assert settings.retrieval.top_k == 3
    assert settings.ingestion.chunk_size == 512
    assert settings.memory.default_user_id == "krishiv"


def test_settings_are_immutable():
    from src.config.settings import Settings

    settings = Settings()

    with pytest.raises(ValidationError):
        settings.retrieval.top_k = 10


def test_api_key_is_redacted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    from src.config.llm import LLMSettings

    settings = LLMSettings()

    assert settings.api_key == "sk-test-secret"
    assert "sk-test-secret" not in repr(settings)
