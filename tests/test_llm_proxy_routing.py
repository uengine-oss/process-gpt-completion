"""LLM 호출은 어떤 경로로도 프록시를 벗어나지 않아야 한다."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROXY = "http://proxy.test:4000"


def _load(path):
    spec = importlib.util.spec_from_file_location(f"factory_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[ROOT / "llm_factory.py", ROOT / "polling_service" / "llm_factory.py"], ids=["root", "polling"])
def factory(request, monkeypatch):
    monkeypatch.setenv("LLM_PROXY_URL", PROXY)
    monkeypatch.setenv("LLM_PROXY_API_KEY", "proxy-key")
    return _load(request.param)


@pytest.mark.parametrize("override", [
    {"base_url": "https://api.openai.com/v1", "api_key": "sk-direct"},
    {"openai_api_base": "https://api.openai.com/v1", "openai_api_key": "sk-direct"},
])
def test_create_llm_ignores_caller_routing(factory, override):
    llm = factory.create_llm(model="m", **override)
    assert llm.openai_api_base == PROXY
    assert llm.openai_api_key.get_secret_value() == "proxy-key"


def test_create_embedding_ignores_caller_routing(factory):
    embedding = factory.create_embedding(model="m", base_url="https://api.openai.com/v1", api_key="sk-direct")
    assert embedding.openai_api_base == PROXY
    assert embedding.openai_api_key.get_secret_value() == "proxy-key"


def test_generate_speech_goes_through_proxy(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_URL", PROXY)
    monkeypatch.setenv("LLM_PROXY_API_KEY", "proxy-key")
    monkeypatch.setenv("TTS_MODEL", "tts-alias")
    import process_var_sql_gen

    seen = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            seen.update(api_key=api_key, base_url=base_url)
            self.audio = types.SimpleNamespace(speech=types.SimpleNamespace(create=self._create))

        @staticmethod
        def _create(**kwargs):
            seen.update(model=kwargs["model"])
            return types.SimpleNamespace(content=b"mp3")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    assert process_var_sql_gen.generate_speech("안녕") == b"mp3"
    assert seen == {"api_key": "proxy-key", "base_url": PROXY, "model": "tts-alias"}
