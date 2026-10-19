"""process_engine 은 LLM 없이 임포트되고 라우트를 등록해야 한다.

모듈 수준에서 체인을 만들며 전역 `model` 을 참조하면 임포트가 NameError 로 실패하고,
main.py 의 선택적 등록이 이를 조용히 건너뛰어 /complete · /handover-suggest 등이
통째로 404 가 된다. LLM 은 요청 시점에 get_model() 로 만든다.
"""
from fastapi import FastAPI

import process_engine


def test_import_does_not_build_the_model():
    assert process_engine._model is None


def test_routes_are_registered():
    app = FastAPI()
    process_engine.add_routes_to_app(app)
    paths = {r.path for r in app.routes}
    assert "/complete" in paths
    assert "/handover-suggest" in paths


def test_handover_suggest_chain_is_built_lazily(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(process_engine, "get_model", lambda: RunnableLambda(lambda _: "```json\n{\"doneLabel\": \"a\"}\n```"))
    assert process_engine.handover_suggest_chain().invoke(
        {"artifacts": "[]", "recentMessages": "[]", "organizationChart": ""}
    ) == {"doneLabel": "a"}
