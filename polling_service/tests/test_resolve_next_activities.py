import sys
import types
import json
import pathlib
import importlib.util
from typing import Dict, Any, List, Optional, Tuple

from process_definition import load_process_definition
import pytest


# ------------------------------------------------------------
# Match local test style: stub heavy deps BEFORE importing module
# ------------------------------------------------------------
class _DummyPrompt:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs):
        return self.template


class _DummyPromptTemplate:
    @classmethod
    def from_template(cls, template: str):
        return _DummyPrompt(template)


class _DummySimpleJsonOutputParser:
    pass


class _DummyModel:
    async def astream(self, *_args, **_kwargs):
        if False:
            yield None
        return

    def invoke(self, *_args, **_kwargs):
        class R:
            content = ""

        return R()


def _install_stub_modules():
    # langchain.*
    langchain = types.ModuleType("langchain")
    langchain_prompts = types.ModuleType("langchain.prompts")
    langchain_prompts.PromptTemplate = _DummyPromptTemplate
    langchain_schema = types.ModuleType("langchain.schema")
    langchain_schema.Document = object
    langchain_output_parsers = types.ModuleType("langchain.output_parsers")
    langchain_output_parsers_json = types.ModuleType("langchain.output_parsers.json")
    langchain_output_parsers_json.SimpleJsonOutputParser = _DummySimpleJsonOutputParser

    sys.modules["langchain"] = langchain
    sys.modules["langchain.prompts"] = langchain_prompts
    sys.modules["langchain.schema"] = langchain_schema
    sys.modules["langchain.output_parsers"] = langchain_output_parsers
    sys.modules["langchain.output_parsers.json"] = langchain_output_parsers_json

    # llm_factory
    llm_factory = types.ModuleType("llm_factory")
    llm_factory.create_llm = lambda **_kwargs: _DummyModel()

    class _DummyEmbeddings:
        def embed_documents(self, texts):  # noqa: ANN001
            return [[0.0] * 4 for _ in texts]

        def embed_query(self, text):  # noqa: ANN001
            return [0.0] * 4

    llm_factory.create_embedding = lambda **_kwargs: _DummyEmbeddings()
    sys.modules["llm_factory"] = llm_factory

    # fastapi
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi

    # dotenv
    dotenv = types.ModuleType("dotenv")

    def load_dotenv(*_args, **_kwargs):
        return None

    dotenv.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv

    # mcp_processor
    mcp_processor_mod = types.ModuleType("mcp_processor")

    class _DummyMCP:
        async def execute_mcp_tools(self, *_args, **_kwargs):
            return {"messages": []}

        async def cleanup(self):
            return None

    mcp_processor_mod.mcp_processor = _DummyMCP()
    sys.modules["mcp_processor"] = mcp_processor_mod

    # code_executor
    code_executor_mod = types.ModuleType("code_executor")

    class _ExecResult:
        def __init__(self, rc=0, out="", err=""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def execute_python_code(*_args, **_kwargs):
        return _ExecResult(0, "ok", "")

    code_executor_mod.execute_python_code = execute_python_code
    sys.modules["code_executor"] = code_executor_mod

    # smtp_handler
    smtp_handler_mod = types.ModuleType("smtp_handler")

    def generate_email_template(*_args, **_kwargs):
        return "<html></html>"

    def send_email(*_args, **_kwargs):
        return None

    smtp_handler_mod.generate_email_template = generate_email_template
    smtp_handler_mod.send_email = send_email
    sys.modules["smtp_handler"] = smtp_handler_mod



def _load_modules():
    """Load needed modules from polling_service/ with stubbed deps."""
    _install_stub_modules()
    base = pathlib.Path(__file__).resolve().parents[1]

    def _load(name: str, rel: str):
        file_path = base / rel
        spec = importlib.util.spec_from_file_location(name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    wiproc = _load("wiproc", "workitem_processor.py")
    return wiproc


@pytest.fixture(scope="module")
def wiproc():
    return _load_modules()


# ------------------------------------------------------------
# Helpers for definitions and selection
# ------------------------------------------------------------
PROCESS_JSON_FILES = [
    "exclusiveExclusive.json",
    "exclusiveInclusive.json",
    "exclusiveParallel.json",
    "inclusiveExclusive.json",
    "inclusiveInclusive.json",
    "inclusiveParallel.json",
    "parallelExclusive.json",
    "parallelInclusive.json",
    "parallelParallel.json",
]


def _definitions_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _load_definition(filename: str):
    path = _definitions_dir() / filename
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return load_process_definition(data)


def _index_sequences(defn) -> tuple[dict[str, list], dict[str, list]]:
    out_by_src: dict[str, list] = {}
    in_by_tgt: dict[str, list] = {}
    for s in defn.sequences or []:
        out_by_src.setdefault(s.source, []).append(s)
        in_by_tgt.setdefault(s.target, []).append(s)
    return out_by_src, in_by_tgt


def _pick_activity_for_gateway_single_out(defn) -> Optional[str]:
    gateways_by_id = {getattr(g, "id", None): g for g in (defn.gateways or [])}
    out_by_src, _ = _index_sequences(defn)

    # Prefer activity -> exclusive gateway with single outgoing (merge)
    for act in defn.activities or []:
        outs = out_by_src.get(act.id, [])
        for seq in outs:
            gw = gateways_by_id.get(seq.target)
            if not gw:
                continue
            gw_type = (getattr(gw, "type", "") or "").lower()
            if "exclusive" in gw_type or gw_type in ("xor", "xorgateway"):
                gw_outs = out_by_src.get(gw.id, [])
                if len(gw_outs) == 1:
                    return act.id

    # Fallback: any activity with an outgoing
    for act in defn.activities or []:
        if out_by_src.get(act.id):
            return act.id
    return None


def _make_dummy_workitem() -> Dict[str, Any]:
    return {
        "id": "wi-1",
        "user_id": "tester@example.com",
        "proc_inst_id": "proc.inst",
        "proc_def_id": "proc.def",
        "activity_id": "",
        "tenant_id": "localhost",
        "assignees": [
            {"name": "고객", "endpoint": "customer@example.com"},
            {"name": "영업 담당자", "endpoint": "sales@example.com"},
            {"name": "영업팀장", "endpoint": "manager@example.com"},
        ],
        "output": {},
    }


@pytest.mark.parametrize("filename", PROCESS_JSON_FILES)
def test_resolve_next_activity_payloads_empty_conditions_treated_true(wiproc, filename: str):
    # Load definition JSON beside this test file
    defn = _load_definition(filename)

    # Pick suitable activity
    activity_id = _pick_activity_for_gateway_single_out(defn)
    assert activity_id, f"No suitable activity found in {filename}"

    # Build workitem bound to this definition
    workitem = _make_dummy_workitem()
    workitem["activity_id"] = activity_id
    workitem["proc_def_id"] = defn.processDefinitionId
    workitem["proc_inst_id"] = f"{defn.processDefinitionId}.test"

    # Empty sequence conditions -> treated as True
    sequence_condition_data: Dict[str, Any] = {}

    next_payloads = wiproc.resolve_next_activity_payloads(
        defn,
        activity_id,
        workitem,
        sequence_condition_data,
    )

    assert isinstance(next_payloads, list)
    assert len(next_payloads) > 0, (
        f"Expected non-empty next activities for {filename} with empty conditions treated as true"
    )

    for entry in next_payloads:
        assert entry.get("nextActivityId"), "missing nextActivityId"
        assert entry.get("type") in ("activity", "userTask", "event"), "invalid type"


