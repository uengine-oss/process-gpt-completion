import json
from pathlib import Path

import pytest
import itertools
from process_definition import load_process_definition


# (filename, expected_split_type, expected_join_type)
GATEWAY_CASES = [
    ("inclusiveInclusive.json", "inclusiveGateway", "inclusiveGateway"),
    ("exclusiveExclusive.json", "exclusiveGateway", "exclusiveGateway"),
    ("parallelParallel.json",  "parallelGateway",  "parallelGateway"),
    ("exclusiveInclusive.json", "exclusiveGateway", "inclusiveGateway"),
    ("inclusiveExclusive.json", "inclusiveGateway", "exclusiveGateway"),
    ("parallelInclusive.json",  "parallelGateway",  "inclusiveGateway"),
    ("inclusiveParallel.json",  "inclusiveGateway", "parallelGateway"),
    ("exclusiveParallel.json",  "exclusiveGateway", "parallelGateway"),
    ("parallelExclusive.json",  "parallelGateway",  "exclusiveGateway"),
]


@pytest.mark.parametrize("filename,split_type,join_type", GATEWAY_CASES)
def test_load_gateway_combinations(filename: str, split_type: str, join_type: str):
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / filename
    assert json_path.exists(), f"Process JSON not found: {json_path}"

    with json_path.open("r", encoding="utf-8") as f:
        proc_def_dict = json.load(f)

    proc_def = load_process_definition(proc_def_dict)
    assert proc_def is not None

    split_gw = proc_def.find_gateway_by_id("Gateway_1x586s7")
    join_gw  = proc_def.find_gateway_by_id("Gateway_1bwgkit")

    assert split_gw is not None, "Split gateway not found: Gateway_1x586s7"
    assert join_gw  is not None, "Join gateway not found: Gateway_1bwgkit"

    assert getattr(split_gw, "type", None) == split_type
    assert getattr(join_gw,  "type", None) == join_type


# ------------------------------
# check_task_status tests
# ------------------------------
# We need to import workitem_processor with stubs for heavy deps
import sys, types, importlib.util

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

    llm_factory = types.ModuleType("llm_factory")
    llm_factory.create_llm = lambda **_kwargs: _DummyModel()

    class _DummyEmbeddings:
        def embed_documents(self, texts):  # noqa: ANN001
            return [[0.0] * 4 for _ in texts]

        def embed_query(self, text):  # noqa: ANN001
            return [0.0] * 4

    llm_factory.create_embedding = lambda **_kwargs: _DummyEmbeddings()
    sys.modules["llm_factory"] = llm_factory

    fastapi = types.ModuleType("fastapi")
    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail
    fastapi.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *_a, **_k: None
    sys.modules["dotenv"] = dotenv

    mcp_processor_mod = types.ModuleType("mcp_processor")
    class _DummyMCP:
        async def execute_mcp_tools(self, *_a, **_k):
            return {"messages": []}
        async def cleanup(self):
            return None
    mcp_processor_mod.mcp_processor = _DummyMCP()
    sys.modules["mcp_processor"] = mcp_processor_mod

    code_executor_mod = types.ModuleType("code_executor")
    class _ExecResult:
        def __init__(self, rc=0, out="ok", err=""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err
    code_executor_mod.execute_python_code = lambda *_a, **_k: _ExecResult(0, "ok", "")
    sys.modules["code_executor"] = code_executor_mod

    smtp_handler_mod = types.ModuleType("smtp_handler")
    smtp_handler_mod.generate_email_template = lambda *_a, **_k: "<html></html>"
    smtp_handler_mod.send_email = lambda *_a, **_k: None
    sys.modules["smtp_handler"] = smtp_handler_mod


def _load_wiproc_module():
    _install_stub_modules()
    file_path = Path(__file__).resolve().parents[1] / "workitem_processor.py"
    spec = importlib.util.spec_from_file_location("wiproc", str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="module")
def wiproc():
    return _load_wiproc_module()


def _load_proc(filename: str):
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / filename
    assert json_path.exists(), f"Process JSON not found: {json_path}"
    with json_path.open("r", encoding="utf-8") as f:
        return load_process_definition(json.load(f))




# Permutation coverage for ["DONE","IN_PROGRESS","PENDING"] across all join types
PERM_STATUSES_DIP = list(itertools.permutations(["DONE", "IN_PROGRESS", "PENDING"], 3))
PERM_STATUSES_DSC = list(itertools.permutations(["DONE", "SUBMITTED", "COMPLETED"], 3))
PERM_STATUSES_DPT = list(itertools.permutations(["DONE", "PENDING", "TODO"], 3))

JOIN_PARALLEL_FILES = [
    "parallelParallel.json",
    "inclusiveParallel.json",
    "exclusiveParallel.json",
]
JOIN_INCLUSIVE_FILES = [
    "inclusiveInclusive.json",
    "exclusiveInclusive.json",
    "parallelInclusive.json",
]
JOIN_EXCLUSIVE_FILES = [
    "exclusiveExclusive.json",
    "inclusiveExclusive.json",
    "parallelExclusive.json",
]

CASES_PERM = []
# Cases with an IN_PROGRESS present: parallel=0, inclusive=0, exclusive=1
for sts in PERM_STATUSES_DIP:
    sts_list = list(sts)
    for f in JOIN_PARALLEL_FILES:
        CASES_PERM.append((f, "parallelGateway", sts_list, 0))
    for f in JOIN_INCLUSIVE_FILES:
        CASES_PERM.append((f, "inclusiveGateway", sts_list, 0))
    for f in JOIN_EXCLUSIVE_FILES:
        CASES_PERM.append((f, "exclusiveGateway", sts_list, 1))

# Cases all in DONE states (no IN_PROGRESS): parallel=1, inclusive=1, exclusive=1
for sts in PERM_STATUSES_DSC:
    sts_list = list(sts)
    for f in JOIN_PARALLEL_FILES:
        CASES_PERM.append((f, "parallelGateway", sts_list, 1))
    for f in JOIN_INCLUSIVE_FILES:
        CASES_PERM.append((f, "inclusiveGateway", sts_list, 1))
    for f in JOIN_EXCLUSIVE_FILES:
        CASES_PERM.append((f, "exclusiveGateway", sts_list, 1))

# Cases with TODO/PENDING and one DONE (no IN_PROGRESS): parallel=0, inclusive=1, exclusive=1
for sts in PERM_STATUSES_DPT:
    sts_list = list(sts)
    for f in JOIN_PARALLEL_FILES:
        CASES_PERM.append((f, "parallelGateway", sts_list, 0))
    for f in JOIN_INCLUSIVE_FILES:
        CASES_PERM.append((f, "inclusiveGateway", sts_list, 1))
    for f in JOIN_EXCLUSIVE_FILES:
        CASES_PERM.append((f, "exclusiveGateway", sts_list, 1))

# Explicit case: ["TODO","SUBMITTED","TODO"]
_extra_case = ["TODO", "SUBMITTED", "TODO"]
for f in JOIN_PARALLEL_FILES:
    CASES_PERM.append((f, "parallelGateway", _extra_case, 0))
for f in JOIN_INCLUSIVE_FILES:
    CASES_PERM.append((f, "inclusiveGateway", _extra_case, 1))
for f in JOIN_EXCLUSIVE_FILES:
    CASES_PERM.append((f, "exclusiveGateway", _extra_case, 1))

# Explicit case: ["TODO","IN_PROGRESS","TODO"]
_extra_case2 = ["TODO", "IN_PROGRESS", "TODO"]
for f in JOIN_PARALLEL_FILES:
    CASES_PERM.append((f, "parallelGateway", _extra_case2, 0))
for f in JOIN_INCLUSIVE_FILES:
    CASES_PERM.append((f, "inclusiveGateway", _extra_case2, 0))
for f in JOIN_EXCLUSIVE_FILES:
    CASES_PERM.append((f, "exclusiveGateway", _extra_case2, 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,join_type,branch_statuses,expected_len", CASES_PERM)
async def test_check_task_status_join_behavior_permutations(wiproc, filename, join_type, branch_statuses, expected_len):
    proc_def = _load_proc(filename)

    join_gw = proc_def.find_gateway_by_id("Gateway_1bwgkit")
    assert getattr(join_gw, "type", None) == join_type

    current_activity_id = "Activity_0wmbn0q"
    target_after_join = "Activity_161as8h"

    next_activity_payloads = [
        {"nextActivityId": target_after_join, "type": "userTask", "result": "IN_PROGRESS"}
    ]

    branches = [
        {"activity_id": "Activity_0wmbn0q", "status": branch_statuses[0]},
        {"activity_id": "Activity_1l2ci7f", "status": branch_statuses[1]},
        {"activity_id": "Activity_0rwrgae", "status": branch_statuses[2]},
    ]

    chain_input_next = {
        "activity_id": current_activity_id,
        "sequences": proc_def.sequences,
        "gateways": proc_def.gateways,
        "branch_merged_workitems": branches,
    }

    filtered = await wiproc.check_task_status(next_activity_payloads, chain_input_next)
    assert isinstance(filtered, list)
    assert len(filtered) == expected_len
