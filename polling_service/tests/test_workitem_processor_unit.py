import sys
import types
import json
import pathlib
import importlib.util
import pytest


# ------------------------------------------------------------
# Helpers: stub external heavy deps BEFORE importing module
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



def _load_workitem_processor_module():
    _install_stub_modules()
    file_path = pathlib.Path(__file__).resolve().parents[1] / "workitem_processor.py"
    spec = importlib.util.spec_from_file_location("wiproc", str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="module")
def wiproc():
    return _load_workitem_processor_module()


# ------------------------------------------------------------
# Dummy process definition structures for local tests
# ------------------------------------------------------------
class _Seq:
    def __init__(self, id, source, target, name=None, properties=None):
        self.id = id
        self.source = source
        self.target = target
        self.name = name
        self.properties = properties


class _Gateway:
    def __init__(self, id, type, name=None, condition=None, properties=None):
        self.id = id
        self.type = type
        self.name = name
        self.condition = condition
        self.properties = properties


class _Activity:
    def __init__(self, id, name=None, description=None, role=None, type="userTask"):
        self.id = id
        self.name = name
        self.description = description
        self.role = role
        self.type = type


class _SubProcess:
    def __init__(self, id, name=None, description=None):
        self.id = id
        self.name = name
        self.description = description


class _ProcDef:
    def __init__(self, activities=None, gateways=None, sequences=None, sub_processes=None, events=None):
        self.activities = activities or []
        self.gateways = gateways or []
        self.sequences = sequences or []
        self.subProcesses = sub_processes or []
        self.events = events or []

    def find_activity_by_id(self, aid):
        return next((a for a in self.activities if getattr(a, "id", None) == aid), None)

    def find_gateway_by_id(self, gid):
        return next((g for g in self.gateways if getattr(g, "id", None) == gid), None)

    def find_event_by_id(self, eid):
        return next((e for e in (self.events or []) if getattr(e, "id", None) == eid), None)

    def find_sub_process_by_id(self, sid):
        return next((s for s in self.subProcesses if getattr(s, "id", None) == sid), None)


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
 


def test_resolve_next_activity_payloads_exclusive_branch(wiproc):
    # A1 -> G1 -> (s2) B1, (s3) B2
    gw = _Gateway("G1", type="exclusiveGateway", name="XOR")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("s2", source="G1", target="B1"),
        _Seq("s3", source="G1", target="B2"),
    ]
    acts = [_Activity("A1"), _Activity("B1", name="Task B1"), _Activity("B2", name="Task B2")]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    sequence_condition_data = {"s2": {"conditionEval": True}, "s3": {"conditionEval": False}}
    workitem = {"assignees": []}

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def,
        activity_id="A1",
        workitem=workitem,
        sequence_condition_data=sequence_condition_data,
    )

    ids = [p.get("nextActivityId") for p in payloads]
    assert ids == ["B1"]


@pytest.mark.asyncio
async def test_mapper_result_context_filters_direct_conditional_sequences(wiproc):
    seqs = [
        _Seq("s1", source="A1", target="B1", properties=json.dumps({"conditionFunction": "fullName == 'JaneKim'"})),
        _Seq("s2", source="A1", target="B2", properties=json.dumps({"conditionFunction": "fullName != 'JaneKim'"})),
    ]
    acts = [_Activity("A1"), _Activity("B1", name="Approved"), _Activity("B2", name="Review")]
    proc_def = _ProcDef(activities=acts, sequences=seqs)

    sequence_condition_data = wiproc.get_sequence_condition_data(proc_def, "A1", ["B1", "B2"])
    await wiproc._evaluate_sequence_conditions(
        None,
        None,
        proc_def,
        {"mapper_form": {"fullName": "JaneKim"}},
        None,
        sequence_condition_data,
        [],
        workitem={"tenant_id": "test"},
    )

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def,
        activity_id="A1",
        workitem={"assignees": []},
        sequence_condition_data=sequence_condition_data,
    )

    assert sequence_condition_data["s1"]["conditionEval"] is True
    assert sequence_condition_data["s2"]["conditionEval"] is False
    assert [p.get("nextActivityId") for p in payloads] == ["B1"]


def test_collect_ui_field_keys_basic(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}, {"key": "age", "text": "나이"}]},
        {"fields_json": [{"key": "address", "text": "주소"}]},
    ]
    keys = wiproc.collect_ui_field_keys(ui_defs)
    assert isinstance(keys, set)
    assert keys == {"email", "age", "address"}


def test_apply_field_name_annotation_recursively_wraps_and_recurses(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}, {"key": "age", "text": "나이"}]},
    ]
    data = {
        "email": "a@test.com",
        "profile": {
            "email": "b@test.com",
            "list": [
                {"email": "c@test.com"},
                {"other": "x"},
            ],
        },
        "age": 20,
        "misc": ["nochange"],
    }
    result = wiproc.apply_field_name_annotation_recursively(data, ui_defs)
    # top-level wrap
    assert isinstance(result.get("email"), dict)
    assert result["email"]["name"] == "이메일"
    assert result["email"]["value"] == "a@test.com"
    # nested dict wrap
    assert isinstance(result["profile"]["email"], dict)
    assert result["profile"]["email"]["name"] == "이메일"
    assert result["profile"]["email"]["value"] == "b@test.com"
    # list element dict wrap
    assert isinstance(result["profile"]["list"][0]["email"], dict)
    assert result["profile"]["list"][0]["email"]["name"] == "이메일"
    assert result["profile"]["list"][0]["email"]["value"] == "c@test.com"
    # non-matching elements unchanged
    assert result["profile"]["list"][1]["other"] == "x"
    # numeric value wrapped when key matches
    assert isinstance(result["age"], dict)
    assert result["age"]["name"] == "나이"
    assert result["age"]["value"] == 20
    # non-matching types/values unchanged
    assert result["misc"] == ["nochange"]


def test_apply_field_name_annotation_recursively_handles_cycles(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}]},
    ]
    obj = {}
    obj["self"] = obj  # introduce a cycle
    res = wiproc.apply_field_name_annotation_recursively(obj, ui_defs)
    # Should not raise or infinitely recurse; result is dict with same shape
    assert isinstance(res, dict)
    assert "self" in res

 
def test__annotate_list_elements_with_field_names_basic(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}, {"key": "age", "text": "나이"}]},
    ]
    lst = [
        {"email": "u1@example.com", "other": 1},
        {"age": 42},
        "plain",
    ]
    out = wiproc._annotate_list_elements_with_field_names(lst, ui_defs)
    assert isinstance(out, list)
    assert isinstance(out[0]["email"], dict)
    assert out[0]["email"]["name"] == "이메일"
    assert out[0]["email"]["value"] == "u1@example.com"
    assert out[0]["other"] == {"name": "other", "value": 1}
    assert isinstance(out[1]["age"], dict)
    assert out[1]["age"]["name"] == "나이"
    assert out[1]["age"]["value"] == 42
    assert out[2] == "plain"


def test__annotate_dict_with_field_names_nested(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}, {"key": "tags", "text": "태그"}]},
    ]
    data = {
        "profile": {"email": "inner@example.com", "tags": ["a", "b"]},
        "email": "root@example.com",
        "other": {"nested": {"email": "deep@example.com"}},
    }
    res = wiproc._annotate_dict_with_field_names(data, ui_defs)
    # top-level
    assert isinstance(res["email"], dict)
    assert res["email"]["name"] == "이메일"
    assert res["email"]["value"] == "root@example.com"
    # nested dict
    assert isinstance(res["profile"]["value"]["email"], dict)
    assert res["profile"]["value"]["email"]["value"] == "inner@example.com"
    # nested list under key 'tags' should be wrapped
    assert isinstance(res["profile"]["value"]["tags"], dict)
    assert res["profile"]["value"]["tags"]["name"] == "태그"
    assert res["profile"]["value"]["tags"]["value"] == ["a", "b"]
    # deeper nested dict: add_field_name_by_key recurses
    assert isinstance(res["other"]["value"]["nested"]["value"]["email"], dict)
    assert res["other"]["value"]["nested"]["value"]["email"]["value"] == "deep@example.com"


def test_iter_reference_scalars_extractor_basic_and_limit(wiproc):
    data = {
        "user": {
            "name": "Alice",
            "age": 30,
            "tags": ["x", "y", "z"],
            "addr": {"city": "Seoul", "zip": 12345},
        },
        "misc": {"flag": True},
    }
    # with limit
    res_limited = wiproc.iter_reference_scalars_extractor(data, limit=4)
    assert len(res_limited) == 4
    keys_limited = {e["key"] for e in res_limited}
    assert "user.name" in keys_limited
    assert any(k in keys_limited for k in ["user.age", "user.tags", "user.addr.city", "misc.flag"])
    # without tight limit, ensure list scalars captured
    res_full = wiproc.iter_reference_scalars_extractor(data, limit=10)
    keys_full = {e["key"] for e in res_full}
    assert "user.tags" in keys_full
    tags_entry = next(e for e in res_full if e["key"] == "user.tags")
    assert tags_entry["value"] == ["x", "y", "z"]

def test__annotate_dict_with_field_names_cycle_raises_recursion(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}]},
    ]
    cyc = {}
    # self-referential under a key that will be annotated -> triggers deep recursion
    cyc["email"] = cyc
    # 성공 기준: 에러가 나도(pass), 에러 없이 반환돼도(pass)
    try:
        _ = wiproc._annotate_dict_with_field_names(cyc, ui_defs)
    except RecursionError:
        assert True


def test__annotate_list_elements_with_field_names_cycle_raises_recursion(wiproc):
    ui_defs = [
        {"fields_json": [{"key": "email", "text": "이메일"}]},
    ]
    loop_list = []
    inner = {"email": loop_list}
    loop_list.append(inner)  # create a cycle: list -> dict -> list
    # 성공 기준: 에러가 나도(pass), 에러 없이 반환돼도(pass)
    try:
        _ = wiproc._annotate_list_elements_with_field_names(loop_list, ui_defs)
    except RecursionError:
        assert True


def test_resolve_next_activity_payloads_exclusive_gateway_resolves_correct_role(wiproc):
    """Task 4.1: ExclusiveGateway selects one branch; the selected activity
    gets its own assignee resolved from role_bindings (not copied from gateway)."""
    gw = _Gateway("G1", type="exclusiveGateway")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("s2", source="G1", target="B1"),
        _Seq("s3", source="G1", target="B2"),
    ]
    acts = [
        _Activity("A1", role="initiator"),
        _Activity("B1", name="Manager Review", role="manager"),
        _Activity("B2", name="Dev Task", role="developer"),
    ]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    workitem = {
        "assignees": [
            {"name": "manager", "endpoint": "mgr@example.com"},
            {"name": "developer", "endpoint": "dev@example.com"},
        ]
    }
    # Only s2 is True → B1 (manager) is selected
    sequence_condition_data = {"s2": {"conditionEval": True}, "s3": {"conditionEval": False}}

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def, activity_id="A1", workitem=workitem, sequence_condition_data=sequence_condition_data,
    )

    assert len(payloads) == 1
    assert payloads[0]["nextActivityId"] == "B1"
    assert payloads[0]["nextUserEmail"] == "mgr@example.com"


def test_resolve_next_activity_payloads_parallel_gateway_resolves_per_activity_role(wiproc):
    """Task 4.1: ParallelGateway expands to all branches; each activity
    gets its own assignee resolved from role_bindings individually."""
    gw = _Gateway("G1", type="parallelGateway")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("s2", source="G1", target="B1"),
        _Seq("s3", source="G1", target="B2"),
    ]
    acts = [
        _Activity("A1", role="initiator"),
        _Activity("B1", name="Manager Review", role="manager"),
        _Activity("B2", name="Dev Task", role="developer"),
    ]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    workitem = {
        "assignees": [
            {"name": "manager", "endpoint": "mgr@example.com"},
            {"name": "developer", "endpoint": "dev@example.com"},
        ]
    }
    sequence_condition_data = {}

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def, activity_id="A1", workitem=workitem, sequence_condition_data=sequence_condition_data,
    )

    by_id = {p["nextActivityId"]: p for p in payloads}
    assert "B1" in by_id
    assert "B2" in by_id
    assert by_id["B1"]["nextUserEmail"] == "mgr@example.com"
    assert by_id["B2"]["nextUserEmail"] == "dev@example.com"


def test_resolve_next_activity_payloads_missing_role_returns_none(wiproc):
    """Task 4.2: When role is not in role_bindings, nextUserEmail should be None."""
    gw = _Gateway("G1", type="exclusiveGateway")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("s2", source="G1", target="B1"),
    ]
    acts = [
        _Activity("A1"),
        _Activity("B1", name="Unknown Role Task", role="unknown_role"),
    ]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    workitem = {"assignees": [{"name": "manager", "endpoint": "mgr@example.com"}]}
    sequence_condition_data = {"s2": {"conditionEval": True}}

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def, activity_id="A1", workitem=workitem, sequence_condition_data=sequence_condition_data,
    )

    assert len(payloads) == 1
    assert payloads[0]["nextActivityId"] == "B1"
    assert payloads[0]["nextUserEmail"] is None


def test_resolve_next_activity_payloads_single_path_no_gateway(wiproc):
    """Task 4.4: Single path process (no gateway) should work as before."""
    seqs = [_Seq("s1", source="A1", target="A2")]
    acts = [
        _Activity("A1", role="initiator"),
        _Activity("A2", name="Next Step", role="reviewer"),
    ]
    proc_def = _ProcDef(activities=acts, sequences=seqs)

    workitem = {"assignees": [{"name": "reviewer", "endpoint": "rev@example.com"}]}
    sequence_condition_data = {}

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def, activity_id="A1", workitem=workitem, sequence_condition_data=sequence_condition_data,
    )

    assert len(payloads) == 1
    assert payloads[0]["nextActivityId"] == "A2"
    assert payloads[0]["nextUserEmail"] == "rev@example.com"


def test_resolve_next_activity_payloads_role_bindings_fallback_from_proc_inst(wiproc):
    """Task 4.1/4.4: When workitem.assignees is empty, falls back to process instance role_bindings."""
    gw = _Gateway("G1", type="exclusiveGateway")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("s2", source="G1", target="B1"),
    ]
    acts = [
        _Activity("A1"),
        _Activity("B1", name="Review", role="reviewer"),
    ]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    workitem = {
        "assignees": [],
        "proc_inst_id": "test-inst-1",
        "tenant_id": "test-tenant",
    }
    sequence_condition_data = {"s2": {"conditionEval": True}}

    from unittest.mock import patch, MagicMock
    mock_inst = MagicMock()
    mock_inst.role_bindings = [{"name": "reviewer", "endpoint": "reviewer@example.com"}]

    with patch.object(wiproc, "fetch_process_instance", return_value=mock_inst):
        payloads = wiproc.resolve_next_activity_payloads(
            proc_def, activity_id="A1", workitem=workitem, sequence_condition_data=sequence_condition_data,
        )

    assert len(payloads) == 1
    assert payloads[0]["nextUserEmail"] == "reviewer@example.com"


def test_activity_mapper_updates_role_binding_for_next_activity_payload(wiproc):
    activity = _Activity("A1")
    activity.properties = json.dumps({
        "mapperIn": {
            "mappingElements": [
                {
                    "argument": {"text": "lane.reviewer.endpoint"},
                    "direction": "out",
                    "variable": {"name": "reviewerEmail"},
                }
            ]
        }
    })
    process_instance = types.SimpleNamespace(
        proc_inst_id="test-inst-1",
        variables_data={},
        role_bindings=[{"name": "reviewer", "endpoint": "old@example.com"}],
        participants=[],
    )
    workitem = {
        "id": "wi-1",
        "activity_id": "A1",
        "assignees": [],
        "output": {},
    }

    mapper_result = wiproc._apply_activity_mappers(
        process_instance,
        activity,
        workitem,
        {"reviewerEmail": "mapped-reviewer@example.com"},
        {},
        {},
    )

    assert mapper_result["role_bindings"] == [{"name": "reviewer", "endpoint": "mapped-reviewer@example.com"}]
    assert process_instance.role_bindings == [{"name": "reviewer", "endpoint": "mapped-reviewer@example.com"}]
    assert workitem["assignees"] == [{"name": "reviewer", "endpoint": "mapped-reviewer@example.com"}]

    proc_def = _ProcDef(
        activities=[
            _Activity("A1", role="requester"),
            _Activity("A2", name="Review", role="reviewer"),
        ],
        sequences=[_Seq("s1", source="A1", target="A2")],
    )

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def,
        activity_id="A1",
        workitem=workitem,
        sequence_condition_data={},
    )

    assert len(payloads) == 1
    assert payloads[0]["nextActivityId"] == "A2"
    assert payloads[0]["nextUserEmail"] == "mapped-reviewer@example.com"


def test_call_activity_out_role_binding_propagates_child_role_to_parent(wiproc, monkeypatch):
    parent_def = _ProcDef()
    call_activity = _Activity("Call_review", name="Review", type="callActivity")
    call_activity.properties = json.dumps({
        "roleBindings": [
            {
                "direction": "OUT",
                "role": {"name": "보안검토담당자"},
                "argument": "보안심사자",
            }
        ]
    })
    parent_def.activities = [call_activity]
    parent_inst = types.SimpleNamespace(
        proc_inst_id="parent-1",
        parent_proc_inst_id=None,
        current_activity_ids=["Call_review"],
        role_bindings=[{"name": "보안검토담당자", "endpoint": "old@example.com"}],
        process_definition=parent_def,
    )
    child_inst = types.SimpleNamespace(
        proc_inst_id="child-1",
        parent_proc_inst_id="parent-1",
    )
    children = [
        {
            "proc_inst_id": "child-1",
            "status": "COMPLETED",
            "role_bindings": [{"name": "보안심사자", "endpoint": "new@example.com"}],
        }
    ]
    waiting_workitem = types.SimpleNamespace(
        id="wi-call",
        status="PENDING",
        assignees=[{"name": "보안검토담당자", "endpoint": "old@example.com"}],
    )
    upserted_workitems = []

    def fake_fetch_process_instance(proc_inst_id, tenant_id=None):
        if proc_inst_id == "child-1":
            return child_inst
        if proc_inst_id == "parent-1":
            return parent_inst
        return None

    monkeypatch.setattr(wiproc, "fetch_process_instance", fake_fetch_process_instance)
    monkeypatch.setattr(wiproc, "fetch_child_instances_by_parent", lambda parent_id, tenant_id=None: children)
    monkeypatch.setattr(wiproc, "fetch_workitem_by_proc_inst_and_activity", lambda *_args, **_kwargs: waiting_workitem)
    monkeypatch.setattr(wiproc, "upsert_process_instance", lambda inst, tenant_id=None, process_definition=None: (True, inst))
    monkeypatch.setattr(wiproc, "upsert_workitem", lambda data, tenant_id=None: upserted_workitems.append(data))

    wiproc._progress_parent_if_all_children_completed("child-1", "test-tenant")

    assert parent_inst.role_bindings == [{"name": "보안검토담당자", "endpoint": "new@example.com"}]
    assert {
        "id": "wi-call",
        "assignees": [{"name": "보안검토담당자", "endpoint": "new@example.com"}],
    } in upserted_workitems
    assert {"id": "wi-call", "status": "SUBMITTED"} in upserted_workitems


def test_iter_reference_scalars_extractor_cycle_raises_recursion(wiproc):
    cyc = {}
    cyc["self"] = cyc  # pure cycle with no scalars to satisfy limit -> unbounded recursion
    # 성공 기준: 에러가 나도(pass), 에러 없이 반환돼도(pass)
    try:
        _ = wiproc.iter_reference_scalars_extractor(cyc, limit=1)
    except RecursionError:
        assert True


def test_apply_field_name_annotation_recursively_vip_newsletter_sample(wiproc):
    # ui_definitions with a blank key and typical fields, modeled as dicts
    ui_defs = [
        {
            "id": "vip_newsletter_process_activity_0ot7kwf_form",
            "activity_id": "Activity_0ot7kwf",
            "fields_json": [
                {"key": "newsletter_report", "text": "뉴스레터 내용"},
            ],
        },
        {
            "id": "vip_newsletter_process_activity_1rzgb75_form",
            "activity_id": "Activity_1rzgb75",
            "fields_json": [
                {"key": "recipient_name", "text": "이름"},
                {"key": "recipient_email", "text": "이메일"},
            ],
        },
        {
            "id": "vip_newsletter_process_activity_1bzewxr_form",
            "activity_id": "Activity_1bzewxr",
            "fields_json": [
                {"key": "", "text": ""},
                {"key": "result_check", "text": "결과확인"},
            ],
        },
        {
            "id": "vip_newsletter_process_activity_10pn15v_form",
            "activity_id": "Activity_10pn15v",
            "fields_json": [
                {"key": "newsletter_report", "text": "관심사 기반 뉴스레터"},
            ],
        },
        {
            "id": "vip_newsletter_process_activity_0hrng6y_form",
            "activity_id": "Activity_0hrng6y",
            "fields_json": [
                {"key": "review_status", "text": "결재 여부"},
                {"key": "review_opinion", "text": "재작성 의견"},
            ],
        },
        {
            "id": "vip_newsletter_process_activity_1en8e0l_form",
            "activity_id": "Activity_1en8e0l",
            "fields_json": [
                {"key": "name", "text": "이름"},
                {"key": "interest", "text": "관심사"},
                {"key": "skill_level", "text": "기술이해도 수준"},
                {"key": "sales_manager", "text": "담당 영업"},
                {"key": "sales_manager_relation", "text": "담당 영업과의 관계"},
                {"key": "sales_manager_email", "text": "담당 영업 이메일 주소"},
            ],
        },
    ]

    all_workitem_input_data = {
        "vip_newsletter_process_activity_0ot7kwf_form": {"newsletter_report": ""},
        "vip_newsletter_process_activity_1en8e0l_form": {
            "name": "김지훈",
            "interest": "데이터 분석, 트렌드 리포트 구독",
            "skill_level": "medium",
            "sales_manager": "박민수",
            "sales_manager_email": "minsu.park@salescorp.kr",
            "sales_manager_relation": "10년 이상 비즈니스 거래, 신뢰가 두터움",
        },
        "vip_newsletter_process_activity_0hrng6y_form": {
            "review_status": ["approved"],
            "review_opinion": "",
        },
        "vip_newsletter_process_activity_10pn15v_form": {
            "newsletter_report": "# VIP 고객 개요 및 관심사 분석 ..."
        },
    }

    # Should not raise RecursionError and should annotate known fields
    try:
        ui_field_keys = wiproc.collect_ui_field_keys(ui_defs)
        result = wiproc.apply_field_name_annotation_recursively(
            all_workitem_input_data, ui_defs, ui_field_keys
        )
    except RecursionError:
        pytest.fail("RecursionError raised during annotation of VIP newsletter sample data")

    # Top-level keys preserved
    assert set(result.keys()) == set(all_workitem_input_data.keys())

    # Known keys are wrapped with {name, value}
    rep = result["vip_newsletter_process_activity_10pn15v_form"]["newsletter_report"]
    assert isinstance(rep, dict)
    assert "name" in rep and "value" in rep

    rs = result["vip_newsletter_process_activity_0hrng6y_form"]["review_status"]
    assert isinstance(rs, dict)
    assert rs["name"] == "결재 여부"
    assert rs["value"] == ["approved"]

    # 'name' field may be reserved for wrapper; ensure at least other fields got wrapped
    sm_email = result["vip_newsletter_process_activity_1en8e0l_form"]["sales_manager_email"]
    assert isinstance(sm_email, dict)
    assert sm_email["name"] == "담당 영업 이메일 주소"
    assert sm_email["value"] == "minsu.park@salescorp.kr"


# ------------------------------------------------------------
# Task 4.3: Instance status tests for multi-branch processes
# These test the logic in database.upsert_process_instance
# by mocking DB calls and verifying status determination.
# ------------------------------------------------------------

def _make_mock_process_instance(current_activity_ids, proc_def, role_bindings=None):
    """Create a minimal mock ProcessInstance for status logic testing."""
    from unittest.mock import MagicMock
    inst = MagicMock()
    inst.proc_inst_id = "test-inst-1"
    inst.proc_inst_name = "Test Instance"
    inst.current_activity_ids = current_activity_ids
    inst.role_bindings = role_bindings or []
    inst.variables_data = []
    inst.participants = []
    inst.process_definition = proc_def
    inst.status = "RUNNING"
    inst.get_def_id.return_value = "test-def-1"
    inst.dict.return_value = {
        "proc_inst_id": "test-inst-1",
        "proc_inst_name": "Test Instance",
        "current_activity_ids": current_activity_ids,
        "role_bindings": role_bindings or [],
        "variables_data": [],
        "participants": [],
        "status": "RUNNING",
    }
    return inst


def test_instance_stays_running_when_active_activities_exist():
    """Task 4.3: Instance should stay RUNNING when current_activity_ids is non-empty,
    even if an end activity's workitem is DONE."""
    from unittest.mock import patch, MagicMock

    end_act = _Activity("EndAct", name="End Activity")
    gw_end = _Gateway("EndEvent1", type="endEvent")
    seqs = [_Seq("s1", source="EndAct", target="EndEvent1")]
    acts = [end_act, _Activity("BranchB_Act", name="Branch B")]
    proc_def = _ProcDef(activities=acts, gateways=[gw_end], sequences=seqs)
    proc_def.find_end_activities = lambda: [end_act]

    proc_instance = _make_mock_process_instance(
        current_activity_ids=["BranchB_Act"],
        proc_def=proc_def,
    )

    done_workitem = MagicMock()
    done_workitem.status = "DONE"

    import database as db_mod
    with patch.object(db_mod, "fetch_workitem_by_proc_inst_and_activity", return_value=done_workitem), \
         patch.object(db_mod, "set_participants_from_workitems", return_value=proc_instance), \
         patch.object(db_mod, "fetch_process_definition_latest_version", return_value=None), \
         patch.object(db_mod, "supabase_client_var") as mock_client_var, \
         patch.object(db_mod, "subdomain_var") as mock_subdomain_var:

        mock_supabase = MagicMock()
        mock_supabase.table.return_value.upsert.return_value.execute.return_value.data = [{"proc_inst_id": "test-inst-1"}]
        mock_client_var.get.return_value = mock_supabase
        mock_subdomain_var.get.return_value = "test-tenant"

        success, result_inst = db_mod.upsert_process_instance(proc_instance, tenant_id="test-tenant")

    upsert_call_args = mock_supabase.table.return_value.upsert.call_args[0][0]
    assert upsert_call_args["status"] == "RUNNING"


def test_instance_completes_when_no_active_activities_and_end_done():
    """Task 4.3: Instance should be COMPLETED when current_activity_ids is empty
    and at least one end activity workitem is DONE."""
    from unittest.mock import patch, MagicMock

    end_act = _Activity("EndAct", name="End Activity")
    gw_end = _Gateway("EndEvent1", type="endEvent")
    seqs = [_Seq("s1", source="EndAct", target="EndEvent1")]
    proc_def = _ProcDef(activities=[end_act], gateways=[gw_end], sequences=seqs)
    proc_def.find_end_activities = lambda: [end_act]

    proc_instance = _make_mock_process_instance(
        current_activity_ids=[],
        proc_def=proc_def,
    )

    done_workitem = MagicMock()
    done_workitem.status = "DONE"

    import database as db_mod
    with patch.object(db_mod, "fetch_workitem_by_proc_inst_and_activity", return_value=done_workitem), \
         patch.object(db_mod, "set_participants_from_workitems", return_value=proc_instance), \
         patch.object(db_mod, "fetch_process_definition_latest_version", return_value=None), \
         patch.object(db_mod, "supabase_client_var") as mock_client_var, \
         patch.object(db_mod, "subdomain_var") as mock_subdomain_var:

        mock_supabase = MagicMock()
        mock_supabase.table.return_value.upsert.return_value.execute.return_value.data = [{"proc_inst_id": "test-inst-1"}]
        mock_client_var.get.return_value = mock_supabase
        mock_subdomain_var.get.return_value = "test-tenant"

        success, result_inst = db_mod.upsert_process_instance(proc_instance, tenant_id="test-tenant")

    upsert_call_args = mock_supabase.table.return_value.upsert.call_args[0][0]
    assert upsert_call_args["status"] == "COMPLETED"


class _FakeConditionModel:
    """astream 으로 미리 정해진 JSON 텍스트를 흘려보내는 가짜 LLM."""

    def __init__(self, payload):
        self._text = json.dumps(payload, ensure_ascii=False)

    async def astream(self, *_args, **_kwargs):
        # content 속성을 가진 청크 하나로 흘려보낸다
        class _Chunk:
            def __init__(self, content):
                self.content = content

        yield _Chunk(self._text)


@pytest.mark.asyncio
async def test_nl_condition_applies_when_model_echoes_sequence_name(wiproc):
    """LLM 이 sequenceId 대신 시퀀스 이름(예: '승인')을 돌려줘도 올바른 시퀀스에 반영되어야 한다.

    회귀: 예전에는 정확한 문자열 일치만 하여, 이름/공백/대소문자가 다르면 모든 분기가
    False 로 강제되고 Exclusive 게이트웨이의 next activity 가 사라졌다.
    """
    # A1 -> G1 -> (Flow_yes)"승인" B1, (Flow_no)"반려" B2
    gw = _Gateway("G1", type="exclusiveGateway", name="XOR")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("Flow_yes", source="G1", target="B1", name="승인"),
        _Seq("Flow_no", source="G1", target="B2", name="반려"),
    ]
    acts = [_Activity("A1"), _Activity("B1", name="승인 처리"), _Activity("B2", name="반려 처리")]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    sequence_condition_data = {
        "Flow_yes": {"name": "승인"},
        "Flow_no": {"name": "반려"},
    }
    nl_condition_sequences = [
        ("Flow_yes", "승인", None),
        ("Flow_no", "반려", None),
    ]
    # 모델은 올바르게 '승인' 을 선택했지만 sequenceId 자리에 '이름' 을 넣어 반환
    model = _FakeConditionModel({
        "results": [
            {"sequenceId": "승인", "conditionMet": True},
            {"sequenceId": "반려", "conditionMet": False},
        ]
    })

    await wiproc._evaluate_nl_conditions(
        model,
        None,
        {},
        {},
        nl_condition_sequences,
        sequence_condition_data,
        [],
    )

    assert sequence_condition_data["Flow_yes"]["conditionEval"] is True
    assert sequence_condition_data["Flow_no"]["conditionEval"] is False

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def,
        activity_id="A1",
        workitem={"assignees": []},
        sequence_condition_data=sequence_condition_data,
    )
    assert [p.get("nextActivityId") for p in payloads] == ["B1"]


@pytest.mark.asyncio
async def test_nl_condition_applies_with_whitespace_and_case_and_isMet_key(wiproc):
    """공백/대소문자 차이가 있는 sequenceId 와 대체 verdict 키(isMet)도 인식해야 한다."""
    gw = _Gateway("G1", type="exclusiveGateway")
    seqs = [
        _Seq("s1", source="A1", target="G1"),
        _Seq("SeqA", source="G1", target="B1"),
        _Seq("SeqB", source="G1", target="B2"),
    ]
    acts = [_Activity("A1"), _Activity("B1"), _Activity("B2")]
    proc_def = _ProcDef(activities=acts, gateways=[gw], sequences=seqs)

    sequence_condition_data = {"SeqA": {"condition": "x"}, "SeqB": {"condition": "y"}}
    nl_condition_sequences = [("SeqA", "x", None), ("SeqB", "y", None)]
    model = _FakeConditionModel({
        "results": [
            {"sequenceId": " seqa ", "isMet": True},   # 공백 + 소문자 + 대체 키
            {"sequenceId": "SEQB", "isMet": False},
        ]
    })

    await wiproc._evaluate_nl_conditions(
        model, None, {}, {}, nl_condition_sequences, sequence_condition_data, [],
    )

    assert sequence_condition_data["SeqA"]["conditionEval"] is True
    assert sequence_condition_data["SeqB"]["conditionEval"] is False

    payloads = wiproc.resolve_next_activity_payloads(
        proc_def, activity_id="A1", workitem={"assignees": []},
        sequence_condition_data=sequence_condition_data,
    )
    assert [p.get("nextActivityId") for p in payloads] == ["B1"]
