import json
from pathlib import Path

import pytest
from process_definition import load_process_definition, convert_definition_to_raw_json


TEST_JSON_PATH = Path(__file__).resolve().parent / "test.json"


@pytest.fixture(scope="module")
def raw_json():
    with TEST_JSON_PATH.open("r", encoding="utf-8") as f:
        definition_dict = json.load(f)
    process_definition = load_process_definition(definition_dict)
    return convert_definition_to_raw_json(process_definition)


def _elements_by_type(raw_json, element_type):
    return [e for e in raw_json["elements"] if e["elementType"] == element_type]


def test_top_level_fields_are_preserved(raw_json):
    assert raw_json["processDefinitionId"] == "research_data_survey_and_implementation_auto_deploy_process"
    assert raw_json["processDefinitionName"] == "연구 자료조사 및 구현체 자동 배포 프로세스"
    assert raw_json["isHorizontal"] is True


def test_linear_activity_becomes_activity_element(raw_json):
    activities = _elements_by_type(raw_json, "Activity")
    activity_ids = {a["id"] for a in activities}
    assert "input_survey_info" in activity_ids
    input_survey_info = next(a for a in activities if a["id"] == "input_survey_info")
    assert input_survey_info["role"] == "기획팀"
    assert input_survey_info["skills"] == []


def test_gateway_with_branching_sequences_becomes_gateway_element(raw_json):
    gateways = _elements_by_type(raw_json, "Gateway")
    gateway_ids = {g["id"] for g in gateways}
    assert "gateway" in gateway_ids
    assert "Gateway_0do2146" in gateway_ids

    sequences = _elements_by_type(raw_json, "Sequence")
    branching_targets = {s["target"] for s in sequences if s["source"] == "gateway"}
    assert branching_targets == {"patent_survey", "paper_survey"}


def test_lowercase_start_end_event_types_convert_to_event_elements(raw_json):
    # tests/test.json stores event gateways as "startEvent"/"endEvent" (lowercase-first)
    events = _elements_by_type(raw_json, "Event")
    event_types = {e["id"]: e["type"] for e in events}
    assert event_types.get("start_event") == "StartEvent"
    assert event_types.get("end_event") == "EndEvent"
    # events must not also appear as Gateway elements
    gateway_ids = {g["id"] for g in _elements_by_type(raw_json, "Gateway")}
    assert "start_event" not in gateway_ids
    assert "end_event" not in gateway_ids


def test_subprocess_converts_with_empty_children_events():
    definition_dict = {
        "processDefinitionName": "부모 프로세스",
        "processDefinitionId": "parent_process",
        "description": "",
        "activities": [],
        "sequences": [],
        "gateways": [],
        "subProcesses": [
            {
                "id": "sub_1",
                "name": "하위 프로세스",
                "type": "subProcess",
                "role": "역할",
                "duration": 3,
                "children": {
                    "processDefinitionName": "하위 프로세스 정의",
                    "processDefinitionId": "sub_process_def",
                    "description": "",
                    "data": [],
                    "roles": [],
                    "activities": [
                        {
                            "id": "sub_task_1",
                            "name": "하위 작업",
                            "type": "userTask",
                            "description": "설명",
                            "role": "역할",
                            "inputData": [],
                            "outputData": [],
                            "checkpoints": [],
                        }
                    ],
                    "sequences": [],
                    "gateways": [],
                    "subProcesses": [],
                },
            }
        ],
    }
    process_definition = load_process_definition(definition_dict)
    raw = convert_definition_to_raw_json(process_definition)

    assert len(raw["subProcesses"]) == 1
    sub_raw = raw["subProcesses"][0]
    assert sub_raw["id"] == "sub_1"
    assert sub_raw["duration"] == "3"
    assert sub_raw["processDefinitionId"] == "sub_process_def"
    assert sub_raw["children"]["events"] == []
    assert [a["id"] for a in sub_raw["children"]["activities"]] == ["sub_task_1"]
