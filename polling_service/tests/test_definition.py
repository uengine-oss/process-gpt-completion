import json
from pathlib import Path

import pytest
from process_definition import load_process_definition


TEST_JSON_PATH = Path(__file__).resolve().parent / "test.json"


@pytest.fixture(scope="module")
def parent_def():
    assert TEST_JSON_PATH.exists(), f"Test data JSON not found: {TEST_JSON_PATH}"
    with TEST_JSON_PATH.open("r", encoding="utf-8") as f:
        parent_def_dict = json.load(f)
    return load_process_definition(parent_def_dict)


def test_loads_process_definition(parent_def):
    # Sanity check: object returned
    assert parent_def is not None


@pytest.mark.parametrize("target_id", ["Gateway_0do2146", "start_event"])
def test_find_known_gateway_block(parent_def, target_id):
    gateway = parent_def.find_gateway_by_id(target_id)
    assert gateway is not None, f"Gateway not found: {target_id}"
    assert getattr(gateway, "id", None) == target_id


def test_load_process_definition_does_not_mutate_input():
    """같은 definition dict 로 여러 번 호출해도 events 가 gateways 에 중복 누적되면 안 된다.

    한 워크아이템을 처리하는 동안 load_process_definition 은 여러 번 호출된다.
    예전에는 이 함수가 입력 dict 의 gateways 에 events 를 직접 append 해서,
    호출부가 같은 dict 를 재사용하면 호출 횟수만큼 이벤트가 중복됐다.
    중복 이벤트는 게이트웨이 분기 판정을 망가뜨려 다음 액티비티가 잘못 정해진다.
    """
    with TEST_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    original_gateways = len(data.get("gateways", []))
    assert data.get("events"), "이 테스트는 events 가 있는 정의를 전제로 한다"

    results = [load_process_definition(data) for _ in range(3)]

    # 입력 dict 는 그대로여야 한다
    assert len(data.get("gateways", [])) == original_gateways

    # 매 호출의 파싱 결과도 동일해야 한다 (누적되지 않음)
    gateway_counts = {len(pd.gateways) for pd in results}
    assert len(gateway_counts) == 1, f"호출마다 gateway 수가 달라짐: {gateway_counts}"

    # 결과는 항상 '원본 gateways + events' 여야 한다.
    # (fixture 의 gateways 에 이미 중복 id 가 들어 있을 수 있으므로 id 유일성은 검사하지 않고,
    #  '호출 횟수만큼 늘어나지 않는다'는 점만 확인한다)
    assert gateway_counts.pop() == original_gateways + len(data["events"])


def test_loads_test_subprocess_definition():
    path = Path(__file__).resolve().parent / "testSubprocess.json"
    assert path.exists(), f"Test data JSON not found: {path}"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    obj = load_process_definition(data)
    assert obj is not None
