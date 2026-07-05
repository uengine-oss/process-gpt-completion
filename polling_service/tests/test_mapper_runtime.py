from types import SimpleNamespace

import pytest

from mapper_runtime import (
    apply_mapping_result_to_instance,
    collect_mapping_contexts,
    evaluate_mapping_context,
    merge_role_bindings,
    merge_form_values,
    read_variables_data,
)


def test_read_variables_data_supports_array_and_object_map():
    assert read_variables_data([{"key": "customerId", "value": "C001"}]) == {"customerId": "C001"}
    assert read_variables_data([{"name": "approved", "value": True}]) == {"approved": True}
    assert read_variables_data({"riskLevel": "HIGH"}) == {"riskLevel": "HIGH"}


def test_collect_mapping_contexts_from_activity_properties():
    activity = SimpleNamespace(
        properties='{"eventSynchronization":{"mappingContext":{"mappingElements":[{"argument":{"text":"customerName"},"variable":{"name":"name"}}]}}}'
    )

    contexts = collect_mapping_contexts(activity)

    assert len(contexts) == 1
    assert contexts[0]["mappingElements"][0]["argument"]["text"] == "customerName"


def test_direct_mapping_writes_plain_target_to_variables_data():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "customerName"},
                    "direction": "out",
                    "variable": {"name": "name"},
                }
            ]
        },
        {
            "forms": {"current": {"name": "Jane Kim"}},
            "instance": {"variablesData": {}},
        },
    )

    assert result["variables_data"] == {"customerName": "Jane Kim"}
    assert result["trace"][0]["status"] == "mapped"
    assert result["trace"][0]["scope"] == "variables_data"


def test_explicit_current_form_target_is_merged_to_default_form():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "forms.current.fullName"},
                    "direction": "out",
                    "variable": {"name": "name"},
                }
            ]
        },
        {
            "forms": {"current": {"name": "Jane Kim"}},
            "instance": {"variablesData": {}},
        },
        default_form_id="approvalForm",
    )

    assert result["form_values"] == {"approvalForm": {"fullName": "Jane Kim"}}
    assert merge_form_values({"approvalForm": {"name": "Jane Kim"}}, result) == {
        "approvalForm": {"name": "Jane Kim", "fullName": "Jane Kim"}
    }


def test_legacy_variables_namespace_can_be_used_as_source():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "customerNameCopy"},
                    "direction": "out",
                    "variable": {"name": "[variables].customerName"},
                }
            ]
        },
        {
            "forms": {"current": {}},
            "instance": {"variablesData": {"customerName": "Jane Kim"}},
        },
    )

    assert result["variables_data"] == {"customerNameCopy": "Jane Kim"}


def test_lane_binding_target_writes_role_binding_patch():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "lane.approver.endpoint"},
                    "direction": "out",
                    "variable": {"name": "approverEmail"},
                }
            ]
        },
        {
            "forms": {"current": {"approverEmail": "approver@example.com"}},
            "instance": {"variablesData": {}, "roleBindings": []},
        },
    )

    assert result["role_bindings"] == [{"name": "approver", "endpoint": "approver@example.com"}]
    assert result["variables_data"] == {}
    assert result["trace"][0]["scope"] == "role_bindings"


def test_lane_binding_target_normalizes_single_user_select_array():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "lane.reviewer.endpoint"},
                    "direction": "out",
                    "variable": {"name": "담당자"},
                }
            ]
        },
        {
            "forms": {"current": {"담당자": ["user-1"]}},
            "instance": {"variablesData": {}, "roleBindings": []},
        },
    )

    assert result["role_bindings"] == [{"name": "reviewer", "endpoint": "user-1"}]


def test_explicit_form_id_source_reads_from_forms_by_id():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "lane.보안검토담당자.endpoint"},
                    "direction": "out",
                    "variable": {"name": "forms.vendor_onboarding_task_submit_vendor_need_form.담당자"},
                }
            ]
        },
        {
            "forms": {
                "current": {"담당자": "user-1"},
                "byId": {"vendor_onboarding_task_submit_vendor_need_form": {"담당자": "user-1"}},
            },
            "instance": {"variablesData": {}, "roleBindings": []},
        },
    )

    assert result["role_bindings"] == [{"name": "보안검토담당자", "endpoint": "user-1"}]
    assert result["trace"][0]["status"] == "mapped"


def test_instance_lane_binding_target_alias_is_supported():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "instance.lane.reviewer.endpoint"},
                    "direction": "out",
                    "variable": {"name": "reviewerEmail"},
                }
            ]
        },
        {
            "forms": {"current": {"reviewerEmail": "reviewer@example.com"}},
            "instance": {"variablesData": {}, "roleBindings": []},
        },
    )

    assert result["role_bindings"] == [{"name": "reviewer", "endpoint": "reviewer@example.com"}]


def test_role_binding_merge_replaces_same_role_and_keeps_other_roles():
    assert merge_role_bindings(
        [{"name": "requester", "endpoint": "requester@example.com"}, {"name": "approver", "endpoint": "old@example.com"}],
        [{"name": "approver", "endpoint": "new@example.com"}],
    ) == [
        {"name": "requester", "endpoint": "requester@example.com"},
        {"name": "approver", "endpoint": "new@example.com"},
    ]


def test_mapping_result_applies_role_bindings_to_process_instance():
    instance = SimpleNamespace(role_bindings=[{"name": "approver", "endpoint": "old@example.com"}], variables_data={})

    apply_mapping_result_to_instance(
        instance,
        {"role_bindings": [{"name": "approver", "endpoint": "new@example.com"}], "variables_data": {}},
    )

    assert instance.role_bindings == [{"name": "approver", "endpoint": "new@example.com"}]


def test_supported_concat_transformer_runs():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "fullName"},
                    "direction": "out",
                    "transformerMapping": {
                        "transformer": {
                            "_type": "org.uengine.processdesigner.mapper.transformers.ConcatTransformer",
                            "argumentSourceMap": {
                                "str1": "firstName",
                                "str2": "lastName",
                            },
                        },
                        "linkedArgumentName": "fullName",
                    },
                }
            ]
        },
        {
            "forms": {"current": {"firstName": "Jane", "lastName": "Kim"}},
            "instance": {"variablesData": {}},
        },
    )

    assert result["variables_data"] == {"fullName": "JaneKim"}


def _evaluate_single_transformer(transformer_name, argument_source_map, form_values=None, transformer_attrs=None):
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "mappedValue"},
                    "direction": "out",
                    "transformerMapping": {
                        "transformer": {
                            "_type": f"org.uengine.processdesigner.mapper.transformers.{transformer_name}",
                            "argumentSourceMap": argument_source_map,
                            **(transformer_attrs or {}),
                        },
                        "linkedArgumentName": "mappedValue",
                    },
                }
            ]
        },
        {
            "forms": {"current": form_values or {}},
            "instance": {"variablesData": {}},
        },
    )
    return result["variables_data"].get("mappedValue")


@pytest.mark.parametrize(
    "transformer_name,argument_source_map,form_values,expected",
    [
        ("SumTransformer", {"val1": "a", "val2": "b", "val3": "c"}, {"a": "1,000", "b": 20, "c": "3.5"}, 1023.5),
        ("AbsTransformer", {"input": "a"}, {"a": -7}, 7.0),
        ("CeilTransformer", {"input": "a"}, {"a": 1.2}, 2),
        ("FloorTransformer", {"input": "a"}, {"a": 1.8}, 1),
        ("RoundTransformer", {"input": "a"}, {"a": 1.6}, 2),
        ("MaxTransformer", {"value1": "a", "value2": "b"}, {"a": 3, "b": 8}, 8.0),
        ("MinTransformer", {"value1": "a", "value2": "b"}, {"a": 3, "b": 8}, 3.0),
    ],
)
def test_supported_numeric_transformers_run(transformer_name, argument_source_map, form_values, expected):
    assert _evaluate_single_transformer(transformer_name, argument_source_map, form_values) == expected


def test_supported_replace_transformer_runs_plain_replace():
    assert _evaluate_single_transformer(
        "ReplaceTransformer",
        {"input": "text"},
        {"text": "Jane Kim"},
        {"oldString": "Kim", "newString": "Lee"},
    ) == "Jane Lee"


def test_supported_replace_transformer_runs_regex_replace():
    assert _evaluate_single_transformer(
        "ReplaceTransformer",
        {"input": "text"},
        {"text": "AB-123-CD"},
        {"oldString": r"\d+", "newString": "000", "isRegularExp": True},
    ) == "AB-000-CD"


def test_supported_direct_value_transformers_return_value_attribute():
    assert _evaluate_single_transformer(
        "DirectValueTransformer",
        {},
        {},
        {"value": "constant-value"},
    ) == "constant-value"
    assert _evaluate_single_transformer(
        "DirectSqlExpressionTransformer",
        {},
        {},
        {"value": "select-not-executed"},
    ) == "select-not-executed"


def test_nested_supported_transformer_runs():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "label"},
                    "direction": "out",
                    "transformerMapping": {
                        "transformer": {
                            "_type": "org.uengine.processdesigner.mapper.transformers.ConcatTransformer",
                            "argumentSourceMap": {
                                "str1": {
                                    "transformer": {
                                        "_type": "org.uengine.processdesigner.mapper.transformers.DirectValueTransformer",
                                        "value": "VIP-",
                                        "argumentSourceMap": {},
                                    }
                                },
                                "str2": "name",
                            },
                        },
                        "linkedArgumentName": "label",
                    },
                }
            ]
        },
        {
            "forms": {"current": {"name": "Jane"}},
            "instance": {"variablesData": {}},
        },
    )

    assert result["variables_data"] == {"label": "VIP-Jane"}


def test_unsupported_transformer_is_reported():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "value"},
                    "direction": "out",
                    "transformerMapping": {
                        "transformer": {
                            "_type": "org.uengine.processdesigner.mapper.transformers.BeanValueTransformer",
                            "argumentSourceMap": {},
                        },
                        "linkedArgumentName": "value",
                    },
                }
            ]
        },
        {"forms": {"current": {}}, "instance": {"variablesData": {}}},
    )

    assert result["errors"][0]["code"] == "UNSUPPORTED_TRANSFORMER"
    assert result["trace"][0]["status"] == "error"

def test_uppercase_variables_namespace_can_be_used_as_source():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "assigneeCopy"},
                    "direction": "out",
                    "variable": {"name": "Variables.담당자"},
                }
            ]
        },
        {
            "forms": {"current": {}},
            "instance": {"variablesData": {"담당자": "security-reviewer@example.com"}},
        },
    )

    assert result["variables_data"] == {"assigneeCopy": "security-reviewer@example.com"}


def test_call_activity_variable_target_writes_child_variable_without_prefix():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "callActivity.variables.담당자"},
                    "direction": "out",
                    "variable": {"name": "Variables.담당자"},
                }
            ]
        },
        {
            "forms": {"current": {}},
            "instance": {"variablesData": {"담당자": "security-reviewer@example.com"}},
        },
    )

    assert result["variables_data"] == {"담당자": "security-reviewer@example.com"}
    assert "callActivity" not in result["variables_data"]


def test_call_activity_lane_endpoint_target_writes_child_role_binding():
    result = evaluate_mapping_context(
        {
            "mappingElements": [
                {
                    "argument": {"text": "callActivity.lane.보안심사자.endpoint"},
                    "direction": "out",
                    "variable": {"name": "Variables.담당자"},
                }
            ]
        },
        {
            "forms": {"current": {}},
            "instance": {"variablesData": {"담당자": "security-reviewer@example.com"}},
        },
    )

    assert result["role_bindings"] == [{"name": "보안심사자", "endpoint": "security-reviewer@example.com"}]
    assert result["trace"][0]["scope"] == "role_bindings"

