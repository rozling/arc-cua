"""Evaluator-only regressions; these use no retained private fixture."""
from __future__ import annotations

from tools.evaluate_bounded_decisions import _private_task, _proposal_tuples_valid, run


def _representative_wire() -> dict:
    return {"state": {"subtask": {
        "goal": "Set Search and turn Filter toggle off",
        "verification": ["Browser visible", "Filter toggle is off and Filter View hidden", "Search matches"],
        "inputs": {"query": "needle"},
        "constraints": [], "metadata": {}, "shortcuts": {}, "allowed_risks": [],
    }}}


def test_private_projection_preserves_exact_three_criteria_and_numeric_filter_literal() -> None:
    task = _private_task(_representative_wire(), bounded=True, filter_view_observed=False)
    contract = task.compact()["decision_contract"]
    assert task.inputs["filter_off"] == 0
    assert type(task.inputs["filter_off"]) is int
    assert contract["bindings"]["filter_view"]["name"] == "Filter View"
    assert contract["verification"] == [
        {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": True}]},
        {"criterion_index": 1, "all": [
            {"binding": "filter", "field": "value", "input_key": "filter_off"},
            {"binding": "filter_view", "field": "visible", "value": False},
        ]},
        {"criterion_index": 2, "all": [{"binding": "search", "field": "value", "input_key": "query"}]},
    ]


def test_oversize_comparison_records_typed_no_http_failure_for_both_paths() -> None:
    for bounded in (False, True):
        result = run("oversize", bounded)
        assert result["outcome"] == "ProviderContextUnrepresentable"
        assert result["http_calls"] == 0


def test_proposal_validator_rejects_swapped_target_literal_or_kind() -> None:
    task = _private_task(_representative_wire(), bounded=True, filter_view_observed=False)
    state = {"bindings": {"search": {"id": "search"}, "filter": {"id": "filter"}}}
    valid = {"proposal": {"intent_id": "set_search", "binding": "search", "kind": "TYPE_TEXT", "target_id": "search", "input_key": "query"}}
    assert _proposal_tuples_valid(valid, task, state)
    assert not _proposal_tuples_valid({"proposal": {**valid["proposal"], "target_id": "filter"}}, task, state)
    assert not _proposal_tuples_valid({"proposal": {**valid["proposal"], "input_key": "filter_off"}}, task, state)
    assert not _proposal_tuples_valid({"proposal": {**valid["proposal"], "kind": "SET_VALUE"}}, task, state)
