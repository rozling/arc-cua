"""Offline contract tests for explicit bounded Subtask decisions."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest

from arc_cua import ActionKind, Bounds, DesktopElement, DesktopSnapshot, Subtask, TerminalKind
from arc_cua.policies import InvalidChoiceResponse, ProviderContextUnrepresentable, TypeSafeJevPolicy
from arc_cua.validation import materialize_action


def _contract(*, verification: list[dict[str, Any]] | None = None, intents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "bindings": {
            "browser": {"role": "Group", "name": "Browser", "source": "macos_ax"},
            "search": {"role": "TextField", "name": "Search", "source": "macos_ax", "parent": "browser"},
            "filter": {"role": "CheckBox", "name": "Show Filter View", "source": "macos_ax", "parent": "browser"},
        },
        "intents": intents or [
            {"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"},
            {"id": "hide_filter", "kind": "ensure_toggle", "binding": "filter", "input_key": "filter_off"},
        ],
        "verification": verification if verification is not None else [
            {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": True}]},
            {"criterion_index": 1, "all": [{"binding": "filter", "field": "value", "input_key": "filter_off"}]},
            {"criterion_index": 2, "all": [{"binding": "search", "field": "value", "input_key": "query"}]},
        ],
    }


def _task(**kwargs: Any) -> Subtask:
    data = {"goal": "Set Search and turn Filter toggle off", "verification": ("Browser visible", "Filter toggle is off", "Search matches"),
            "inputs": {"query": "needle", "filter_off": False}, "decision_contract": _contract()}
    data.update(kwargs)
    return Subtask(**data)


def _snapshot(*, search_value: str = "", filter_value: bool = True, browser_visible: bool = True,
              modal: bool = False, duplicate_search: bool = False) -> DesktopSnapshot:
    rows = [
        DesktopElement(id="browser", role="Group", name="Browser", source="macos_ax", visible=browser_visible),
        DesktopElement(id="search", role="TextField", name="Search", source="macos_ax", parent_id="browser", value=search_value,
                       actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True}, guard="search-guard"),
        DesktopElement(id="filter", role="CheckBox", name="Show Filter View", source="macos_ax", parent_id="browser", value=filter_value,
                       actions=(ActionKind.CLICK,), guard="filter-guard"),
    ]
    if duplicate_search:
        rows.append(DesktopElement(id="duplicate", role="TextField", name="Search", source="macos_ax", parent_id="browser",
                                   actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True}))
    if modal:
        rows.append(DesktopElement(id="modal", role="AXSheet", name="", source="macos_ax"))
    return DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(rows))


def _answer_for(body: dict[str, Any], choice: str) -> httpx.Response:
    criteria = body["questions"]["next_action"]["criteria"]
    return httpx.Response(200, json={"answers": {"next_action": {"choice": choice, "confidence": 1.0,
        "probabilities": {key: float(key == choice) for key in criteria}}}})


def _run(task: Subtask, snapshot: DesktopSnapshot, selector=None) -> tuple[Any, list[dict[str, Any]], int]:
    bodies: list[dict[str, Any]] = []; calls = 0
    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1; body = json.loads(request.content); bodies.append(body)
        choice = selector(body) if selector else next(key for key in body["state"]["proposals"] if "set_search" in key)
        return _answer_for(body, choice)
    policy = TypeSafeJevPolicy(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        decision = policy.decide(subtask=task, snapshot=snapshot, history=())
    finally:
        policy.transport.client.close()
    return decision, bodies, calls


@pytest.mark.parametrize("mutate", [
    lambda contract: contract.update({"extra": 1}),
    lambda contract: contract.update({"version": True}),
    lambda contract: contract["bindings"]["search"].update({"parent": "missing"}),
    lambda contract: contract["bindings"]["browser"].update({"parent": "search"}),
    lambda contract: contract["intents"].append(dict(contract["intents"][0])),
])
def test_contract_boundary_rejects_unknown_keys_coercion_missing_or_cyclic_parents_and_duplicate_intents(mutate) -> None:
    contract = _contract(); mutate(contract)
    with pytest.raises(ValueError):
        _task(decision_contract=contract)


def test_contract_is_frozen_and_compact_only_adds_it_when_supplied() -> None:
    raw = _contract(); task = _task(decision_contract=raw)
    raw["bindings"]["search"]["name"] = "Mutated later"
    assert task.compact()["decision_contract"]["bindings"]["search"]["name"] == "Search"
    with pytest.raises(TypeError):
        task.decision_contract.bindings["new"] = object()  # type: ignore[index]
    legacy = Subtask(goal="Legacy", verification=("Done",))
    assert "decision_contract" not in legacy.compact()


def test_next_action_is_one_atomic_proposal_and_materializes_against_original_guard() -> None:
    task = _task(); snapshot = _snapshot()
    decision, bodies, calls = _run(task, snapshot)
    assert calls == 1
    body = bodies[0]
    assert set(body["questions"]) == {"next_action"}
    proposal_id, proposal = next((key, value) for key, value in body["state"]["proposals"].items() if value["input_key"] == "query")
    assert proposal_id in body["questions"]["next_action"]["criteria"]
    assert {key: proposal[key] for key in ("kind", "target_id", "input_key")} == {
        "kind": "TYPE_TEXT", "target_id": "search", "input_key": "query",
    }
    assert decision.kind == ActionKind.TYPE_TEXT and decision.target_id == "search" and decision.input_key == "query"
    action = materialize_action(decision, snapshot, task)
    assert action.target_id == "search" and action.target_guard == "search-guard" and action.value == "needle"


def test_exact_binding_ambiguity_hidden_missing_and_disabled_targets_do_not_spend_http() -> None:
    for snapshot in (_snapshot(duplicate_search=True),
                     DesktopSnapshot(application="App", window="Window", revision="1", elements=())):
        # The empty snapshot and ambiguous snapshot exercise binding failure. Hidden is covered below.
        decision, _, calls = _run(_task(), snapshot)
        assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0
    visible = list(_snapshot().elements)
    visible[1] = DesktopElement(id="search", role="TextField", name="Search", source="macos_ax", parent_id="browser",
                                visible=False, actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True})
    decision, _, calls = _run(_task(), DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(visible)))
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0


def test_secret_redaction_determinism_and_forged_next_action_are_fail_closed() -> None:
    secret = "do-not-send"
    task = _task(inputs={"query": secret, "filter_off": False}, secret_inputs=("query",))
    bodies: list[dict[str, Any]] = []
    def choose(body: dict[str, Any]) -> str:
        bodies.append(deepcopy(body)); return "forged"
    with pytest.raises(InvalidChoiceResponse):
        _run(task, _snapshot(), choose)
    assert secret not in json.dumps(bodies[0], ensure_ascii=False)


def test_unsatisfied_intent_cannot_offer_completion_even_when_bound_criterion_is_true() -> None:
    task = _task()
    seen: list[dict[str, Any]] = []
    def forge(body: dict[str, Any]) -> str:
        seen.append(body)
        return "SUBTASK_COMPLETE"
    with pytest.raises(InvalidChoiceResponse):
        _run(task, _snapshot(search_value="needle", filter_value=True), selector=forge)
    assert "SUBTASK_COMPLETE" not in seen[0]["questions"]["next_action"]["criteria"]


def test_false_and_zero_are_not_equal_in_bound_verification_evidence() -> None:
    contract = _contract(verification=[
        {"criterion_index": 0, "all": [{"binding": "filter", "field": "value", "value": False}]},
    ], intents=[{"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"}])
    task = _task(verification=("Filter is off",), decision_contract=contract)
    decision, bodies, _ = _run(task, _snapshot(filter_value=0), selector=lambda body: next(iter(body["state"]["proposals"])))
    evidence = decision.raw["decision_contract"]["evidence"]
    # Exact scalar equality is type-aware: observed integer zero cannot establish boolean False.
    assert evidence == [{"criterion_index": 0, "status": "UNKNOWN", "covered": False,
                         "predicates": [{"binding": "filter", "field": "value", "status": "UNKNOWN"}]}]
    assert "SUBTASK_COMPLETE" not in bodies[0]["questions"]["next_action"]["criteria"]


def test_conflicting_intents_and_active_modal_handoff_before_http() -> None:
    conflict = _contract(intents=[
        {"id": "first", "kind": "set_text", "binding": "search", "input_key": "query"},
        {"id": "second", "kind": "set_value", "binding": "search", "input_key": "other"},
    ])
    task = _task(inputs={"query": "one", "other": "two", "filter_off": False}, decision_contract=conflict)
    decision, _, calls = _run(task, _snapshot())
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0
    decision, _, calls = _run(_task(), _snapshot(modal=True))
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0


def test_missing_and_observed_hidden_are_unknown_not_silent_visible_false() -> None:
    contract = _contract(verification=[
        {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": False}]},
    ], intents=[{"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"}])
    task = _task(verification=("Browser hidden",), decision_contract=contract)
    missing = DesktopSnapshot(application="App", window="Window", revision="1", elements=())
    decision, _, calls = _run(task, missing)
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0
    hidden = _snapshot(browser_visible=False)
    # Binding sees invisible elements, but the hidden browser cannot imply a missing/hidden target is safe to act on.
    decision, _, calls = _run(task, hidden)
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0


def test_observed_hidden_view_satisfies_its_own_predicate_but_missing_view_is_unknown() -> None:
    contract = _contract(verification=[
        {"criterion_index": 0, "all": [{"binding": "view", "field": "visible", "value": False}]},
    ], intents=[{"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"}])
    contract["bindings"]["view"] = {"role": "Group", "name": "Filter View", "source": "macos_ax", "parent": "browser"}
    task = _task(verification=("Filter View hidden",), decision_contract=contract)
    base = _snapshot()
    hidden_view = DesktopSnapshot(application=base.application, window=base.window, revision=base.revision, elements=(*base.elements,
        DesktopElement(id="view", role="Group", name="Filter View", source="macos_ax", parent_id="browser", visible=False)))
    _, bodies, _ = _run(task, hidden_view, selector=lambda body: next(iter(body["state"]["proposals"])))
    assert bodies[0]["state"]["evidence"][0]["status"] == "SATISFIED"
    decision, bodies, _ = _run(task, base, selector=lambda body: next(iter(body["state"]["proposals"])))
    assert decision.raw["decision_contract"]["evidence"][0]["status"] == "UNKNOWN"
    assert bodies[0]["state"]["evidence"][0]["covered"] is False


def test_giant_essential_bound_context_fails_before_mock_http() -> None:
    giant = "x" * 30_000
    task = _task()
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=(
        DesktopElement(id="browser", role="Group", name="Browser", source="macos_ax", metadata={"unavoidable": giant}),
        DesktopElement(id="search", role="TextField", name="Search", source="macos_ax", parent_id="browser", actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True}),
        DesktopElement(id="filter", role="CheckBox", name="Show Filter View", source="macos_ax", parent_id="browser", value=True, actions=(ActionKind.CLICK,)),
    ))
    with pytest.raises(ProviderContextUnrepresentable):
        _run(task, snapshot)


def test_all_satisfied_intents_and_every_criterion_offer_only_narrow_completion() -> None:
    task = _task()
    decision, bodies, calls = _run(task, _snapshot(search_value="needle", filter_value=False),
                                   selector=lambda body: "SUBTASK_COMPLETE")
    assert calls == 1 and decision.terminal == TerminalKind.SUBTASK_COMPLETE
    criteria = bodies[0]["questions"]["next_action"]["criteria"]
    assert set(criteria) == {"NEEDS_AGENT", "BLOCKED", "SUBTASK_COMPLETE"}
    assert bodies[0]["state"]["evidence"] == decision.raw["decision_contract"]["evidence"]


def test_satisfied_intent_but_unsatisfied_criterion_handoffs_without_http() -> None:
    contract = _contract(verification=[
        {"criterion_index": 0, "all": [{"binding": "filter", "field": "value", "input_key": "filter_off"}]},
    ], intents=[{"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"}])
    task = _task(verification=("Filter is off",), decision_contract=contract)
    decision, _, calls = _run(task, _snapshot(search_value="needle", filter_value=True))
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0
    assert decision.raw["decision_contract"]["evidence"][0]["status"] == "NOT_SATISFIED"


def test_parent_binding_resolves_when_child_precedes_parent_and_duplicate_ids_fail_closed() -> None:
    original = _snapshot()
    reordered = DesktopSnapshot(application=original.application, window=original.window, revision=original.revision,
                                elements=(original.elements[1], original.elements[2], original.elements[0]))
    _, bodies, calls = _run(_task(), reordered)
    assert calls == 1 and bodies[0]["state"]["bindings"]["search"]["id"] == "search"
    duplicate = DesktopSnapshot(application="App", window="Window", revision="1", elements=(*original.elements,
        DesktopElement(id="search", role="Label", name="Unrelated")))
    decision, _, calls = _run(_task(), duplicate)
    assert decision.terminal == TerminalKind.NEEDS_AGENT and calls == 0
    assert decision.raw["decision_contract"]["code"] == "duplicate_observed_ids"


def test_multiple_fields_keep_each_proposal_paired_with_its_own_literal() -> None:
    contract = _contract()
    contract["bindings"]["replace"] = {"role": "TextField", "name": "Replace", "source": "macos_ax", "parent": "browser"}
    contract["intents"] = [
        {"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"},
        {"id": "set_replace", "kind": "set_text", "binding": "replace", "input_key": "replacement"},
    ]
    task = _task(inputs={"query": "needle", "replacement": "replacement", "filter_off": False}, decision_contract=contract)
    base = _snapshot()
    snapshot = DesktopSnapshot(application=base.application, window=base.window, revision=base.revision, elements=(*base.elements,
        DesktopElement(id="replace", role="TextField", name="Replace", source="macos_ax", parent_id="browser", value="",
                       actions=(ActionKind.TYPE_TEXT,), metadata={"text_editable": True}, guard="replace-guard")))
    decision, bodies, _ = _run(task, snapshot, selector=lambda body: next(
        key for key, proposal in body["state"]["proposals"].items() if proposal["input_key"] == "replacement"))
    assert decision.kind == ActionKind.TYPE_TEXT and decision.target_id == "replace" and decision.input_key == "replacement"
    action = materialize_action(decision, snapshot, task)
    assert action.value == "replacement" and action.target_guard == "replace-guard"
    assert len(bodies[0]["state"]["proposals"]) == 2
