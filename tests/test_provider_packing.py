"""Offline contract tests for TypeSafe provider-context packing.

The replay test deliberately reads the retained private request in place. It never
copies that payload into this repository or assertion output.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from arc_cua.models import ActionKind, DesktopElement, DesktopSnapshot, Subtask, TerminalKind
from arc_cua.policies import ProviderContextUnrepresentable, TypeSafeJevPolicy
from arc_cua.policies.packing import closure, task_words
from arc_cua.policies.typesafe import _assert_provider_bounds
from arc_cua.validation import materialize_action


PRIVATE_REPLAY = Path("/private/tmp/maxmcp-jev-replay-provenance-20261006/request-wire.json")
PRIVATE_REPLAY_SHA256 = "9d39ae752b5e9c93731e94e7dfa6f30a30c06acc40a0ca5643c76d3b8caf932a"


def _compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode_rows(desktop: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: value for key, value in zip(desktop["element_columns"], row) if value is not None}
            for row in desktop["elements"]]


def _terminal_response(body: dict[str, Any], choice: str = "NEEDS_AGENT") -> httpx.Response:
    criteria = body["questions"]["operation"]["criteria"]
    return httpx.Response(200, json={"answers": {"operation": {
        "choice": choice, "confidence": 1.0,
        "probabilities": {key: float(key == choice) for key in criteria},
    }}})


def _capture(policy: TypeSafeJevPolicy, task: Subtask, snapshot: DesktopSnapshot) -> tuple[Any, list[dict[str, Any]], int]:
    seen: list[dict[str, Any]] = []
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        seen.append(body)
        return _terminal_response(body)

    policy.transport.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        decision = policy.decide(subtask=task, snapshot=snapshot, history=())
    finally:
        policy.transport.client.close()
    return decision, seen, calls


def _adversarial_snapshot(count: int = 250, *, giant_help: bool = True) -> DesktopSnapshot:
    noise_help = "\nSearch Filter Browser " + ("x" * 3_000 if giant_help else "")
    rows = [DesktopElement(id=f"noise-{index}", role="Button", name=f"conciseNoise{index}{noise_help}",
                           actions=(ActionKind.CLICK,)) for index in range(count)]
    rows.extend((
        DesktopElement(id="root", role="Group", name="Filter results"),
        DesktopElement(id="search", role="TextField", name="Search\n" + ("help " * 3_000), value="",
                       actions=(ActionKind.CLICK, ActionKind.TYPE_TEXT), focused=True, parent_id="root"),
        DesktopElement(id="filter", role="Button", name="Show Filter View", value=False,
                       actions=(ActionKind.CLICK,), parent_id="root"),
        DesktopElement(id="status", role="Status", parent_id="root", value=0,
                       name="Status\nTransfer failed: Filter View hidden"),
        DesktopElement(id="late", role="Button", name="Hide Filter View", actions=(ActionKind.CLICK,), parent_id="root"),
    ))
    return DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(rows))


def _task() -> Subtask:
    return Subtask(goal="Hide Filter View and set Search", verification=("Filter View hidden", "Transfer failed"),
                   inputs={"query": "arc fixture"})


def _packed_body(snapshot: DesktopSnapshot, task: Subtask | None = None) -> tuple[dict[str, Any], Any]:
    policy = TypeSafeJevPolicy(api_key="test", model="jev-test")
    decision, seen, calls = _capture(policy, task or _task(), snapshot)
    assert calls == 1
    assert decision.terminal == TerminalKind.NEEDS_AGENT
    return seen[0], decision


def test_adversarial_late_evidence_is_kept_without_treating_giant_action_help_as_relevance() -> None:
    """The specified 200-noise adversary: useful evidence is late, help is misleading."""
    body, _ = _packed_body(_adversarial_snapshot(200))
    rows = _decode_rows(body["state"]["desktop"])
    ids = {row["id"] for row in rows}
    assert {"root", "search", "filter", "status", "late"} <= ids
    assert sum(row["id"].startswith("noise-") for row in rows) < 200
    assert next(row for row in rows if row["id"] == "filter")["value"] is False
    assert next(row for row in rows if row["id"] == "status")["value"] == 0


def test_every_offered_target_and_destination_exists_in_packed_state_and_original_literals_are_unchanged() -> None:
    task = _task()
    original = copy.deepcopy(task.compact())
    body, _ = _packed_body(_adversarial_snapshot(600, giant_help=False), task)
    ids = {row[0] for row in body["state"]["desktop"]["elements"]}
    for name, question in body["questions"].items():
        if name.endswith("_target") or name == "drag_to_destination":
            assert set(question["criteria"]) <= ids
    assert body["state"]["subtask"] == original
    assert task.compact() == original
    assert body["questions"]["type_text_input"]["criteria"]["query"]["value"] == "arc fixture"


def test_many_nonsecret_input_heads_and_literals_survive_target_candidate_packing_unchanged() -> None:
    """Provider packing may trim target heads, never caller-supplied input heads."""
    inputs = {f"literal_{index}": f"provided literal {index}" for index in range(20)}
    task = Subtask(goal="Update field", verification=("Field updated",), inputs=inputs)
    rows = [DesktopElement(id=f"noise-{index}", role="Button", name=f"Unrelated {index}",
                           actions=(ActionKind.CLICK,)) for index in range(700)]
    rows.append(DesktopElement(id="field", role="TextField", name="Update field",
                               actions=(ActionKind.TYPE_TEXT, ActionKind.SET_VALUE), metadata={"value_type": "text"}))
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(rows))
    baseline = TypeSafeJevPolicy(api_key="test")
    try:
        original_questions, _, _ = baseline._build_questions(task, snapshot)
    finally:
        baseline.transport.client.close()
    body, _ = _packed_body(snapshot, task)
    assert body["state"]["subtask"]["inputs"] == inputs
    for head in ("type_text_input", "set_value_input"):
        assert body["questions"][head]["criteria"] == original_questions[head]["criteria"]
        assert set(body["questions"][head]["criteria"]) == set(inputs) | {"NONE"}
    assert len(body["questions"]["click_target"]["criteria"]) < len(original_questions["click_target"]["criteria"])


def test_packing_is_byte_deterministic_and_never_mutates_the_snapshot() -> None:
    snapshot = _adversarial_snapshot(500, giant_help=False)
    before = snapshot.compact()
    first, _ = _packed_body(snapshot)
    second, _ = _packed_body(snapshot)
    assert _compact_json(first) == _compact_json(second)
    assert snapshot.compact() == before


def test_packed_choice_materializes_against_the_original_trusted_snapshot_without_remapping() -> None:
    """A packed provider ID stays an ID into the untouched execution snapshot."""
    original = _adversarial_snapshot(500, giant_help=False)
    trusted_target = DesktopElement(id="late", role="Button", name="Hide Filter View", guard="trusted-guard",
                                    actions=(ActionKind.CLICK,), parent_id="root")
    original = DesktopSnapshot(application=original.application, window=original.window, revision=original.revision,
                               elements=tuple(trusted_target if row.id == "late" else row for row in original.elements))
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content); seen.append(body)
        answers = {
            "operation": {"choice": "CLICK", "confidence": 1.0,
                          "probabilities": {key: float(key == "CLICK") for key in body["questions"]["operation"]["criteria"]}},
            "click_target": {"choice": "late", "confidence": 1.0,
                             "probabilities": {key: float(key == "late") for key in body["questions"]["click_target"]["criteria"]}},
            "click_modifier": {"choice": "NONE", "confidence": 1.0,
                               "probabilities": {key: float(key == "NONE") for key in body["questions"]["click_modifier"]["criteria"]}},
        }
        return httpx.Response(200, json={"answers": answers})

    policy = TypeSafeJevPolicy(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    task = _task()
    decision = policy.decide(subtask=task, snapshot=original, history=())
    assert decision.target_id == "late"
    assert "late" in seen[0]["questions"]["click_target"]["criteria"]
    action = materialize_action(decision, original, task)
    assert action.target_id == "late"
    assert action.target_guard == "trusted-guard"


def test_partial_projection_forces_needs_agent_after_model_claims_completion() -> None:
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content); seen.append(body)
        answers = {}
        for name, question in body["questions"].items():
            choice = "SUBTASK_COMPLETE" if name == "operation" else "SATISFIED"
            answers[name] = {"choice": choice, "confidence": 1.0,
                             "probabilities": {key: float(key == choice) for key in question["criteria"]}}
        return httpx.Response(200, json={"answers": answers})

    policy = TypeSafeJevPolicy(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    decision = policy.decide(subtask=_task(), snapshot=_adversarial_snapshot(400, giant_help=False), history=())
    assert seen[0]["state"]["provider_projection_partial"] is True
    assert decision.terminal == TerminalKind.NEEDS_AGENT
    assert "partial provider projection" in decision.reason


def test_secret_input_never_influences_matching_or_escapes_redaction() -> None:
    secret = "private-never-send"
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(
        DesktopElement(id=f"noise-{index}", role="Label", name=f"Unrelated {index}") for index in range(600)
    ) + (DesktopElement(id="secret-field", role="TextField", name="Credential", value=secret,
                        actions=(ActionKind.TYPE_TEXT,)),))
    task = Subtask(goal="Update credential", verification=("Credential updated",),
                   inputs={"credential": secret}, secret_inputs=("credential",))
    body, _ = _packed_body(snapshot, task)
    encoded = _compact_json(body).decode("utf-8")
    assert secret not in encoded
    assert body["state"]["subtask"]["inputs"]["credential"] == task.compact()["inputs"]["credential"]


@pytest.mark.parametrize(("budget", "oversized"), [((1_800, 48_000), "state_longest"), ((24_000, 1_800), "request")])
def test_each_provider_ceiling_fails_before_http_independently(budget: tuple[int, int], oversized: str) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _terminal_response(json.loads(request.content))

    policy = TypeSafeJevPolicy(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    policy.provider_budget = budget
    with pytest.raises(ProviderContextUnrepresentable, match="provider_context_unrepresentable"):
        policy.decide(subtask=_task(), snapshot=_adversarial_snapshot(10), history=())
    assert calls == 0, oversized


def test_essential_oversized_element_fails_closed_before_http_but_giant_optional_name_is_skipped_losslessly() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _terminal_response(json.loads(request.content))

    essential = DesktopSnapshot(application="App", window="Window", revision="1", elements=(
        DesktopElement(id="needed", role="Status", name="Transfer failed " + ("z" * 30_000)),
    ))
    policy = TypeSafeJevPolicy(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(ProviderContextUnrepresentable):
        policy.decide(subtask=Subtask(goal="Transfer", verification=("Transfer failed",)), snapshot=essential, history=())
    assert calls == 0

    optional_name = "Optional pane " + ("q" * 35_000)
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=(
        DesktopElement(id="needed", role="Button", name="Hide Filter View", actions=(ActionKind.CLICK,)),
        DesktopElement(id="optional", role="Label", name=optional_name),
    ))
    body, _ = _packed_body(snapshot)
    rows = _decode_rows(body["state"]["desktop"])
    assert "optional" not in {row["id"] for row in rows}
    assert snapshot.element("optional").name == optional_name


def test_small_full_state_keeps_the_existing_wire_shape() -> None:
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=(
        DesktopElement(id="search", role="TextField", name="Search", actions=(ActionKind.TYPE_TEXT,)),
        DesktopElement(id="filter", role="Button", name="Hide Filter View", actions=(ActionKind.CLICK,)),
    ))
    body, decision = _packed_body(snapshot)
    assert "provider_projection_partial" not in body["state"]
    assert decision.raw["provider_packing"]["dropped_elements"] == 0
    assert [row[0] for row in body["state"]["desktop"]["elements"]] == ["search", "filter"]


def test_hidden_relevant_or_focused_rows_never_leak_or_make_packing_counts_negative() -> None:
    hidden = DesktopElement(id="hidden-secret", role="Status", name="Transfer failed", focused=True, visible=False)
    visible = _adversarial_snapshot(500, giant_help=False)
    snapshot = DesktopSnapshot(application=visible.application, window=visible.window, revision=visible.revision,
                               elements=(*visible.elements, hidden))
    body, decision = _packed_body(snapshot)
    assert "hidden-secret" not in {row[0] for row in body["state"]["desktop"]["elements"]}
    accounting = decision.raw["provider_packing"]
    assert accounting["mandatory_elements"] >= 0
    assert accounting["optional_elements"] >= 0
    assert accounting["dropped_elements"] >= 0
    assert snapshot.element("hidden-secret") is hidden


def test_relevant_named_group_preserves_nonlexical_nonactionable_status_child_without_window_noise_fanout() -> None:
    task = Subtask(goal="Upload report", verification=("Upload complete",))
    rows = [DesktopElement(id=f"noise-{index}", role="Label", name=f"Window noise {index}", parent_id="root")
            for index in range(700)]
    rows.extend((
        DesktopElement(id="root", role="Window", name="Window noise"),
        DesktopElement(id="upload", role="Group", name="Upload", parent_id="root"),
        DesktopElement(id="ready", role="Status", name="Ready", value=0, parent_id="upload"),
        DesktopElement(id="upload-button", role="Button", name="Upload report", parent_id="upload",
                       actions=(ActionKind.CLICK,)),
    ))
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(rows))
    body, _ = _packed_body(snapshot, task)
    packed = {row["id"]: row for row in _decode_rows(body["state"]["desktop"])}
    original = {element.id: element.compact() for element in snapshot.elements if element.visible}
    assert packed["upload"] == original["upload"]
    assert packed["ready"] == original["ready"]
    assert packed["ready"]["value"] == 0
    assert sum(element_id.startswith("noise-") for element_id in packed) < 700


def test_unnamed_unfocused_visible_ax_modal_context_roles_are_preserved_when_packing() -> None:
    rows = [DesktopElement(id=f"noise-{index}", role="Label", name=f"Unrelated {index}") for index in range(700)]
    rows.extend((
        DesktopElement(id="sheet", role="AXSheet"),
        DesktopElement(id="popover", role="AXPopover"),
        DesktopElement(id="menu", role="AXMenu"),
        DesktopElement(id="target", role="Button", name="Upload report", actions=(ActionKind.CLICK,)),
    ))
    snapshot = DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(rows))
    body, _ = _packed_body(snapshot, Subtask(goal="Upload report", verification=("Report uploaded",)))
    packed = {row["id"]: row for row in _decode_rows(body["state"]["desktop"])}
    original = {element.id: element.compact() for element in snapshot.elements if element.visible}
    for element_id in ("sheet", "popover", "menu"):
        assert packed[element_id] == original[element_id]


def test_relevant_or_focused_window_root_does_not_promote_every_nonlexical_status_or_label_child() -> None:
    task = Subtask(goal="Upload report", verification=("Report uploaded",))
    rows = [
        DesktopElement(id="root", role="Window", name="Upload window", focused=True),
        DesktopElement(id="upload", role="Group", name="Upload", parent_id="root"),
        DesktopElement(id="ready", role="Status", name="Ready", value=0, parent_id="upload"),
    ]
    rows.extend(DesktopElement(id=f"unrelated-{index}", role="Status" if index % 2 else "Label",
                               name=f"Unrelated evidence {index}", parent_id="root") for index in range(300))
    planned = closure(rows, set(), task_words(task))
    assert {"root", "upload", "ready"} <= planned
    assert not any(element_id.startswith("unrelated-") for element_id in planned)


def _snapshot_from_private_wire(wire: dict[str, Any]) -> tuple[Subtask, DesktopSnapshot]:
    state = wire["state"]
    rows = _decode_rows(state["desktop"])
    elements = tuple(DesktopElement(
        id=row["id"], role=row["role"], name=row.get("name", ""), value=row.get("value"),
        actions=tuple(ActionKind(value) for value in row.get("actions", ())), enabled=True, visible=True,
        focused=bool(row.get("focused", False)), selected=row.get("selected"), expanded=row.get("expanded"),
        parent_id=row.get("parent_id"), source=row.get("source", "unknown"),
        accepts_drop=bool(row.get("accepts_drop", False)), metadata=row.get("metadata", {}),
    ) for row in rows)
    compact = state["subtask"]
    task = Subtask(goal=compact["goal"], verification=tuple(compact["verification"]), inputs=compact["inputs"],
                   constraints=tuple(compact["constraints"]), metadata=compact["metadata"],
                   shortcuts=compact["shortcuts"], allowed_risks=tuple(compact["allowed_risks"]))
    return task, DesktopSnapshot(application=state["desktop"]["application"], window=state["desktop"]["window"],
                                 revision="private-replay", context=state["desktop"].get("context", {}), elements=elements)


@pytest.mark.skipif(not PRIVATE_REPLAY.exists(), reason="private retained replay fixture is unavailable")
def test_private_473_row_replay_packs_losslessly_without_copying_private_payload() -> None:
    raw = PRIVATE_REPLAY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PRIVATE_REPLAY_SHA256
    wire = json.loads(raw)
    task, snapshot = _snapshot_from_private_wire(wire)
    before_snapshot = snapshot.compact()
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content); seen.append(body)
        return _terminal_response(body)

    policy = TypeSafeJevPolicy(api_key="test", model=wire["model"], client=httpx.Client(transport=httpx.MockTransport(respond)))
    decision = policy.decide(subtask=task, snapshot=snapshot, history=())
    assert decision.terminal == TerminalKind.NEEDS_AGENT
    assert len(snapshot.elements) == 473
    assert task.compact() == wire["state"]["subtask"]
    assert snapshot.compact() == before_snapshot
    body = seen[0]
    assert len(_compact_json(body)) < len(raw)
    assert body["state"]["subtask"] == wire["state"]["subtask"]
    ids = {row[0] for row in body["state"]["desktop"]["elements"]}
    for question in body["questions"].values():
        if question["type"] == "choice":
            if question is body["questions"].get("drag_to_destination") or question in (
                    body["questions"].get("click_target"), body["questions"].get("double_click_target"),
                    body["questions"].get("right_click_target"), body["questions"].get("type_text_target"),
                    body["questions"].get("set_value_target")):
                assert set(question["criteria"]) <= ids
    original_rows = {row["id"]: row for row in _decode_rows(wire["state"]["desktop"])}
    packed_rows = {row["id"]: row for row in _decode_rows(body["state"]["desktop"])}
    for element_id, packed in packed_rows.items():
        assert packed == original_rows[element_id]
    # Publicly documented replay facts; preserve exact original rows and offered IDs.
    replay_rows = _decode_rows(wire["state"]["desktop"])
    by_exact_name = {row["name"]: row for row in replay_rows}
    search, filter_, browser = (by_exact_name[name] for name in ("Search", "Show Filter View", "Browser"))
    for row in (search, filter_, browser):
        assert packed_rows[row["id"]] == row
    assert search["id"] in body["questions"]["type_text_target"]["criteria"]
    assert filter_["id"] in body["questions"]["click_target"]["criteria"]
    for name, original_question in wire["questions"].items():
        packed_question = body["questions"][name]
        if name.endswith("_target") or name == "drag_to_destination":
            assert set(packed_question["criteria"]) <= set(original_question["criteria"])
        else:
            assert packed_question["criteria"] == original_question["criteria"]


def test_transport_guard_rejects_known_oversized_body_without_network() -> None:
    with pytest.raises(ProviderContextUnrepresentable):
        _assert_provider_bounds({"state": {"x": "z" * 24_000}, "questions": {"q": {}}})
