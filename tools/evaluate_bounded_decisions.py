#!/usr/bin/env python3
"""Offline oracle comparison for generic packed and explicit bounded decisions.

This measures request construction only.  It uses MockTransport and deterministic
terminal answers; it makes no claim about provider selection quality or latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from arc_cua import ActionKind, DesktopElement, DesktopSnapshot, Subtask
from arc_cua.policies import TypeSafeJevPolicy


PRIVATE_SHA256 = "9d39ae752b5e9c93731e94e7dfa6f30a30c06acc40a0ca5643c76d3b8caf932a"


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def contract(*, second_field: bool = False) -> dict[str, Any]:
    return {"version": 1, "bindings": {
        "browser": {"role": "Group", "name": "Browser", "source": "mock"},
        "search": {"role": "TextField", "name": "Search", "source": "mock", "parent": "browser"},
        "filter": {"role": "CheckBox", "name": "Filter", "source": "mock", "parent": "browser"},
        "filter_view": {"role": "Group", "name": "Filter View", "source": "mock", "parent": "browser"},
        **({"replace": {"role": "TextField", "name": "Replace", "source": "mock", "parent": "browser"}} if second_field else {}),
    }, "intents": [
        {"id": "search", "kind": "set_text", "binding": "search", "input_key": "query"},
        {"id": "filter", "kind": "ensure_toggle", "binding": "filter", "input_key": "filter_off"},
    ], "verification": [
        {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": True}]},
        {"criterion_index": 1, "all": [{"binding": "filter", "field": "value", "input_key": "filter_off"}]},
        {"criterion_index": 2, "all": [{"binding": "filter_view", "field": "visible", "value": False}]},
    ]}


def snapshot(name: str) -> DesktopSnapshot:
    elements = [
        DesktopElement(id="browser", role="Group", name="Browser", source="mock"),
        DesktopElement(id="search", role="TextField", name="Search", source="mock", parent_id="browser", value="", actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True}),
        DesktopElement(id="filter", role="CheckBox", name="Filter", source="mock", parent_id="browser", value=True, actions=(ActionKind.CLICK,)),
        DesktopElement(id="filter-view", role="Group", name="Filter View", source="mock", parent_id="browser", visible=False),
    ]
    if name == "ambiguity": elements.append(DesktopElement(id="search-2", role="TextField", name="Search", source="mock", parent_id="browser", actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True}))
    if name == "multifield": elements.append(DesktopElement(id="replace", role="TextField", name="Replace", source="mock", parent_id="browser", value="", actions=(ActionKind.TYPE_TEXT,), metadata={"editable": True}))
    if name == "missing": elements = []
    if name == "hidden": elements[0] = DesktopElement(id="browser", role="Group", name="Browser", source="mock", visible=False)
    if name == "modal": elements.append(DesktopElement(id="sheet", role="AXSheet", name="", source="mock"))
    if name == "noise": elements.extend(DesktopElement(id=f"noise-{i}", role="Label", name=f"Unrelated {i}", source="mock") for i in range(700))
    if name == "oversize": elements[0] = DesktopElement(id="browser", role="Group", name="Browser", source="mock", value="x" * 30_000)
    return DesktopSnapshot(application="Mock", window="Mock", revision="1", elements=tuple(elements))


def task(name: str, bounded: bool) -> Subtask:
    inputs = {"query": "needle", "filter_off": False}
    mapping = contract(second_field=name == "multifield")
    goal = "Set Search and Filter"
    verification = ("Browser visible", "Filter toggle is off", "Filter View hidden")
    if name == "multifield":
        inputs["query_2"] = "other"
        mapping["intents"] = mapping["intents"][:1] + [{"id": "replace", "kind": "set_text", "binding": "replace", "input_key": "query_2"}]
        goal = "Set Search and Replace"
        verification = ("Browser visible", "Search matches", "Replace matches")
        mapping["verification"] = [
            {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": True}]},
            {"criterion_index": 1, "all": [{"binding": "search", "field": "value", "input_key": "query"}]},
            {"criterion_index": 2, "all": [{"binding": "replace", "field": "value", "input_key": "query_2"}]},
        ]
    if name == "contradiction":
        inputs["other"] = "other"
        mapping["intents"] = mapping["intents"][:1] + [{"id": "conflict", "kind": "set_value", "binding": "search", "input_key": "other"}]
    return Subtask(goal=goal, verification=verification, inputs=inputs,
                   decision_contract=mapping if bounded else None)


def run(name: str, bounded: bool) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content); captured.append(body)
        questions = body["questions"]
        key = "next_action" if "next_action" in questions else "operation"
        criteria = questions[key]["criteria"]
        choice = next((candidate for candidate in criteria if candidate.startswith("proposal_")), "NEEDS_AGENT") if key == "next_action" else "NEEDS_AGENT"
        return httpx.Response(200, json={"answers": {key: {"choice": choice, "confidence": 1.0,
            "probabilities": {candidate: float(candidate == choice) for candidate in criteria}}}})
    policy = TypeSafeJevPolicy(api_key="offline", model="offline", client=httpx.Client(transport=httpx.MockTransport(respond)))
    started = time.perf_counter()
    subtask = task(name, bounded)
    try:
        decision = policy.decide(subtask=subtask, snapshot=snapshot(name), history=())
        error = None
    except Exception as exc:  # Contract failures are outcomes; no private exception payload is retained.
        decision = None; error = type(exc).__name__
    finally:
        policy.transport.client.close()
    body = captured[0] if captured else None
    proposals = body["state"].get("proposals", {}) if body else {}
    coherent = _proposal_tuples_valid(proposals, subtask, body["state"] if body else {}) if bounded else True
    return {"http_calls": len(captured), "construction_ms": round((time.perf_counter() - started) * 1000, 3),
            "outcome": error or (decision.terminal.value if decision and decision.terminal else "ACTION"),
            "body_bytes": len(encoded(body)) if body else 0,
            "question_count": len(body["questions"]) if body else 0,
            "choice_count": sum(len(question["criteria"]) for question in body["questions"].values()) if body else 0,
            "proposal_count": len(proposals),
            "proposal_tuples_valid": coherent,
            "completion_eligible": decision.raw.get("decision_contract", {}).get("completion_eligible") if decision else None,
            "evidence": body["state"].get("evidence", []) if body else [],
            "oracle": "offline_mock_contract_construction_only"}


def _proposal_tuples_valid(proposals: dict[str, dict[str, Any]], subtask: Subtask, state: dict[str, Any]) -> bool:
    """Check captured provider tuples against the typed intent that constructed them."""
    if subtask.decision_contract is None:
        return not proposals
    expected = {intent.id: intent for intent in subtask.decision_contract.intents}
    bindings = state.get("bindings", {})
    for proposal in proposals.values():
        intent = expected.get(proposal.get("intent_id"))
        if intent is None or proposal.get("binding") != intent.binding:
            return False
        if proposal.get("target_id") != bindings.get(intent.binding, {}).get("id"):
            return False
        expected_kind = {"set_text": "TYPE_TEXT", "set_value": "SET_VALUE", "ensure_toggle": "CLICK"}[intent.kind]
        if proposal.get("kind") != expected_kind:
            return False
        if expected_kind != "CLICK" and proposal.get("input_key") != intent.input_key:
            return False
        if expected_kind == "CLICK" and "input_key" in proposal:
            return False
    return True


def source_hashes(root: Path) -> dict[str, str]:
    paths = [*sorted((root / "src/arc_cua").rglob("*.py")), Path(__file__).resolve()]
    return {str(path.relative_to(root)) if path.is_relative_to(root) else path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _decode_rows(desktop: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: value for key, value in zip(desktop["element_columns"], row) if value is not None}
            for row in desktop["elements"]]


def _private_snapshot(wire: dict[str, Any]) -> DesktopSnapshot:
    state = wire["state"]; desktop = state["desktop"]
    elements = tuple(DesktopElement(id=row["id"], role=row["role"], name=row.get("name", ""), value=row.get("value"),
        actions=tuple(ActionKind(value) for value in row.get("actions", ())), visible=True, enabled=True,
        focused=bool(row.get("focused", False)), selected=row.get("selected"), expanded=row.get("expanded"),
        parent_id=row.get("parent_id"), source=row.get("source", "unknown"), metadata=row.get("metadata", {}))
        for row in _decode_rows(desktop))
    return DesktopSnapshot(application=desktop["application"], window=desktop["window"], revision="private-473",
                           context=desktop.get("context", {}), elements=elements)


def _private_task(wire: dict[str, Any], *, bounded: bool, filter_view_observed: bool) -> Subtask:
    compact = wire["state"]["subtask"]; inputs = dict(compact["inputs"])
    if bounded:
        # This retained UI reports the checkbox as a numeric binary state, and bounded equality is type-aware.
        inputs["filter_off"] = 0
    fields = {"goal": compact["goal"], "verification": tuple(compact["verification"]), "inputs": inputs,
              "constraints": tuple(compact["constraints"]), "metadata": compact["metadata"],
              "shortcuts": compact["shortcuts"], "allowed_risks": tuple(compact["allowed_risks"])}
    if bounded:
        # The retained fixture's private labels never leave this process. Exact selectors either bind observed
        # rows or deliberately remain unresolved, which produces UNKNOWN rather than claiming hidden.
        fields["decision_contract"] = {"version": 1, "bindings": {
            "browser": {"role": "Group", "name": "Browser"},
            "search": {"role": "TextField", "name": "Search", "parent": "browser"},
            "filter": {"role": "CheckBox", "name": "Show Filter View", "parent": "browser"},
            "filter_view": {"role": "Group", "name": "Filter View", "parent": "browser"},
        }, "intents": [
            {"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"},
            {"id": "hide_filter", "kind": "ensure_toggle", "binding": "filter", "input_key": "filter_off"},
        ], "verification": [
            {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": True}]},
            {"criterion_index": 1, "all": [
                {"binding": "filter", "field": "value", "input_key": "filter_off"},
                {"binding": "filter_view", "field": "visible", "value": False},
            ]},
            {"criterion_index": 2, "all": [{"binding": "search", "field": "value", "input_key": "query"}]},
        ]}
    return Subtask(**fields)


def private_473(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIVATE_SHA256:
        raise ValueError("private fixture SHA-256 does not match the retained replay")
    wire = json.loads(raw); snapshot = _private_snapshot(wire)
    rows = _decode_rows(wire["state"]["desktop"])
    observed_filter_view = any(row.get("name") == "Filter View" for row in rows)
    # Reuse the same strictly offline request capture used by the synthetic comparison.
    def capture(task: Subtask) -> dict[str, Any]:
        captured: list[dict[str, Any]] = []
        def respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content); captured.append(body)
            questions = body["questions"]; key = "next_action" if "next_action" in questions else "operation"
            criteria = questions[key]["criteria"]; choice = next((key for key in criteria if key.startswith("proposal_")), "NEEDS_AGENT")
            return httpx.Response(200, json={"answers": {key: {"choice": choice, "confidence": 1.0,
                "probabilities": {candidate: float(candidate == choice) for candidate in criteria}}}})
        policy = TypeSafeJevPolicy(api_key="offline", model=wire["model"], client=httpx.Client(transport=httpx.MockTransport(respond)))
        try:
            decision = policy.decide(subtask=task, snapshot=snapshot, history=())
        except Exception as exc:
            return {"http_calls": len(captured), "outcome": type(exc).__name__}
        finally:
            policy.transport.client.close()
        body = captured[0] if captured else None
        return {"http_calls": len(captured), "outcome": decision.terminal.value if decision.terminal else "ACTION",
                "body_bytes": len(encoded(body)) if body else 0, "question_count": len(body["questions"]) if body else 0,
                "proposal_count": len(body["state"].get("proposals", {})) if body else 0,
                "completion_eligible": decision.raw.get("decision_contract", {}).get("completion_eligible"),
                "evidence": body["state"].get("evidence", []) if body else [],
                "oracle": "offline_mock_contract_construction_only"}
    return {"sha256": PRIVATE_SHA256, "exported": False, "rows": len(rows), "filter_view_observed": observed_filter_view,
            "extra_input": "filter_off", "generic_packed": capture(_private_task(wire, bounded=False, filter_view_observed=observed_filter_view)),
            "explicit_bounded": capture(_private_task(wire, bounded=True, filter_view_observed=observed_filter_view))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/bounded-decision-evaluation.json"))
    parser.add_argument("--private-473", type=Path, help="Optional retained fixture; hash is checked but UI is never exported.")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False).stdout.strip()
    scenarios = ("simple", "multifield", "ambiguity", "missing", "hidden", "modal", "noise", "contradiction", "oversize")
    result = {"kind": "offline_mock_oracle_contract_evaluation", "source_revision": revision, "source_hashes": source_hashes(root),
              "limitations": ["No provider request was sent.", "Construction timing is not provider latency.", "Mock choices do not measure model accuracy."],
              "scenarios": {name: {"generic_packed": run(name, False), "explicit_bounded": run(name, True)} for name in scenarios}}
    if args.private_473:
        try:
            result["private_473"] = private_473(args.private_473)
        except ValueError as exc:
            parser.error(str(exc))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "source_revision": revision, "scenario_count": len(scenarios)}, sort_keys=True))


if __name__ == "__main__":
    main()
