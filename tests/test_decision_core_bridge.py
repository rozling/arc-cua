"""Golden real-builder migration and provider-boundary guards, without inference."""
from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

pytest.importorskip("decision_core", reason="optional decision-core extra is not installed")

from decision_core import (
    CallContext,
    DecisionError,
    ProviderProfile,
    normalize_systemone,
    prepare_systemone,
)

from arc_cua.models import ActionKind, DesktopElement, DesktopSnapshot, Subtask, TerminalKind
from arc_cua.policies import ChoicePolicy, ProviderContextUnrepresentable, TypeSafeTransport
from arc_cua.policies.decision_core import DecisionCorePolicy, DecisionCoreTransport, DecisionProviderFailure
from arc_cua.validation import materialize_action

FIXTURE = json.loads((Path(__file__).parent / "fixtures/decision-core-consumer-cases.json").read_text())
CASES = {case["id"]: case for case in FIXTURE["cases"]}


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def profile(**changes):
    base = ProviderProfile("fixture", "jev-latest", "synthetic-jev", "synthetic", "synthetic",
                           "external", frozenset({"choice"}), 1000, 10, 100, 48000, 1000000,
                           24000, min_choices=1)
    return replace(base, **changes)


class NoAllowance:
    def reserve(self, **kwargs):
        raise AssertionError("offline replay must not reserve an attempt")


def context(ident="operation-a"):
    return CallContext(ident, ident + "-binding", time.monotonic() + 30,
                       frozenset({"fixture"}), NoAllowance())


class ReplayProvider:
    def __init__(self, case, selected_profile=None):
        self.case = case
        self.profile = selected_profile or profile()
        self.calls = []
        self.response = copy.deepcopy(case.get("core_response"))

    def evaluate(self, request, *, context):
        prepared = prepare_systemone(request, self.profile)
        self.calls.append((prepared.body, context))
        return normalize_systemone(prepared, encode(self.response))


def inputs_for(ident):
    buttons = [DesktopElement("apply", "button", "Apply label", actions=(ActionKind.CLICK,), source="synthetic"),
               DesktopElement("cancel", "button", "Cancel", actions=(ActionKind.CLICK,), source="synthetic")]
    elements = buttons
    values = {}
    if ident == "singleton-target":
        elements = buttons[:1]
    elif ident == "input-selection":
        elements = [DesktopElement("label", "textfield", "Label", actions=(ActionKind.TYPE_TEXT,), source="synthetic")]
        values = {"label_value": "Café", "other_value": "Other"}
    elif ident == "oversized-evidence-packed":
        elements += [DesktopElement(f"noise-{n}", "label", f"Unrelated telemetry row {n} " + "x" * 100,
                                    source="synthetic") for n in range(400)]
    elif ident == "oversized-evidence-unrepresentable":
        elements = [DesktopElement("apply", "button", "Apply label " + "x" * 100000,
                                   actions=(ActionKind.CLICK,), source="synthetic")]
    task = Subtask(goal="Set synthetic label to Café", verification=("Synthetic label is Café",), inputs=values)
    snapshot = DesktopSnapshot(application="SyntheticFixture", window="Offline fixture", revision="synthetic-1",
                               elements=tuple(elements))
    return task, snapshot


def legacy(case):
    calls = []

    def respond(request):
        calls.append(request.content)
        return httpx.Response(200, json=case["response"])

    client = httpx.Client(transport=httpx.MockTransport(respond))
    transport = TypeSafeTransport(api_key="offline", model="jev-latest", client=client)
    policy = ChoicePolicy(transport, invalid_retries=0, provider_budget=(24000, 48000), request_model="jev-latest")
    return policy, calls, client


def relevant(decision):
    return {key: getattr(decision, key) for key in ("kind", "terminal", "target_id", "input_key")}


@pytest.mark.parametrize("ident", CASES)
def test_real_builder_both_paths_replay_pinned_goldens(ident):
    case = CASES[ident]
    assert case["origin"]["revision"] == "8acf83ff21ce3c8901f94b5f78b175d36b24afa9"
    task, snapshot = inputs_for(ident)
    old, calls, client = legacy(case)
    provider = ReplayProvider(case)
    new = DecisionCorePolicy(provider).bind(context())
    try:
        if case["request"] is None:
            for policy in (old, new):
                with pytest.raises(ProviderContextUnrepresentable):
                    policy.decide(subtask=task, snapshot=snapshot, history=())
            assert calls == provider.calls == []
            return
        previous = old.decide(subtask=task, snapshot=snapshot, history=())
        migrated = new.decide(subtask=task, snapshot=snapshot, history=())
        expected_wire = case["wire"]["utf8"].encode()
        assert calls == [expected_wire]
        assert [body for body, _ in provider.calls] == [expected_wire]
        assert relevant(previous) == relevant(migrated) == case["expected"]["decision"]
        assert (previous.confidence, previous.margin) == (migrated.confidence, migrated.margin)
        assert previous.raw["provider_packing"] == migrated.raw["provider_packing"]
        if migrated.kind:
            assert materialize_action(previous, snapshot, task) == materialize_action(migrated, snapshot, task)
            rows = migrated.raw["provider_packing"]
            assert rows["mandatory_elements"] >= 1
    finally:
        client.close()


def test_partial_projection_still_cannot_complete():
    case = copy.deepcopy(CASES["oversized-evidence-packed"])
    # Both paths must consume valid verification before rejecting partial completion.
    for key in ("response", "core_response"):
        for ident, choice in (("operation", "SUBTASK_COMPLETE"), ("verification_0", "SATISFIED")):
            answer = case[key]["answers"][ident]
            answer["choice"] = choice
            answer["probabilities"] = {candidate: float(candidate == choice) for candidate in answer["probabilities"]}
    task, snapshot = inputs_for(case["id"])
    old, _, client = legacy(case)
    try:
        for policy in (old, DecisionCorePolicy(ReplayProvider(case)).bind(context())):
            decision = policy.decide(subtask=task, snapshot=snapshot, history=())
            assert decision.terminal == TerminalKind.NEEDS_AGENT
            assert "partial provider projection" in decision.reason
    finally:
        client.close()


def test_unused_invalid_head_rejects_entire_batch_in_migrated_path():
    case = copy.deepcopy(CASES["ordinary-action"])
    for key in ("response", "core_response"):
        case[key]["answers"]["hotkey_value"]["confidence"] = -1
    task, snapshot = inputs_for(case["id"])
    old, _, client = legacy(case)
    provider = ReplayProvider(case)
    try:
        assert old.decide(subtask=task, snapshot=snapshot, history=()).kind == ActionKind.CLICK
        policy = DecisionCorePolicy(provider).bind(context())
        assert policy.invalid_retries == 0
        with pytest.raises(DecisionProviderFailure) as failure:
            policy.decide(subtask=task, snapshot=snapshot, history=())
        assert failure.value.code == "invalid_response"
        assert len(provider.calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("family", ["operation", "hotkey_value", "type_text_input", "click_target"])
def test_every_choice_family_obeys_profile_cap_without_truncation(family):
    case = CASES["ordinary-action"]
    provider = ReplayProvider(case, profile(max_choices=26))
    questions = {family: {"type": "choice", "criteria": {str(n): str(n) for n in range(27)},
                          "instructions": {"rules": "Preserve all original choices"}}}
    original = copy.deepcopy(questions)
    with pytest.raises(DecisionProviderFailure) as failure:
        DecisionCoreTransport(provider, context()).ask({"evidence": "synthetic"}, questions)
    assert failure.value.code == "unsupported"
    assert provider.calls == []
    assert questions == original


@pytest.mark.parametrize("changes,code", [
    ({"supported": frozenset({"binary"})}, "unsupported"),
    ({"max_questions": 1}, "unsupported"),
    ({"min_choices": 2}, "invalid_request"),
])
def test_actual_builder_capability_question_count_and_singleton_fail_before_provider(changes, code):
    provider = ReplayProvider(CASES["singleton-target"], profile(**changes))
    task, snapshot = inputs_for("singleton-target")
    with pytest.raises(DecisionProviderFailure) as failure:
        DecisionCorePolicy(provider).bind(context()).decide(subtask=task, snapshot=snapshot, history=())
    assert failure.value.code == code
    assert not provider.calls


def test_operation_contexts_are_local_during_concurrent_inflight_decisions():
    barrier = threading.Barrier(2)

    class ConcurrentProvider(ReplayProvider):
        def evaluate(self, request, *, context):
            barrier.wait(timeout=5)
            return super().evaluate(request, context=context)

    provider = ConcurrentProvider(CASES["ordinary-action"])
    factory = DecisionCorePolicy(provider)
    first, second = context("first"), context("second")
    policies = [factory.bind(first), factory.bind(second)]
    assert policies[0] is not policies[1]
    task, snapshot = inputs_for("ordinary-action")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(policy.decide, subtask=task, snapshot=snapshot, history=()) for policy in policies]
        assert all(f.result().kind == ActionKind.CLICK for f in futures)
    assert {id(ctx) for _, ctx in provider.calls} == {id(first), id(second)}
    assert {ctx.binding_token for _, ctx in provider.calls} == {"first-binding", "second-binding"}
    assert all(b"binding" not in body for body, _ in provider.calls)


def test_typed_failure_has_fixed_diagnostics_and_does_not_retry():
    class FailedProvider(ReplayProvider):
        def evaluate(self, request, *, context):
            self.calls.append(context)
            raise DecisionError("transport_error", "secret provider body")

    provider = FailedProvider(CASES["ordinary-action"])
    task, snapshot = inputs_for("ordinary-action")
    with pytest.raises(DecisionProviderFailure) as failure:
        DecisionCorePolicy(provider).bind(context()).decide(subtask=task, snapshot=snapshot, history=())
    assert failure.value.code == "transport_error"
    assert "secret" not in str(failure.value)
    assert len(provider.calls) == 1


@pytest.mark.parametrize("status,payload,code", [
    (503, b'{}', "http_error"),
    (200, b'{"model":"synthetic-jev","usage":{"input_tokens":0,"output_tokens":0},"answers":{}}', "invalid_response"),
])
def test_bounded_http_provider_attempt_and_receipt_survive_bridge_failure(monkeypatch, status, payload, code):
    from decision_core import CostBound, Endpoint, HTTPResponse, SystemOneHTTPProvider
    from decision_core import transport as core_transport

    calls = []
    reservations = []

    class Allowance:
        def reserve(self, **kwargs):
            reservations.append(kwargs)
            return "synthetic-reservation"

    def execute(wire, cap, tls_context, checkpoint, transmitted, responded):
        calls.append(wire)
        checkpoint()
        transmitted()
        responded()
        return HTTPResponse(status, payload)

    monkeypatch.setattr(core_transport, "_execute", execute)
    provider = SystemOneHTTPProvider(profile(locality="on_device"),
        Endpoint("http://127.0.0.1:11434/v1/systemone", "127.0.0.1"),
        CostBound("fixture", "synthetic", "1", "0"))
    bound = replace(context(), allowance=Allowance())
    task, snapshot = inputs_for("ordinary-action")
    with pytest.raises(DecisionProviderFailure) as failure:
        DecisionCorePolicy(provider).bind(bound).decide(subtask=task, snapshot=snapshot, history=())
    assert failure.value.code == code
    assert len(calls) == len(reservations) == 1
    assert failure.value.receipt.reservation_id == "synthetic-reservation"
    assert failure.value.receipt.delivery == "response_received"


def test_legacy_http_retry_remains_separate_from_bounded_attempt(monkeypatch):
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(503, json={})

    monkeypatch.setattr("arc_cua.policies.typesafe.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        transport = TypeSafeTransport(api_key="synthetic", client=client)
        with pytest.raises(RuntimeError):
            transport.ask({"synthetic": True}, CASES["ordinary-action"]["request"]["questions"])
    assert len(calls) == 3


def test_consumed_invalid_answer_retries_change_from_two_to_one():
    case = copy.deepcopy(CASES["ordinary-action"])
    for key in ("response", "core_response"):
        case[key]["answers"]["operation"]["confidence"] = -1
    task, snapshot = inputs_for(case["id"])
    old, calls, client = legacy(case)
    old.invalid_retries = 1
    provider = ReplayProvider(case)
    try:
        from arc_cua.policies import InvalidChoiceResponse
        with pytest.raises(InvalidChoiceResponse):
            old.decide(subtask=task, snapshot=snapshot, history=())
        with pytest.raises(DecisionProviderFailure):
            DecisionCorePolicy(provider).bind(context()).decide(subtask=task, snapshot=snapshot, history=())
        assert len(calls) == 2
        assert len(provider.calls) == 1
    finally:
        client.close()


def test_probability_sum_migration_is_explicit_without_changing_interpretation():
    case = copy.deepcopy(CASES["ordinary-action"])
    for key in ("response", "core_response"):
        case[key]["answers"]["operation"]["probabilities"]["CLICK"] = .99
    task, snapshot = inputs_for(case["id"])
    old, _, client = legacy(case)
    try:
        assert old.decide(subtask=task, snapshot=snapshot, history=()).kind == ActionKind.CLICK
        with pytest.raises(DecisionProviderFailure) as failure:
            DecisionCorePolicy(ReplayProvider(case)).bind(context()).decide(subtask=task, snapshot=snapshot, history=())
        assert failure.value.code == "invalid_response"
    finally:
        client.close()


@pytest.mark.parametrize("family", ["operations", "shortcuts", "inputs", "targets"])
def test_real_builder_profile_limits_reject_whole_family(family):
    task, snapshot = inputs_for("ordinary-action")
    cap = 26
    if family == "operations":
        cap = 6
    elif family == "shortcuts":
        task = replace(task, shortcuts={f"{modifier}+{letter}": "synthetic shortcut"
                                        for modifier in ("ALT", "SHIFT") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
    elif family == "inputs":
        task, snapshot = inputs_for("input-selection")
        task = replace(task, inputs={f"input-{n}": f"value-{n}" for n in range(30)})
    else:
        snapshot = replace(snapshot, elements=tuple(
            DesktopElement(f"target-{n}", "button", f"Apply label {n}", actions=(ActionKind.CLICK,))
            for n in range(30)))
    provider = ReplayProvider(CASES["ordinary-action"], profile(max_choices=cap))
    with pytest.raises(DecisionProviderFailure) as failure:
        DecisionCorePolicy(provider).bind(context()).decide(subtask=task, snapshot=snapshot, history=())
    assert failure.value.code == "unsupported"
    assert provider.calls == []
