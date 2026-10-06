import json

import httpx

from arc_cua.models import ActionKind, DesktopElement, DesktopSnapshot, Subtask, TerminalKind
from arc_cua.policies import TypeSafeJevPolicy


def _snapshot() -> DesktopSnapshot:
    rows = []
    for index in range(470):
        rows.append(DesktopElement(id=f"noise-{index}", role="Label", name=f"Unrelated verbose panel {index}"))
    rows.extend((
        DesktopElement(id="search", role="TextField", name="Search\nVery long help text that is not task evidence", value="",
                       actions=(ActionKind.CLICK, ActionKind.TYPE_TEXT), focused=True),
        DesktopElement(id="filter", role="Button", name="Show Filter View", value=False, actions=(ActionKind.CLICK,)),
        DesktopElement(id="status", role="Status", name="Browser status\nFilter View hidden", value=0),
    ))
    return DesktopSnapshot(application="App", window="Window", revision="1", elements=tuple(rows))


def test_typesafe_packs_large_state_coherently_and_deterministically() -> None:
    bodies = []
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        criteria = body["questions"]["operation"]["criteria"]
        probabilities = {key: 1.0 if key == "NEEDS_AGENT" else 0.0 for key in criteria}
        return httpx.Response(200, json={"answers": {"operation": {"choice": "NEEDS_AGENT", "confidence": 1, "probabilities": probabilities}}})
    task = Subtask(goal="Hide Filter View and set Search", verification=("Filter View hidden", "Search contains query"),
                   inputs={"query": "arc fixture"})
    for _ in range(2):
        policy = TypeSafeJevPolicy(api_key="test", model="jev-test", client=httpx.Client(transport=httpx.MockTransport(respond)))
        decision = policy.decide(subtask=task, snapshot=_snapshot(), history=())
        assert decision.terminal == TerminalKind.NEEDS_AGENT
    assert json.dumps(bodies[0], ensure_ascii=False, separators=(",", ":")).encode().__len__() <= 48_000
    state = bodies[0]["state"]
    assert state["provider_projection_partial"] is True
    rows = state["desktop"]["elements"]
    ids = {row[0] for row in rows}
    assert {"search", "filter", "status"} <= ids
    for question in bodies[0]["questions"].values():
        for choice in question["criteria"]:
            if choice in {"search", "filter"} or choice.startswith("noise-"):
                assert choice in ids
    assert bodies[0] == bodies[1]


def test_partial_projection_turns_claimed_completion_into_needs_agent() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        questions = json.loads(request.content)["questions"]
        choices = {}
        for key, question in questions.items():
            choice = "SUBTASK_COMPLETE" if key == "operation" else "SATISFIED"
            choices[key] = {"choice": choice, "confidence": 1,
                            "probabilities": {candidate: 1.0 if candidate == choice else 0.0 for candidate in question["criteria"]}}
            if key.startswith("verification_"):
                continue
        return httpx.Response(200, json={"answers": choices})
    policy = TypeSafeJevPolicy(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    decision = policy.decide(subtask=Subtask(goal="Find target", verification=("Target found",)), snapshot=_snapshot(), history=())
    assert decision.terminal == TerminalKind.NEEDS_AGENT
    assert decision.reason == "Completion cannot be established from a partial provider projection."
