# Explicit bounded task decisions

`Subtask.decision_contract` opts a task into exact control binding, typed intents,
and local verification predicates. The caller supplies the contract. The policy
does not infer it from prose. Tasks without the field keep the existing packed
`ChoicePolicy` behavior, including its conservative partial-state completion gate.

The provider answers one `next_action` choice among complete proposals. Each
proposal binds an operation, observed target ID and supplied input key. An answer
cannot mix the target of one proposal with another proposal's value. The executor
still materializes the selected action against the original snapshot and semantic
guard, checks freshness, executes one operation and observes again.

## Caller contract

```python
task = Subtask(
    goal="Set Search and turn the filter toggle off",
    verification=("Browser is visible", "The filter toggle is off", "Search equals the supplied query"),
    inputs={"query": "arc fixture", "filter_off": 0},
    constraints=("Keep transport stopped", "Do not load a browser result"),
    decision_contract={
        "version": 1,
        "bindings": {
            "browser": {"role": "Group", "name": "Browser", "source": "macos_ax"},
            "search": {"role": "TextField", "name": "Search", "parent": "browser"},
            "filter": {"role": "CheckBox", "name": "Show Filter View", "parent": "browser"},
        },
        "intents": [
            {"id": "set_search", "kind": "set_text", "binding": "search", "input_key": "query"},
            {"id": "turn_filter_off", "kind": "ensure_toggle", "binding": "filter", "input_key": "filter_off"},
        ],
        "verification": [
            {"criterion_index": 0, "all": [{"binding": "browser", "field": "visible", "value": True}]},
            {"criterion_index": 1, "all": [{"binding": "filter", "field": "value", "input_key": "filter_off"}]},
            {"criterion_index": 2, "all": [{"binding": "search", "field": "value", "input_key": "query"}]},
        ],
    },
)
```

Bindings use the full exact `role` and `name`, optional exact `source`, and optional
direct parent binding. Declaration order does not matter. Missing or ambiguous
bindings never select an arbitrary first match. A typed intent with an unresolved,
hidden, disabled or unsupported target hands off before provider spend. Duplicate
observed IDs, cyclic ancestry, conflicting intents and active modal contexts
outside the target's ancestry also cause a handoff.

The supported intents are `set_text`, `set_value` and `ensure_toggle`. Inputs are
caller-owned literals. Text intents require a string and a text-editable semantic
target; they do not append Enter or Tab. Value intents use existing advertised
SET_VALUE/type compatibility. Toggle intents require an observed checkbox/switch
role, an advertised CLICK and a known binary current/desired state. Their explicit
binary semantics accept booleans or integers 0/1. Other scalar comparisons remain
type-aware. Intents already in their desired state produce no action proposal.

The schema rejects unknown fields and invalid references. It permits up to 32
bindings, 32 intents, 32 criterion entries and 8 predicates per criterion. Names
are bounded to 4096 characters and identifiers/roles/sources to 128. Parsed objects
are immutable copies. Oversized retained values still encounter both provider
request bounds before transport; no essential string is silently shortened.

## Verification and completion

Each predicate names one binding and a field: `value`, `visible`, `enabled`,
`selected` or `expanded`. It supplies exactly one literal `value` or an `input_key`.
State flags require a boolean expected value. Conjunctions become SATISFIED only
when every predicate is observed and true. A known false conjunct yields
NOT_SATISFIED; missing fields, type mismatches and unresolved observations yield
UNKNOWN. A criterion with no declared predicate mapping is UNKNOWN.

The caller is responsible for mapping its criterion faithfully. This contract
provides typed checks; it does not prove that arbitrary natural-language prose is
logically equivalent to those checks. If a criterion says both "toggle is off"
and "Filter View is hidden", it needs both observations. Checking the toggle
alone proves only its value. A hidden-view predicate must bind an actually
observed element with `visible=False`. Absence from a snapshot is UNKNOWN and
cannot prove hidden state. Global absence claims need an independently complete
observation scope and are not supported by this version.

SUBTASK_COMPLETE is offered only when every caller criterion is locally satisfied
and every intent is in its desired state. The provider still chooses the terminal
answer in the context of the caller's goal and free-form constraints. Local typed
checks do not enforce arbitrary natural-language constraints. If there are no
remaining proposals but verification is incomplete, the policy hands off without
HTTP. The returned local evidence is available in `Decision.raw.decision_contract`.
Dropping unrelated UI does not invalidate a fully covered explicit contract.

## Provider projection and ownership

The request includes named bindings, useful ancestors, full retained semantic
facts, local evidence, caller inputs/criteria/constraints, recent actions and
complete proposals. UI text is untrusted data. Secret values are redacted through
the existing policy. Guard bytes, coordinates and screenshots are not transmitted.
Backend-specific privacy filtering remains the host's responsibility. Selection
never prunes the trusted execution snapshot or remaps IDs.

The existing TypeSafe limits remain: 24,000 serialized UTF-8 bytes for state plus
the longest question and 48,000 for the complete request. These engineering limits
are not token counts. Essential overflow raises `ProviderContextUnrepresentable`
before HTTP; final transport accounting remains a second guard.

## Offline comparison

```bash
PYTHONPATH=src python -m pytest tests/test_bounded_decisions.py
PYTHONPATH=src python tools/evaluate_bounded_decisions.py --output /tmp/bounded-evaluation.json
```

The comparison uses synthetic observations and fake HTTP. It records source
hashes and request sizes for generic packed and explicit bounded paths, including
missing evidence, ambiguous bindings and modal interruption. A local retained
473-row replay can be supplied with `--private-473`; its SHA is verified and its
UI payload is never exported. Explicit contracts add caller-authored structure
and may add an input literal, so this is not a byte-identical task comparison.

Mock outcomes verify contract behavior. They do not establish model accuracy,
real provider latency, false-completion rate in production, or native application
acceptance. A separately authorized provider comparison should use frozen cases,
record actual usage/latency and selected proposals, and score joint target/value
correctness, false completion and unnecessary handoff against independent labels.
