"""Immutable caller-authored intent and observation contracts.

Predicates formalize the caller's criteria; this parser does not infer equivalence
between a predicate and arbitrary natural-language verification prose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

Scalar = str | int | float | bool


def scalar(value: Any) -> bool:
    return type(value) in (str, int, float, bool) and (type(value) is not float or math.isfinite(value))


def _object(value: Any, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError("invalid decision_contract object or unknown fields")
    return value


def _text(value: Any, limit: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("decision_contract strings must be nonempty and bounded")
    return value


@dataclass(frozen=True, slots=True)
class Binding:
    role: str
    name: str
    source: str | None = None
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class Intent:
    id: str
    kind: str
    binding: str
    input_key: str


@dataclass(frozen=True, slots=True)
class Predicate:
    binding: str
    field: str
    value: Scalar | None = None
    input_key: str | None = None


@dataclass(frozen=True, slots=True)
class Criterion:
    criterion_index: int
    predicates: tuple[Predicate, ...]


@dataclass(frozen=True, slots=True)
class DecisionContract:
    bindings: Mapping[str, Binding]
    intents: tuple[Intent, ...]
    verification: tuple[Criterion, ...]

    def compact(self) -> dict[str, Any]:
        return {
            "version": 1,
            "bindings": {
                key: {k: v for k, v in {
                    "role": item.role, "name": item.name, "source": item.source, "parent": item.parent,
                }.items() if v is not None}
                for key, item in self.bindings.items()
            },
            "intents": [
                {"id": item.id, "kind": item.kind, "binding": item.binding, "input_key": item.input_key}
                for item in self.intents
            ],
            "verification": [
                {"criterion_index": item.criterion_index, "all": [
                    {"binding": p.binding, "field": p.field,
                     **({"input_key": p.input_key} if p.input_key is not None else {"value": p.value})}
                    for p in item.predicates
                ]}
                for item in self.verification
            ],
        }


def parse_contract(
    raw: Mapping[str, Any] | DecisionContract, inputs: Mapping[str, Scalar], verification_count: int,
) -> DecisionContract:
    # Reparse typed objects too: dataclasses.replace must validate against new inputs/criteria.
    if isinstance(raw, DecisionContract):
        raw = raw.compact()
    raw = _object(raw, {"version", "bindings", "intents", "verification"})
    if type(raw.get("version")) is not int or raw["version"] != 1:
        raise ValueError("decision_contract.version must be 1")
    bindings_raw, intents_raw, checks_raw = raw.get("bindings"), raw.get("intents"), raw.get("verification", [])
    if not isinstance(bindings_raw, Mapping) or not 0 < len(bindings_raw) <= 32:
        raise ValueError("decision_contract needs 1..32 bindings")
    if not isinstance(intents_raw, list) or not 0 < len(intents_raw) <= 32:
        raise ValueError("decision_contract needs 1..32 intents")
    if not isinstance(checks_raw, list) or len(checks_raw) > 32:
        raise ValueError("decision_contract permits up to 32 criterion entries")
    bindings: dict[str, Binding] = {}
    for key, value in bindings_raw.items():
        key = _text(key)
        value = _object(value, {"role", "name", "source", "parent"})
        bindings[key] = Binding(
            _text(value.get("role")), _text(value.get("name"), 4096),
            _text(value["source"]) if "source" in value else None,
            _text(value["parent"]) if "parent" in value else None,
        )
    for key, binding in bindings.items():
        seen, parent = {key}, binding.parent
        while parent is not None:
            if parent not in bindings or parent in seen:
                raise ValueError("missing or cyclic binding parent")
            seen.add(parent)
            parent = bindings[parent].parent
    intents, ids = [], set()
    for value in intents_raw:
        value = _object(value, {"id", "kind", "binding", "input_key"})
        ident, kind, binding, input_key = (_text(value.get(k)) for k in ("id", "kind", "binding", "input_key"))
        if ident in ids or kind not in {"set_text", "set_value", "ensure_toggle"}:
            raise ValueError("duplicate intent id or unsupported intent kind")
        if binding not in bindings or input_key not in inputs:
            raise ValueError("intent references unknown binding or input")
        ids.add(ident)
        intents.append(Intent(ident, kind, binding, input_key))
    criteria, indexes = [], set()
    for value in checks_raw:
        value = _object(value, {"criterion_index", "all"})
        index, predicates = value.get("criterion_index"), value.get("all")
        if type(index) is not int or not 0 <= index < verification_count or index in indexes:
            raise ValueError("invalid or duplicate criterion index")
        if not isinstance(predicates, list) or not 0 < len(predicates) <= 8:
            raise ValueError("criterion needs 1..8 predicates")
        indexes.add(index)
        parsed = []
        for predicate in predicates:
            predicate = _object(predicate, {"binding", "field", "value", "input_key"})
            binding, field = _text(predicate.get("binding")), _text(predicate.get("field"))
            if binding not in bindings or field not in {"value", "visible", "enabled", "selected", "expanded"}:
                raise ValueError("invalid predicate binding or field")
            if ("value" in predicate) == ("input_key" in predicate):
                raise ValueError("predicate needs exactly one value or input_key")
            input_key = _text(predicate["input_key"]) if "input_key" in predicate else None
            if input_key is not None and input_key not in inputs:
                raise ValueError("predicate references unknown input")
            expected = inputs[input_key] if input_key is not None else predicate["value"]
            if not scalar(expected) or (field != "value" and type(expected) is not bool):
                raise ValueError("predicate needs a finite scalar; state flags require bool")
            parsed.append(Predicate(binding, field, None if input_key is not None else expected, input_key))
        criteria.append(Criterion(index, tuple(parsed)))
    return DecisionContract(MappingProxyType(bindings), tuple(intents), tuple(criteria))
