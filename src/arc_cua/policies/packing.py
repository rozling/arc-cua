"""Deterministic provider-context packing for choice policies."""

from __future__ import annotations

import re
from typing import Iterable

from ..models import DesktopElement, Subtask
_WORD = re.compile(r"[^\W_]{3,}", re.UNICODE)
_FUNCTION_WORDS = frozenset({"and", "are", "for", "from", "has", "have", "into", "its", "not", "that", "the", "this", "with"})


def _words(text: str) -> set[str]:
    return {word.casefold() for word in _WORD.findall(text)}


def task_words(subtask: Subtask) -> set[str]:
    return _words(" ".join((subtask.goal, *subtask.verification, *subtask.constraints,
        *(str(value) for key, value in subtask.inputs.items() if key not in subtask.secret_inputs)))) - _FUNCTION_WORDS


def element_words(element: DesktopElement) -> set[str]:
    # Long accessibility help attached to actionable labels must not dominate ranking.
    name = element.name.split("\n", 1)[0] if element.actions else element.name
    value = str(element.value) if element.value is not None else ""
    return _words(f"{name} {value}")


def relevant(element: DesktopElement, words: set[str]) -> bool:
    return bool(element_words(element) & words)


def closure(elements: Iterable[DesktopElement], mandatory_ids: set[str], words: set[str]) -> set[str]:
    """Keep mandatory controls plus useful named ancestors, safely through empty wrappers."""
    rows = list(elements)
    by_id = {e.id: e for e in rows}
    kept = set(mandatory_ids)
    for element in rows:
        role = element.role.casefold()
        represented_active = any(token in role for token in ("modal", "dialog", "editor"))
        if element.focused or element.selected or represented_active or relevant(element, words):
            kept.add(element.id)
    for element_id in tuple(kept):
        parent = by_id.get(element_id).parent_id if element_id in by_id else None
        seen: set[str] = set()
        while parent and parent not in seen:
            seen.add(parent)
            ancestor = by_id.get(parent)
            if ancestor is None:
                break
            if ancestor.name or ancestor.value is not None or ancestor.selected is not None or ancestor.expanded is not None:
                kept.add(parent)
            parent = ancestor.parent_id
    return kept
