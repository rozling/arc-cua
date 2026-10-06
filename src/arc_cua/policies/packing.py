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
    matched = element_words(element) & words
    if not matched:
        return False
    if not element.actions:
        return True
    label_words = _words(element.name.split("\n", 1)[0])
    # A concise label such as Search is strong; Browser Content Pane is not.
    return len(matched) >= 2 or (bool(label_words) and label_words <= words)


def rank(element: DesktopElement, words: set[str], index: int) -> tuple[int, int, int]:
    """Shared stable target ranking; callers serialize selected rows in observation order."""
    return (0 if element.focused or element.selected else 1, -len(element_words(element) & words), index)


def closure(elements: Iterable[DesktopElement], mandatory_ids: set[str], words: set[str]) -> set[str]:
    """Keep mandatory controls plus useful named ancestors, safely through empty wrappers."""
    rows = [element for element in elements if element.visible]
    by_id = {e.id: e for e in rows}
    kept = set(mandatory_ids)
    directly_relevant: set[str] = set()
    for element in rows:
        role = element.role.casefold()
        normalized_role = role.removeprefix("ax")
        represented_active = (any(token in role for token in ("modal", "dialog", "editor")) or
                              normalized_role in {"sheet", "popover", "menu"} or
                              element.metadata.get("active") is True or element.metadata.get("modal") is True or
                              element.metadata.get("editor") is True)
        if element.focused or element.selected or represented_active or relevant(element, words):
            kept.add(element.id)
            directly_relevant.add(element.id)
    # Retain compact non-actionable status/label evidence within a relevant
    # container, but never fan out automatically from an unnamed root.
    relation_roots = {"window", "root", "webarea", "application"}
    for element in rows:
        if element.actions or not element.parent_id or element.parent_id not in directly_relevant:
            continue
        parent = by_id.get(element.parent_id)
        if parent is not None and parent.role.casefold().removeprefix("ax") in relation_roots:
            continue
        role = element.role.casefold()
        if ("status" in role or "label" in role or "statictext" in role) and (element.name or element.value is not None):
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
