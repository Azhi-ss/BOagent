"""Strict parser for Chem-LGBO sparse-subspace guidance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

Subspace = dict[str, list[str]]
ParseResult = tuple[Subspace | None, str]


def _extract_final_object(text: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    starts: list[int] = []
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"' and depth:
            in_string = True
        elif char == "{":
            if depth == 0:
                starts.append(index)
            depth += 1
        elif char == "}" and depth:
            depth -= 1

    for start in reversed(starts):
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[start + end :].strip():
            return value
    return None


def parse_subspace_response(
    text: str,
    feature_cols: Sequence[str],
    options: Mapping[str, Sequence[str]],
) -> ParseResult:
    """Parse the final JSON object and validate the sparse subspace exactly."""
    if not text.strip():
        return None, "empty_response"

    payload = _extract_final_object(text)
    if payload is None:
        return None, "invalid_json"
    if set(payload) != {"subspace"}:
        return None, "invalid_schema"

    raw_subspace = payload["subspace"]
    if not isinstance(raw_subspace, dict) or not raw_subspace:
        return None, "invalid_schema"

    allowed_fields = set(feature_cols)
    if any(field not in allowed_fields for field in raw_subspace):
        return None, "unknown_field"

    subspace: Subspace = {}
    for field, raw_values in raw_subspace.items():
        if not isinstance(raw_values, list):
            return None, "invalid_schema"
        if not raw_values:
            return None, "empty_choice"
        if any(not isinstance(value, str) for value in raw_values):
            return None, "invalid_schema"
        if len(set(raw_values)) != len(raw_values):
            return None, "duplicate_value"
        allowed_values = options.get(field, ())
        if any(value not in allowed_values for value in raw_values):
            return None, "unknown_value"
        subspace[field] = list(raw_values)

    return subspace, "accepted"
