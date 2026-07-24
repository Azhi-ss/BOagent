"""Parser for the LGBO LLM Final Answer (JSON form of the paper's point/region+c).

The paper's Appendix B emits ``[point, [v1..vd], c]``. IUPAC option names contain
commas and brackets (e.g. ``3-phenyl-1,2-oxazole``,
``dicyclohexyl-[2-[2,4,6-tri(propan-2-yl)phenyl]phenyl]phosphane``), which make
that bracketed format unparseable by regex. The prompt therefore asks the LLM to
emit the Final Answer as a JSON object with identical semantics:

    {"mode": "point", "values": ["v1", ..., "vd"], "confidence": c}
    {"mode": "region", "lb": ["lb1", ...], "ub": ["ub1", ...], "confidence": c}

This module extracts the last JSON object in the response (the Final Answer,
after the Thinking block), validates it against the declared option space, and
returns ``(mode, values, confidence)`` or ``None`` on any failure (caller falls
back to a pure-GP step with lambda=0).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

ParsedProposal = Tuple[str, List[str], float]


def _extract_last_json_object(text: str) -> Optional[dict]:
    """Return the last valid JSON object occurring in ``text``, or None.

    Scans right-to-left for ``{`` and uses ``JSONDecoder.raw_decode`` so that a
    Thinking block containing brace-like text does not shadow the Final Answer.
    """
    decoder = json.JSONDecoder()
    for i in range(len(text) - 1, -1, -1):
        if text[i] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_llm_response(
    text: str,
    feature_cols: Sequence[str],
    options: Dict[str, Sequence[str]],
) -> Optional[ParsedProposal]:
    """Parse + validate the LLM Final Answer.

    Args:
        text: full LLM response (Thinking + Final Answer JSON).
        feature_cols: declared parameter order (d columns).
        options: per-column valid option lists (the sub4 options.json space).

    Returns:
        (mode, values, confidence) with ``mode`` in {"point","region"},
        ``values`` a length-d list of validated option strings, and
        ``confidence`` clamped to [0, 1]; or ``None`` if no valid proposal.
    """
    obj = _extract_last_json_object(text)
    if obj is None:
        return None

    mode = obj.get("mode")
    if mode not in ("point", "region"):
        return None

    # Confidence: must be a real number; clamp to [0, 1].
    c = obj.get("confidence")
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        return None
    confidence = max(0.0, min(1.0, float(c)))

    d = len(feature_cols)
    option_sets = {col: set(str(v) for v in options.get(col, [])) for col in feature_cols}

    if mode == "point":
        raw_values = obj.get("values")
        if not isinstance(raw_values, list) or len(raw_values) != d:
            return None
        values = [str(v) for v in raw_values]
    else:  # region: for fully-categorical, require lb[i] == ub[i] per dimension.
        lb = obj.get("lb")
        ub = obj.get("ub")
        if not isinstance(lb, list) or not isinstance(ub, list):
            return None
        if len(lb) != d or len(ub) != d:
            return None
        values = []
        for lo, hi in zip(lb, ub):
            if str(lo) != str(hi):
                # A range over categorical variables is not meaningful.
                return None
            values.append(str(lo))

    # Validate every value against its column's option set.
    for col, val in zip(feature_cols, values):
        if val not in option_sets[col]:
            return None

    return mode, values, confidence
