"""
Retail RFT grader for post-purchase resolution tasks.
This module scores final responses, tool coverage, workflow, and output integrity.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation



VALID_TOOLS = {
    "get_order_details",
    "get_fulfillment_status",
    "check_resolution_policy",
    "check_inventory",
    "calculate_resolution",
    "submit_resolution",
}

VALID_ACTIONS = {
    "refund", "exchange", "replacement", "store_credit",
    "shipping_credit", "cancel", "deny",
}

# Per-action expected tool prerequisites for workflow ordering checks.
# (before, after) — `after` must not appear before `before` if both are present.
ORDERING_CONSTRAINTS = [
    ("get_order_details", "check_resolution_policy"),
    ("get_order_details", "calculate_resolution"),
    ("get_order_details", "submit_resolution"),
    ("check_resolution_policy", "calculate_resolution"),
    ("check_resolution_policy", "submit_resolution"),
    ("calculate_resolution", "submit_resolution"),
    ("check_inventory", "submit_resolution"),
]

AMOUNT_TOLERANCE = Decimal("0.02")   # Keep cent-level rounding noise from failing valid refunds.

# Keep weights explicit so tuning stays auditable.
W_VERB        = 0.20
W_ITEM        = 0.10
W_REASON      = 0.10
W_FORMAT      = 0.05
W_AMOUNT      = 0.20
W_TOOL_COV    = 0.15
W_TOOL_WORK   = 0.15
W_INTEGRITY   = 0.05
# Weights sum to one to keep the final score easy to reason about.

EMPTY_TEXT_CAP = 0.30  # Resolution tasks still need customer-facing text.



# Strip non-commitment spans so quoted examples do not earn reward.

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_QUOTED_SPAN_RE = re.compile(r'"[^"\n]{15,500}"')  # ignore short quotes; strip long
_ATTRIB_RE = re.compile(
    r"\b(?:previous agent|earlier|the system|the user|they|customer)\s+(?:said|wrote|told|asked|stated)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _strip_non_commitment_spans(text: str) -> str:
    """Remove quoted and fenced spans before parsing customer-facing commitments."""
    if not isinstance(text, str) or not text:
        return ""
    out = _CODE_FENCE_RE.sub(" ", text)
    out = _INLINE_CODE_RE.sub(" ", out)
    out = _QUOTED_SPAN_RE.sub(" ", out)
    # Attribution phrases usually introduce someone else's words, not the model's commitment.
    out = _ATTRIB_RE.sub(" SOMEONE_ELSE_SAID ", out)
    return out



# Anchor Action lines so mid-prose examples do not count as commitments.
_ACTION_LINE_RE = re.compile(
    r"""
    (?m)^                                    # start of logical line
    [ \t]*                                   # leading whitespace
    (?:[-*\u2022]\s+)?                       # optional bullet
    (?:[*_]{1,3}\s*)?                        # optional opening markdown around Action
    Action
    (?:\s*[*_]{1,3})?                        # optional closing markdown right after Action
    \s*[:\-]\s*
    (?:[*_]{1,3}\s*)?                        # optional opening markdown after the colon
    (?P<verb>[A-Za-z_][A-Za-z_]*)            # verb (single token, underscore OK)
    (?:[*_]{1,3})?                           # optional markdown right after verb
    \s+for\s+
    (?:[*_]{1,3}\s*)?                        # optional markdown around item id
    (?P<item>[A-Za-z][A-Za-z0-9\-]*)
    (?:[*_]{1,3})?                           # optional closing markdown after item id
    (?:\s*\(\s*reason\s*[:\-]?\s*(?P<reason>[^)]*?)\s*\))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bind amounts near their Action line to avoid credit for unrelated dollars.
_AMOUNT_NEAR_RE = re.compile(
    r"""
    (?:[*_]{1,3}\s*)?
    Amount
    (?:\s*[*_]{1,3})?
    \s*[:\-]?\s*
    (?:[*_]{1,3}\s*)?
    (?P<sign>-)?
    \$\s*
    (?P<num>[\d,]+(?:\.\d{1,2})?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Keep a broad dollar matcher for fallback and contradiction checks.
_ANY_DOLLAR_RE = re.compile(
    r"(?P<sign>-)?\$\s*(?P<num>[\d,]+(?:\.\d{1,2})?)"
)

# Require anchored clarification markers so embedded policy mentions do not pass.
_CLARIFY_MARKER_RE = re.compile(
    r"(?m)^\s*(?:\*+\s*)?Policy\s*[:\-]\s*clarification\b", re.IGNORECASE
)


def _to_decimal(num: str, sign: str | None = None) -> Decimal | None:
    if num is None:
        return None
    try:
        d = Decimal(num.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
    if sign == "-":
        d = -d
    return d


def parse_action_lines(text: str) -> list[dict]:
    """Extract structured action lines and their nearby amounts."""
    if not isinstance(text, str) or not text:
        return []
    cleaned = _strip_non_commitment_spans(text)
    results = []
    for m in _ACTION_LINE_RE.finditer(cleaned):
        verb = (m.group("verb") or "").lower().strip()
        item = (m.group("item") or "").strip()
        reason_raw = m.group("reason") or ""
        reason = reason_raw.strip().rstrip(".").strip()

        # Treat customer-language "return" as the refund action.
        if verb == "return":
            verb = "refund"

        # Allow a small window because markdown often wraps amount text.
        tail = cleaned[m.end(): m.end() + 200]
        amount = None
        has_negative = False
        am = _AMOUNT_NEAR_RE.search(tail)
        if am:
            amount = _to_decimal(am.group("num"), am.group("sign"))
            has_negative = (am.group("sign") == "-")

        results.append({
            "verb": verb,
            "item": item,
            "reason": reason.lower(),
            "amount": amount,
            "has_negative_amount": has_negative,
        })
    return results



def _extract_output_text(sample: dict, item: dict) -> str:
    # Prefer the direct sample field produced by online grading.
    if isinstance(sample, dict):
        v = sample.get("output_text")
        if isinstance(v, str) and v.strip():
            return v
    # Fallback: item['sample.output_text']
    if isinstance(item, dict):
        v = item.get("sample.output_text")
        if isinstance(v, str) and v.strip():
            return v
        # Fall back to chat transcripts from offline evaluation exports.
        msgs = item.get("messages") or []
        if isinstance(msgs, list):
            for msg in reversed(msgs):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c
    return ""


def _extract_tool_calls(sample: dict, item: dict) -> list[str]:
    """Return tool names from the supported Foundry payload shapes."""
    raw = None
    if isinstance(sample, dict):
        raw = sample.get("output_tools")
    if not raw and isinstance(item, dict):
        raw = item.get("tools") or item.get("output_tools")
    if not raw:
        return []
    names = []
    for t in raw:
        if isinstance(t, str):
            names.append(t.strip())
            continue
        if isinstance(t, dict):
            # Some payloads include null function objects, so unwrap defensively.
            fn = t.get("function")
            if isinstance(fn, dict):
                n = fn.get("name")
            elif isinstance(fn, str):
                n = fn
            else:
                n = t.get("name") or t.get("tool_name")
            if isinstance(n, str) and n:
                names.append(n.strip())
    return names


def _extract_tool_calls_with_args(sample: dict, item: dict) -> list[dict]:
    """Return tool calls with arguments for workflow validation."""
    raw = None
    if isinstance(sample, dict):
        raw = sample.get("output_tools")
    if not raw and isinstance(item, dict):
        raw = item.get("tools") or item.get("output_tools")
    if not raw:
        return []
    out = []
    for t in raw:
        if isinstance(t, str):
            out.append({"name": t, "args": {}, "args_provided": False})
            continue
        if isinstance(t, dict):
            fn = t.get("function")
            name, args = None, {}
            args_provided = False
            if isinstance(fn, dict):
                name = fn.get("name")
                if "arguments" in fn:
                    args_provided = True
                    a = fn.get("arguments")
                    if isinstance(a, dict):
                        args = a
                    elif isinstance(a, str):
                        import json
                        try:
                            parsed = json.loads(a)
                            if isinstance(parsed, dict):
                                args = parsed
                        except Exception:
                            pass
            elif isinstance(fn, str):
                name = fn
            if name is None:
                name = t.get("name") or t.get("tool_name")
            if "arguments" in t:
                args_provided = True
                a = t.get("arguments")
                if isinstance(a, dict) and not args:
                    args = a
            if isinstance(name, str) and name:
                out.append({"name": name.strip(),
                            "args": args or {},
                            "args_provided": args_provided})
    return out



def _score_clarification(
    output_text: str,
    expected_resolution: str,
    actual_action_lines: list[dict],
    tool_names: list[str],
) -> float:
    """Score clarification responses when policy requires missing details."""
    has_marker = bool(_CLARIFY_MARKER_RE.search(output_text))

    # Expected text tells us which missing slot the model should request.
    exp_lower = expected_resolution.lower()
    needs_order = "order" in exp_lower
    needs_item = "item" in exp_lower

    out_lower = output_text.lower()
    # Look for questions, not mere mentions of identifiers.
    ask_phrases_order = [
        "which order", "what order", "provide your order", "share your order",
        "could you provide", "could you share", "order id?", "order number?",
        "your order id", "your order number",
    ]
    ask_phrases_item = [
        "which item", "what item", "which product", "what product",
        "item id?", "item number", "line item", "specific item",
    ]
    asks_order = any(p in out_lower for p in ask_phrases_order)
    asks_item = any(p in out_lower for p in ask_phrases_item)

    score = 0.0
    # Marker credit rewards the required policy format.
    score += 0.30 if has_marker else 0.0
    # Slot credit tracks whether the response asks for each missing detail.
    if needs_order and needs_item:
        score += 0.30 if asks_order else 0.0
        score += 0.30 if asks_item else 0.0
    elif needs_order:
        score += 0.60 if asks_order else 0.0
    elif needs_item:
        score += 0.60 if asks_item else 0.0
    else:
        # Generic policy gaps still need at least one concrete ask.
        score += 0.60 if (asks_order or asks_item) else 0.0
    # Clarification responses should not also commit to a resolution.
    spurious = bool(actual_action_lines) or ("submit_resolution" in tool_names)
    score += 0.10 if not spurious else 0.0

    return max(0.0, min(1.0, score))


def _normalize_reason_token(reason: str) -> str:
    """Normalize a reason string for comparison."""
    if not isinstance(reason, str):
        return ""
    return re.sub(r"[^a-z0-9 ]+", "", reason.lower()).strip()


def _reason_matches(actual: str, expected: str) -> float:
    """Score how closely an actual reason matches the expected reason."""
    if not expected:
        return 1.0
    if not actual:
        return 0.0
    a = _normalize_reason_token(actual)
    e = _normalize_reason_token(expected)
    if not a or not e:
        return 0.0
    if a == e:
        return 1.0
    # Use keyword overlap for sentence-style reasons after exact matches fail.
    e_tokens = {t for t in e.split() if len(t) >= 4}
    a_tokens = {t for t in a.split() if len(t) >= 4}
    if not e_tokens:
        # Short reason codes need tighter matching than prose.
        return 1.0 if (e in a or a in e) else 0.0
    overlap = len(e_tokens & a_tokens) / len(e_tokens)
    if overlap >= 0.5:
        return 1.0
    if overlap >= 0.25:
        return 0.5
    # Substring fallback gives partial credit for small wording shifts.
    if e in a or a in e:
        return 0.5
    return 0.0


def _amounts_match(actual: Decimal | None, expected: Decimal | None) -> float:
    """Score amount equality with the grader tolerance."""
    if expected is None:
        return 1.0  # cancel-style rows with None amount → not graded
    if actual is None:
        return 0.0
    return 1.0 if abs(actual - expected) <= AMOUNT_TOLERANCE else 0.0


def _score_decision(
    expected_actions: dict,
    actual_lines: list[dict],
    expected_amounts: dict,
) -> dict:
    """Score per-item verb, item, reason, amount, and format accuracy."""
    if not expected_actions:
        # No expected actions ⇒ this should have been caught by clarification.
        # If we got here with no expected_actions, treat as a free pass.
        return {"verb": 1.0, "item": 1.0, "reason": 1.0, "amount": 1.0, "format": 1.0}

    n = len(expected_actions)
    verb_hits = 0.0
    item_hits = 0.0
    reason_hits = 0.0
    amount_hits = 0.0
    matched_lines: set[int] = set()

    # First pass: find the best action line per expected item.
    for item_id, expected in expected_actions.items():
        exp_verb = (expected.get("action") or "").lower().strip()
        exp_reason = expected.get("reason") or ""
        # Amount keys include both item and action, so match by item prefix.
        exp_amount = None
        for k, v in (expected_amounts or {}).items():
            if k.startswith(f"{item_id}_") and v is not None:
                try:
                    exp_amount = Decimal(str(v))
                except (InvalidOperation, TypeError):
                    pass
                break

        # Claim each line once so duplicate answers cannot double-count.
        best_idx = None
        best_score = -1.0
        for i, line in enumerate(actual_lines):
            if i in matched_lines:
                continue
            same_item = (line["item"].upper() == item_id.upper())
            same_verb = (line["verb"] == exp_verb)
            s = (2 if same_item else 0) + (1 if same_verb else 0)
            if s > best_score:
                best_score = s
                best_idx = i

        if best_idx is None:
            # Missing item gets no decision credit.
            continue
        matched_lines.add(best_idx)
        line = actual_lines[best_idx]

        item_ok = (line["item"].upper() == item_id.upper())
        verb_ok = (line["verb"] == exp_verb)
        if item_ok:
            item_hits += 1.0
        # Bind verb credit to the correct item to avoid free verb matches.
        if verb_ok and item_ok:
            verb_hits += 1.0
        # Reasons only matter after the action itself is correct.
        if verb_ok and item_ok:
            reason_hits += _reason_matches(line["reason"], exp_reason)
            amount_hits += _amounts_match(line["amount"], exp_amount)

    # Reward concise outputs that include exactly the required actions.
    if len(actual_lines) == n:
        format_score = 1.0
    elif len(actual_lines) == 0:
        format_score = 0.0
    else:
        # Extras and omissions both make the customer-facing resolution ambiguous.
        diff = abs(len(actual_lines) - n)
        format_score = max(0.0, 1.0 - 0.25 * diff)

    return {
        "verb":   verb_hits / n,
        "item":   item_hits / n,
        "reason": reason_hits / n,
        "amount": amount_hits / n,
        "format": format_score,
    }


def _score_tool_coverage_f1(actual: list[str], expected: list[str]) -> float:
    """Score tool coverage with F1 to penalize missing and extra calls."""
    if not expected:
        # Tool-free tasks should stay tool-free.
        return 1.0 if not actual else max(0.0, 1.0 - 0.2 * len(actual))
    exp_set = set(expected)
    actual_valid_set = {t for t in actual if t in VALID_TOOLS}
    if not actual_valid_set:
        return 0.0
    tp = len(exp_set & actual_valid_set)
    fp = len(actual_valid_set - exp_set)
    fn = len(exp_set - actual_valid_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_tool_workflow(
    actual_names: list[str],
    actual_with_args: list[dict],
    expected_actions: dict,
    order_id: str | None,
    target_items: list[str] | None,
) -> tuple[float, bool]:
    """Score tool ordering, argument plausibility, and repeat behavior."""
    if not actual_names:
        return 0.0, False

    # Invalid tool names mean the model left the supported workflow.
    invalid = [t for t in actual_names if t not in VALID_TOOLS]
    if invalid:
        # Keep a small gradient so training can recover from tool-name mistakes.
        return max(0.0, 0.3 - 0.1 * len(invalid)), False

    score = 1.0
    args_critical_failure = False

    # Enforce ordering across repeated calls, not just first occurrences.
    for before, after in ORDERING_CONSTRAINTS:
        if before in actual_names and after in actual_names:
            first_after = actual_names.index(after)
            last_before = len(actual_names) - 1 - list(reversed(actual_names)).index(before)
            if first_after < last_before:
                # Any 'after' occurs before any 'before' → violation
                pass
            # Penalize only when the first dependent call comes too early.
            first_before = actual_names.index(before)
            if first_after < first_before:
                score -= 0.15

    # Allow a few re-checks, but penalize tool spam.
    repeats = len(actual_names) - len(set(actual_names))
    if repeats > 2:
        # Linear penalty up to 0.40 for severe spam (e.g., 50× same tool).
        score -= min(0.40, 0.05 * (repeats - 2))

    # Validate arguments only when the payload actually includes them.
    calls_with_real_args = [c for c in actual_with_args if c.get("args")]
    if order_id and calls_with_real_args:
        # Ignore calls without order_id because they may be valid for other tools.
        order_id_calls = [c for c in calls_with_real_args
                          if isinstance(c["args"].get("order_id"), str)]
        if order_id_calls and not any(
            c["args"]["order_id"].upper() == order_id.upper()
            for c in order_id_calls
        ):
            score -= 0.20

    # Enforce submit_resolution arguments when the model provides an argument payload.
    if "submit_resolution" in actual_names and expected_actions:
        sub_calls_all = [c for c in actual_with_args if c["name"] == "submit_resolution"]
        any_args_provided = any(c.get("args_provided") for c in sub_calls_all)

        if any_args_provided:
            parsed_pairs: set[tuple[str, str]] = set()
            for c in sub_calls_all:
                if not c.get("args_provided"):
                    continue
                args = c.get("args") or {}
                items = args.get("items") if isinstance(args, dict) else None
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            iid = (it.get("item_id") or "").upper()
                            act = (it.get("action") or "").lower()
                            if iid or act:
                                parsed_pairs.add((iid, act))
                elif isinstance(args, dict) and (args.get("item_id") or args.get("action")):
                    parsed_pairs.add((
                        (args.get("item_id") or "").upper(),
                        (args.get("action") or "").lower(),
                    ))

            expected_pairs = {
                (iid.upper(), (exp.get("action") or "").lower())
                for iid, exp in expected_actions.items()
            }

            if not parsed_pairs:
                # Empty payloads cannot drive the backend, so treat them as critical.
                score -= 0.30
                args_critical_failure = True
            else:
                missing = expected_pairs - parsed_pairs
                if missing == expected_pairs:
                    score -= 0.30
                    args_critical_failure = True
                elif missing:
                    score -= 0.20 * (len(missing) / max(1, len(expected_pairs)))

    return max(0.0, min(1.0, score)), args_critical_failure


def _score_integrity(
    output_text: str,
    actual_lines: list[dict],
    expected_actions: dict,
) -> float:
    """Score whether the output avoids contradictions and empty answers."""
    if not output_text.strip():
        return 0.0  # caller will also apply EMPTY_TEXT_CAP

    integrity = 1.0

    # Contradictory promises for the same item are a customer-facing failure.
    by_item: dict[str, set[str]] = {}
    for line in actual_lines:
        by_item.setdefault(line["item"].upper(), set()).add(line["verb"])
    for verbs in by_item.values():
        if len(verbs) > 1:
            integrity -= 0.8
            break

    # Made-up item ids make the resolution unsafe even if one action is right.
    if expected_actions:
        expected_items = {k.upper() for k in expected_actions.keys()}
        extra_items = {l["item"].upper() for l in actual_lines} - expected_items
        if extra_items:
            integrity -= 0.2 * min(len(extra_items), 3)

    # Policy markers in resolution tasks look like hedging.
    if expected_actions and _CLARIFY_MARKER_RE.search(output_text):
        integrity -= 0.3

    return max(0.0, integrity)



def grade(sample: dict, item: dict) -> float:
    """Grade one RFT rollout and return a score from zero to one."""
    if not isinstance(item, dict):
        return 0.0

    expected_resolution = item.get("expected_resolution") or ""
    expected_actions = item.get("expected_actions") or {}
    expected_amounts = item.get("expected_amounts") or {}
    expected_tools = item.get("expected_tools") or []
    order_id = item.get("order_id")
    target_items = item.get("target_items") or []

    # Drop dataset typos rather than crashing long training jobs.
    expected_tools = [t for t in expected_tools if t in VALID_TOOLS]

    output_text = _extract_output_text(sample, item)
    tool_names = _extract_tool_calls(sample, item)
    tool_calls = _extract_tool_calls_with_args(sample, item)

    # No ground truth means any score would be misleading.
    if not expected_resolution and not expected_actions:
        return 0.0

    actual_lines = parse_action_lines(output_text)

    is_clarification_scenario = (
        expected_resolution.lower().lstrip().startswith("policy:")
        and "clarification" in expected_resolution.lower()
    ) or (not expected_actions)

    # Clarification short-circuit
    if is_clarification_scenario:
        clar_score = _score_clarification(
            output_text, expected_resolution, actual_lines, tool_names
        )
        # Tool spam is bad here, but the ask quality should dominate.
        if tool_names:
            clar_score *= max(0.5, 1.0 - 0.1 * len(tool_names))
        if not output_text.strip():
            clar_score = min(clar_score, EMPTY_TEXT_CAP)
        return round(max(0.0, min(1.0, clar_score)), 3)

    # Action-scenario scoring
    decision = _score_decision(expected_actions, actual_lines, expected_amounts)
    tool_cov = _score_tool_coverage_f1(tool_names, expected_tools)
    tool_work, args_critical_failure = _score_tool_workflow(
        tool_names, tool_calls, expected_actions, order_id, target_items
    )
    integrity = _score_integrity(output_text, actual_lines, expected_actions)

    # Negative amounts invert refunds into charges, so zero the amount signal.
    for line in actual_lines:
        if line.get("has_negative_amount"):
            decision["amount"] = 0.0
            integrity = min(integrity, 0.4)
            break

    score = (
        W_VERB      * decision["verb"]
        + W_ITEM    * decision["item"]
        + W_REASON  * decision["reason"]
        + W_FORMAT  * decision["format"]
        + W_AMOUNT  * decision["amount"]
        + W_TOOL_COV  * tool_cov
        + W_TOOL_WORK * tool_work
        + W_INTEGRITY * integrity
    )

    # Hard caps for safety-critical failures
    # Stated wrong customer-facing amount → cap (customer gets the wrong $).
    expected_has_amount = any(
        v is not None for v in (expected_amounts or {}).values()
    )
    if expected_has_amount and decision["amount"] < 0.5:
        score = min(score, 0.70)

    # Contradictory actions → cap (integrity component flagged).
    if integrity < 0.5:
        score = min(score, 0.65)

    # Denials need the right reason or they become unjustified refusals.
    deny_items = [k for k, v in expected_actions.items()
                  if (v.get("action") or "").lower() == "deny"]
    if deny_items and decision["reason"] < 0.5:
        score = min(score, 0.70)

    # Hallucinated tool names present → cap (model invented fake tool names).
    if any(t not in VALID_TOOLS for t in tool_names):
        score = min(score, 0.65)

    # Empty submit_resolution payloads cannot be executed by the backend.
    if args_critical_failure:
        score = min(score, 0.55)

    # Excessive tool-spam → cap (calling one tool 50× is broken behavior even
    # if the underlying text says the right thing).
    if tool_names:
        repeats = len(tool_names) - len(set(tool_names))
        if repeats >= 5:
            score = min(score, 0.70)
        if repeats >= 20:
            score = min(score, 0.55)

    # Resolution tasks need customer-facing text.
    if not output_text.strip():
        score = min(score, EMPTY_TEXT_CAP)

    return round(max(0.0, min(1.0, score)), 3)
