"""
Reward shim for Retail RFT rollouts.
This module delegates scoring to the hardened grader and returns Slime metrics.
"""
from __future__ import annotations

import logging
from typing import Any

import retail_grader_rft_tools_v3 as grader

logger = logging.getLogger("retail-reward")

PASS_THRESHOLD = 0.80


def score_retail(
    final_response: str,
    expected_actions: dict,
    expected_amounts: dict,
    expected_resolution: str = "",
    *,
    expected_tools: list | None = None,
    order_id: str | None = None,
    target_items: list | None = None,
    tool_calls: list | None = None,
    n_tool_calls: int = 0,
    n_assistant_turns: int = 0,
    submitted_via_tool: bool = False,
) -> dict:
    """Score one trajectory with the Retail grader and return reward metrics."""
    sample = {
        "output_text": final_response or "",
        "output_tools": tool_calls or [],
    }
    item = {
        "expected_resolution": expected_resolution or "",
        "expected_actions": dict(expected_actions or {}),
        "expected_amounts": dict(expected_amounts or {}),
        "expected_tools": list(expected_tools or []),
        "order_id": order_id,
        "target_items": list(target_items or []),
    }

    try:
        score = float(grader.grade(sample, item))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"v3 grader raised: {exc}")
        score = 0.0

    # Keep lightweight diagnostics for dashboards without changing the score.
    try:
        parsed = grader.parse_action_lines(final_response or "")
    except Exception:  # noqa: BLE001
        parsed = []

    n_expected = len(expected_actions or {})
    return {
        "score": float(score),
        "binary_reward": float(score >= PASS_THRESHOLD),
        "pass_threshold": PASS_THRESHOLD,
        "grader_version": "v3",
        "n_parsed_actions": len(parsed),
        "n_expected_actions": n_expected,
        "n_tool_calls": int(n_tool_calls),
        "n_assistant_turns": int(n_assistant_turns),
        "submitted_via_tool": bool(submitted_via_tool),
        "tool_calls_emitted": [
            (tc.get("name") if isinstance(tc, dict) else str(tc))
            for tc in (tool_calls or [])
        ],
        "scenario_type": (
            "clarification"
            if (expected_resolution or "").lstrip().lower().startswith("policy:")
            or not expected_actions
            else "resolution"
        ),
    }


# Keep this compatibility shim for data-prep round trips.
parse_response = lambda text: {
    "actions": [
        {
            "item_id": p["item"],
            "action": p["verb"],
            "reason": p["reason"],
            "amount": float(p["amount"]) if p.get("amount") is not None else None,
        }
        for p in grader.parse_action_lines(text or "")
    ],
    "clarification": bool(grader._CLARIFY_MARKER_RE.search(text or "")),
}


async def custom_rm(args, sample, **kwargs):
    """Return the reward already attached by custom_generate."""
    if isinstance(getattr(sample, "reward", None), dict):
        return sample.reward
    return {"score": 0.0}
