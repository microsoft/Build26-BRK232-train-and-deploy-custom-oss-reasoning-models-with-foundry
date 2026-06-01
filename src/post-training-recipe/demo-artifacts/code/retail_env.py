"""
In-memory Retail environment for multi-turn RFT rollouts.
This module dispatches deterministic tools and records episode state.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import retail_tools  # Keep tool dispatch in-process for deterministic rollouts.

logger = logging.getLogger("retail-env")


class RetailEnv:
    """Main class for managing one Retail tool-use episode."""

    def __init__(self, scenario: dict[str, Any]):
        """Initialize an episode from sample metadata."""
        self.scenario = scenario
        self.expected_actions = dict(scenario.get("expected_actions") or {})
        self.expected_amounts = dict(scenario.get("expected_amounts") or {})
        self.expected_resolution = scenario.get("expected_resolution") or ""
        self.tools_called: list[str] = []
        self.terminated = False
        self.final_text: str | None = None
        # Keep prompts in Slime so the env only owns tool state.

    def reset(self) -> tuple[str, dict]:
        """Reset episode state and expose the tool schema."""
        self.tools_called = []
        self.terminated = False
        self.final_text = None
        info: dict[str, Any] = {
            "tools_schema": retail_tools.TOOL_SCHEMAS,
            "expected_actions": self.expected_actions,
            "expected_amounts": self.expected_amounts,
            "expected_resolution": self.expected_resolution,
            "scenario_id": self.scenario.get("scenario_id"),
        }
        # Return no observation because the user request is already in the prompt.
        return "", info

    def step(self, action_str: str) -> tuple[str, float, bool, bool, dict]:
        """Execute one agent action against the Retail tool layer."""
        if self.terminated:
            return "", 0.0, True, False, {}

        action_str = (action_str or "").strip()
        if not action_str:
            # Stop empty turns early so broken generations do not loop.
            self.terminated = True
            self.final_text = ""
            return "", 0.0, True, False, {}

        # Accept the same compact JSON action shape emitted by custom_generate.
        parsed: dict[str, Any] | None = None
        try:
            obj = json.loads(action_str)
            if isinstance(obj, dict) and "name" in obj:
                parsed = obj
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if parsed is None:
            # Plain text → agent's final answer. Episode ends.
            self.terminated = True
            self.final_text = action_str
            return "", 0.0, True, False, {"final_text": action_str}

        name = str(parsed.get("name") or "")
        args = parsed.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}

        fn = retail_tools.TOOL_FUNCTIONS.get(name)
        if fn is None:
            obs = json.dumps({"error": f"Unknown tool: {name}"})
            return obs, 0.0, False, False, {"tool_name": name, "tool_ok": False}

        try:
            result = fn(**args)
        except TypeError as e:
            result = json.dumps({"error": f"bad arguments for {name}: {e}"})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tool {name} raised: {e}")
            result = json.dumps({"error": f"{name} raised: {e}"})

        if not isinstance(result, str):
            try:
                result = json.dumps(result)
            except Exception:  # noqa: BLE001
                result = str(result)

        self.tools_called.append(name)
        # Let the model write a final text turn after submit_resolution so grading sees the customer answer.
        return result, 0.0, False, False, {"tool_name": name, "tool_ok": True}

    def close(self) -> None:
        return None


def make_env(scenario: dict[str, Any]) -> RetailEnv:
    return RetailEnv(scenario)
