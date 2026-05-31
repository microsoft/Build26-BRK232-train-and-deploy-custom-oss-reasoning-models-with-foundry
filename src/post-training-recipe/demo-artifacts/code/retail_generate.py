"""
Custom Retail rollout generator for Slime RFT (agent-style tool-calling).

This module runs one Retail multi-turn tool-use episode per Slime sample:
  - Renders the chat-template prompt + tool schemas
  - Asks SGLang for one assistant turn at a time
  - Parses tool calls (sglang native parser + XML/JSON/function fallbacks)
  - Executes tools locally via ``retail_tools.TOOL_FUNCTIONS``
  - Appends a ``tool`` message and continues until the model stops calling tools
  - Computes the Retail reward via ``retail_reward.score_retail``
  - Populates the metadata fields the rollout logger persists
    (``conversation_trace``, ``input_messages``, ``output_text``,
    ``output_tools``, ``tool_call_count``, etc.)
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
import time
from typing import Any

from retail_tools import TOOL_FUNCTIONS

import retail_reward

from slime.rollout.sglang_rollout import GenerateState  # type: ignore[reportMissingImports]
from slime.utils.http_utils import post  # type: ignore[reportMissingImports]
from slime.utils.types import Sample  # type: ignore[reportMissingImports]

try:
    from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - Older Slime images do not ship trace_utils.
    build_sglang_meta_trace_attrs = None
    trace_span = None


logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_TURNS = 10
DEFAULT_MAX_TRAJ_TOKENS = 32768
TOOL_STOP = "</tool_call>"
VALID_TOOL_NAMES = set(TOOL_FUNCTIONS)

TOOL_PROTOCOL = """
## Tool Calling Protocol
You can call exactly one Retail tool at a time. To call a tool, respond only with:
<tool_call>{"name":"tool_name","arguments":{"arg":"value"}}</tool_call>

After a call, the environment will append:
<tool_result name="tool_name">...</tool_result>

Then continue deciding whether another tool is needed. When the case is resolved,
stop calling tools and respond with the required Retail summary line only.
""".strip()

ARG_ALIASES = {
    "exchangeSku": "exchange_sku",
    "function_name": "name",
    "itemId": "item_id",
    "itemID": "item_id",
    "orderId": "order_id",
    "orderID": "order_id",
    "resolutionSummary": "resolution_summary",
    "tool_name": "name",
    "toolName": "name",
}


def _max_tool_turns() -> int:
    raw_value = (
        os.environ.get("RETAIL_AGENT_MAX_NUM_STEPS")
        or os.environ.get("RETAIL_AGENT_MAX_TOOL_TURNS")
        or os.environ.get("RETAIL_MAX_TURNS")
    )
    if not raw_value:
        return DEFAULT_MAX_TOOL_TURNS
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_TOOL_TURNS


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _set_status(sample: Any, name: str) -> None:
    status_cls = getattr(sample, "Status", None)
    if status_cls is None and hasattr(Sample, "Status"):
        status_cls = getattr(Sample, "Status")
    status_value = getattr(status_cls, name, None) if status_cls is not None else None
    sample.status = status_value if status_value is not None else name


def _sample_trace_id(sample: Any) -> str:
    for attr in ("session_id", "sample_id", "id", "group_index"):
        value = getattr(sample, attr, None)
        if value is not None:
            return str(value)

    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("sample_id", "scenario_id", "case_id", "order_id"):
            value = metadata.get(key)
            if value is not None:
                return str(value)

    return "unknown"


def _log_extra(sample_id: str, evaluation: bool, **values: Any) -> dict[str, Any]:
    extra = {
        "retail_sample_id": sample_id,
        "retail_evaluation": bool(evaluation),
    }
    extra.update({f"retail_{key}": value for key, value in values.items()})
    return extra


def _preview_text(text: str, max_chars: int = 300) -> str:
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= max_chars:
        return preview
    return f"{preview[:max_chars]}..."


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        return list(encoded.get("input_ids", []))
    return list(getattr(encoded, "input_ids", []))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _prompt_to_log_string(prompt: Any) -> str:
    if isinstance(prompt, list):
        return json.dumps(prompt, ensure_ascii=False)
    return "" if prompt is None else str(prompt)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _conversation_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace = []
    for message in messages:
        safe_message = _json_safe(message)
        if isinstance(safe_message, dict):
            trace.append(safe_message)
    return trace


def _sample_tool_schemas(sample: Any) -> Any:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict) or "tools" not in metadata:
        raise ValueError("sample.metadata['tools'] is required to render tool-calling prompts")
    return metadata["tools"]


def _openai_tool_schemas(tool_schemas: Any) -> list[dict[str, Any]]:
    tools = tool_schemas if isinstance(tool_schemas, list) else []
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            normalized.append({"type": tool.get("type") or "function", "function": tool["function"]})
        elif isinstance(tool.get("name"), str):
            normalized.append({"type": "function", "function": tool})
    return normalized


def _normalize_prompt_messages(prompt: Any, tool_schemas: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        messages = []
        for message in prompt:
            if not isinstance(message, dict):
                messages.append({"role": "user", "content": _content_to_text(message)})
                continue
            role = str(message.get("role", "user")).lower()
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _content_to_text(message.get("content", ""))})
    else:
        messages = [{"role": "user", "content": _content_to_text(prompt)}]

    if os.environ.get("RETAIL_USE_XML_TOOL_PROTOCOL", "0").lower() in {"1", "true", "yes"}:
        tool_instructions = (
            TOOL_PROTOCOL
            + "\n\nAvailable tools:\n"
            + json.dumps(tool_schemas, ensure_ascii=False, indent=2)
        )
        if messages and messages[0]["role"] == "system":
            messages[0] = {
                "role": "system",
                "content": f"{messages[0]['content'].rstrip()}\n\n{tool_instructions}",
            }
        else:
            messages.insert(0, {"role": "system", "content": tool_instructions})
    return messages


def _render_messages(tokenizer: Any, messages: list[dict[str, Any]], tool_schemas: Any, *, add_generation_prompt: bool) -> str:
    openai_tools = _openai_tool_schemas(tool_schemas)
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = {"enable_thinking": _env_bool("RETAIL_ENABLE_THINKING", False)}
        try:
            return tokenizer.apply_chat_template(
                messages,
                tools=openai_tools or tool_schemas,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tools=openai_tools or tool_schemas,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except TypeError:
                try:
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        **template_kwargs,
                    )
                except TypeError:
                    pass
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception as exc:
                logger.warning("chat template rendering failed, falling back to plain text: %s", exc)
        except Exception as exc:
            logger.warning("chat template rendering failed, falling back to plain text: %s", exc)

    rendered = [f"{message['role']}: {message['content']}" for message in messages]
    if add_generation_prompt:
        rendered.append("assistant:")
    return "\n".join(rendered)


def _get_token_delta(tokenizer: Any, messages: list[dict[str, Any]], tool_schemas: Any) -> tuple[list[int], list[int]]:
    current_text = _render_messages(tokenizer, messages, tool_schemas, add_generation_prompt=False)
    last_message = messages[-1] if messages else {}
    previous_text = _render_messages(
        tokenizer,
        messages[:-1],
        tool_schemas,
        add_generation_prompt=last_message.get("role") == "assistant",
    )
    if current_text.startswith(previous_text):
        new_text = current_text[len(previous_text):]
    else:
        new_text = current_text[-(len(current_text) - len(previous_text)):] if len(current_text) > len(previous_text) else ""
    token_ids = _tokenize(tokenizer, new_text)
    loss_value = 1 if last_message.get("role") == "assistant" else 0
    return token_ids, [loss_value] * len(token_ids)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else stripped


def _first_json(text: str) -> Any | None:
    payload = _strip_code_fence(text)
    decoder = json.JSONDecoder()
    starts = [idx for idx in (payload.find("{"), payload.find("[")) if idx >= 0]
    if not starts:
        return None
    try:
        value, _ = decoder.raw_decode(payload[min(starts) :])
        return value
    except json.JSONDecodeError:
        return None


def _normalize_arg_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {ARG_ALIASES.get(key, key): _normalize_arg_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_arg_value(item) for item in value]
    return value


def _normalize_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        arguments = _first_json(arguments) or {"value": arguments}
    if not isinstance(arguments, dict):
        return {}
    return _normalize_arg_value(arguments)


def _normalize_tool_call(raw_call: Any, default_name: str | None = None) -> dict[str, Any] | None:
    if isinstance(raw_call, dict) and isinstance(raw_call.get("tool_calls"), list):
        raw_call = raw_call["tool_calls"][0] if raw_call["tool_calls"] else None
    if isinstance(raw_call, list):
        raw_call = raw_call[0] if raw_call else None
    if not isinstance(raw_call, dict):
        return None

    function_block = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
    name = default_name or raw_call.get("name") or raw_call.get("tool_name") or function_block.get("name")
    arguments = (
        raw_call.get("arguments")
        if "arguments" in raw_call
        else raw_call.get("args", raw_call.get("parameters", raw_call.get("input")))
    )
    if arguments is None and function_block:
        arguments = function_block.get("arguments")

    if not isinstance(name, str):
        return None
    name = name.strip()
    if name not in VALID_TOOL_NAMES:
        return None
    return {"name": name, "arguments": _normalize_args(arguments)}


def _extract_xml_tool_call(text: str) -> dict[str, Any] | None:
    tag_match = re.search(r"<tool_call[^>]*>\s*(.*?)(?:</tool_call>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if not tag_match:
        return None
    return _normalize_tool_call(_first_json(tag_match.group(1)))


def _extract_json_tool_call(text: str) -> dict[str, Any] | None:
    stripped = _strip_code_fence(text)
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return _normalize_tool_call(json.loads(stripped))
    except json.JSONDecodeError:
        return None


def _extract_function_tool_call(text: str) -> dict[str, Any] | None:
    tool_pattern = "|".join(re.escape(name) for name in sorted(VALID_TOOL_NAMES, key=len, reverse=True))
    match = re.search(rf"\b(?P<name>{tool_pattern})\s*\((?P<args>.*?)\)", text, flags=re.DOTALL)
    if not match:
        return None

    arguments: dict[str, Any] = {}
    args_text = match.group("args").strip()
    parsed_args = _first_json(args_text)
    if isinstance(parsed_args, dict):
        arguments = parsed_args
    else:
        for part in re.split(r",\s*", args_text):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            arguments[key.strip()] = value.strip().strip("\"'")
    return {"name": match.group("name"), "arguments": _normalize_args(arguments)}


def _parse_model_response(text: str, tool_schemas: Any) -> tuple[str, list[dict[str, Any]]]:
    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser  # type: ignore[reportMissingImports]
        from sglang.srt.managers.io_struct import Function, Tool as SglangTool  # type: ignore[reportMissingImports]

        tools = []
        for tool in _openai_tool_schemas(tool_schemas):
            function = tool.get("function") or {}
            if not isinstance(function, dict) or not function.get("name"):
                continue
            tools.append(
                SglangTool(
                    function=Function(
                        name=function["name"],
                        description=function.get("description") or "",
                        parameters=function.get("parameters") or {"type": "object", "properties": {}},
                    ),
                    type="function",
                )
            )
        if tools:
            parser = FunctionCallParser(tools=tools, tool_call_parser=os.environ.get("RETAIL_TOOL_PARSER", "qwen25"))
            normal_text, calls = parser.parse_non_stream(text)
            parsed_calls = []
            for call in calls[:1]:
                dumped = call.model_dump()
                normalized = _normalize_tool_call({"name": dumped.get("name"), "arguments": dumped.get("parameters") or {}})
                if normalized is not None:
                    parsed_calls.append(normalized)
            return normal_text or "", parsed_calls
    except Exception as exc:
        logger.debug("native tool parse failed: %s", exc)

    fallback_call = _extract_xml_tool_call(text) or _extract_json_tool_call(text) or _extract_function_tool_call(text)
    if fallback_call is None:
        return text, []
    cleaned = re.sub(r"<tool_call[^>]*>.*?(?:</tool_call>|$)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned, [fallback_call]


def _compact_json_text(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return text


def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        signature = inspect.signature(function)
        accepted_arguments = {key: value for key, value in arguments.items() if key in signature.parameters}
        missing = [
            key
            for key, parameter in signature.parameters.items()
            if parameter.default is inspect._empty and key not in accepted_arguments
        ]
        if missing:
            return json.dumps({"error": f"Missing required arguments for {name}: {', '.join(missing)}"})
        return _compact_json_text(function(**accepted_arguments))
    except Exception as exc:
        logger.exception("tool execution failed: %s", name)
        return json.dumps({"error": f"Tool {name} failed: {type(exc).__name__}: {exc}"})


def _tool_record(name: str, arguments: dict[str, Any], result: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "arguments": arguments,
        "result": result,
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
    }


def _final_output_text(response: str) -> str:
    cleaned = re.sub(r"<tool_result\b[^>]*>.*?</tool_result>", "", response, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<tool_call\b[^>]*>.*?</tool_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _max_context_length(args: Any) -> int:
    configured = getattr(args, "rollout_max_context_len", None)
    if configured is not None:
        return int(configured)
    return _env_int("RETAIL_MAX_TRAJ_TOKENS", DEFAULT_MAX_TRAJ_TOKENS)


def _max_response_length(args: Any, sampling_params: dict[str, Any], prompt_length: int) -> int:
    context_limit = max(0, _max_context_length(args) - prompt_length)
    configured = os.environ.get("RETAIL_MAX_RESPONSE_TOKENS")
    if not configured:
        return context_limit
    try:
        return min(int(configured), context_limit)
    except ValueError:
        logger.warning("Ignoring invalid RETAIL_MAX_RESPONSE_TOKENS=%r", configured)
        return context_limit


def _sampling_params_for_turn(sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    params = dict(sampling_params)
    params["max_new_tokens"] = max_new_tokens
    params.setdefault("no_stop_trim", True)
    params.setdefault("skip_special_tokens", False)
    params.setdefault("spaces_between_special_tokens", False)

    stop = params.get("stop") or []
    stop_values = [stop] if isinstance(stop, str) else list(stop)
    if TOOL_STOP not in stop_values:
        stop_values.append(TOOL_STOP)
    params["stop"] = stop_values
    return params


def _finish_type(output: dict[str, Any]) -> str:
    finish_reason = (output.get("meta_info") or {}).get("finish_reason") or {}
    if isinstance(finish_reason, dict):
        return str(finish_reason.get("type") or "stop")
    return str(finish_reason or "stop")


async def custom_generate(args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False) -> Any:
    """Slime custom generation function that runs one Retail tool-use episode."""
    if GenerateState is None or post is None:
        raise RuntimeError("Slime runtime is not available. Run this inside the Slime training image.")
    if getattr(args, "partial_rollout", False):
        raise AssertionError("retail_generate.custom_generate does not support partial rollout")

    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None
    sample.rollout_log_probs = []

    metadata = getattr(sample, "metadata", None) or {}
    if not isinstance(metadata, dict):
        try:
            metadata = json.loads(metadata)
        except (TypeError, json.JSONDecodeError, ValueError):
            metadata = {}
    original_prompt = getattr(sample, "prompt", "")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    tool_schemas = _openai_tool_schemas(_sample_tool_schemas(sample))
    messages = _normalize_prompt_messages(original_prompt, tool_schemas)
    if not messages:
        logger.warning("[retail_agent_generate] no prompt found; using generic fallback")
        messages = [{"role": "user", "content": "Help me."}]
    input_messages = _conversation_trace(messages)

    prompt_text = _render_messages(tokenizer, messages, tool_schemas, add_generation_prompt=True)
    prompt_token_ids = _tokenize(tokenizer, prompt_text)
    max_response_tokens = _max_response_length(args, sampling_params, len(prompt_token_ids))
    max_tool_turns = _max_tool_turns()
    sample_id = _sample_trace_id(sample)
    started_at = time.monotonic()
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    response_parts: list[str] = []
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    output_tools: list[dict[str, Any]] = []
    final_text: str | None = None
    empty_streak = 0
    terminated = False
    truncated = False
    aborted = False
    final_status = "COMPLETED"
    turn_count = 0
    submitted_via_tool = False

    headers = None
    if getattr(sample, "session_id", None) and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}

    logger.info(
        "[retail_agent_generate] sample_start sample_id=%s evaluation=%s prompt_tokens=%d "
        "max_response_tokens=%d max_tool_turns=%d router_policy=%s routed=%s",
        sample_id,
        bool(evaluation),
        len(prompt_token_ids),
        max_response_tokens,
        max_tool_turns,
        getattr(args, "router_policy", None),
        headers is not None,
        extra=_log_extra(
            sample_id,
            evaluation,
            prompt_tokens=len(prompt_token_ids),
            max_response_tokens=max_response_tokens,
            max_tool_turns=max_tool_turns,
            routed=headers is not None,
        ),
    )

    for turn_index in range(max_tool_turns):
        turn_count = turn_index + 1
        if terminated or truncated or aborted:
            break

        if len(prompt_token_ids) + len(response_token_ids) >= _max_context_length(args):
            final_status = "TRUNCATED"
            truncated = True
            break

        text_input = _render_messages(tokenizer, messages, tool_schemas, add_generation_prompt=True)
        input_token_ids = _tokenize(tokenizer, text_input)
        remaining_response = min(max_response_tokens, _max_context_length(args) - len(input_token_ids))
        if remaining_response <= 0:
            final_status = "TRUNCATED"
            truncated = True
            logger.warning(
                "[retail_agent_generate] response_budget_exhausted sample_id=%s turn=%d "
                "input_tokens=%d response_tokens=%d max_response_tokens=%d",
                sample_id,
                turn_count,
                len(input_token_ids),
                len(response_token_ids),
                max_response_tokens,
                extra=_log_extra(
                    sample_id,
                    evaluation,
                    turn=turn_count,
                    input_tokens=len(input_token_ids),
                    response_tokens=len(response_token_ids),
                    max_response_tokens=max_response_tokens,
                ),
            )
            break

        per_turn_max_tokens = min(int(sampling_params.get("max_new_tokens", remaining_response)), remaining_response)
        if per_turn_max_tokens <= 0:
            final_status = "TRUNCATED"
            truncated = True
            logger.warning(
                "[retail_agent_generate] turn_budget_exhausted sample_id=%s turn=%d remaining_response=%d",
                sample_id,
                turn_count,
                remaining_response,
                extra=_log_extra(sample_id, evaluation, turn=turn_count, remaining_response=remaining_response),
            )
            break

        payload = {"text": text_input, "sampling_params": _sampling_params_for_turn(sampling_params, per_turn_max_tokens)}

        logger.debug(
            "[retail_agent_generate] turn_request sample_id=%s turn=%d input_tokens=%d "
            "response_tokens=%d remaining_response=%d max_new_tokens=%d",
            sample_id,
            turn_count,
            len(input_token_ids),
            len(response_token_ids),
            remaining_response,
            per_turn_max_tokens,
            extra=_log_extra(
                sample_id,
                evaluation,
                turn=turn_count,
                input_tokens=len(input_token_ids),
                response_tokens=len(response_token_ids),
                remaining_response=remaining_response,
                max_new_tokens=per_turn_max_tokens,
            ),
        )

        try:
            if trace_span is None:
                output = await post(url, payload, headers=headers)
            else:
                trace_attrs = {
                    "sample_id": sample_id,
                    "evaluation": bool(evaluation),
                    "turn": turn_count,
                    "max_new_tokens": per_turn_max_tokens,
                    "prompt_tokens": len(prompt_token_ids),
                    "response_tokens": len(response_token_ids),
                }
                with trace_span(sample, "retail_agent_generate", attrs=trace_attrs) as span:
                    output = await post(url, payload, headers=headers)
                    if build_sglang_meta_trace_attrs is not None:
                        span.update(build_sglang_meta_trace_attrs(output.get("meta_info") or {}))
        except Exception:
            logger.exception(
                "[retail_agent_generate] turn_request_failed sample_id=%s turn=%d url=%s",
                sample_id,
                turn_count,
                url,
                extra=_log_extra(sample_id, evaluation, turn=turn_count, url=url),
            )
            raise

        raw_response = output.get("text") or ""
        if raw_response.endswith("<|im_end|>"):
            raw_response = raw_response[: -len("<|im_end|>")]
        normal_text, tool_calls = _parse_model_response(raw_response, tool_schemas)
        tool_calls = tool_calls[:1]
        if not raw_response and not tool_calls:
            logger.warning(
                "[retail_agent_generate] empty_model_output sample_id=%s turn=%d",
                sample_id,
                turn_count,
                extra=_log_extra(sample_id, evaluation, turn=turn_count),
            )
            empty_streak += 1
            if empty_streak >= 2:
                final_status = "TRUNCATED"
                truncated = True
                break
            continue

        if tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": normal_text.strip() or None,
                "tool_calls": [
                    {
                        "id": f"call_{len(output_tools)}_{call['name']}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True),
                        },
                    }
                    for call in tool_calls
                ],
            }
        else:
            assistant_msg = {"role": "assistant", "content": (normal_text or raw_response).strip()}

        content_present = bool(assistant_msg.get("content")) or bool(assistant_msg.get("tool_calls"))
        if not content_present:
            logger.warning(
                "[retail_agent_generate] empty_assistant_message sample_id=%s turn=%d",
                sample_id,
                turn_count,
                extra=_log_extra(sample_id, evaluation, turn=turn_count),
            )
            empty_streak += 1
            if empty_streak >= 2:
                final_status = "TRUNCATED"
                truncated = True
                break
            continue
        empty_streak = 0

        messages.append(assistant_msg)
        new_tokens, new_mask = _get_token_delta(tokenizer, messages, tool_schemas)
        if len(response_token_ids) + len(new_tokens) > max_response_tokens:
            available = max_response_tokens - len(response_token_ids)
            new_tokens = new_tokens[:available]
            new_mask = new_mask[:available]
            final_status = "TRUNCATED"
            truncated = True
        response_token_ids.extend(new_tokens)
        loss_mask.extend(new_mask)
        rollout_log_probs.extend([0.0] * len(new_tokens))
        if normal_text:
            response_parts.append(normal_text)

        finish_type = _finish_type(output)
        logger.debug(
            "[retail_agent_generate] turn_response sample_id=%s turn=%d finish_type=%s "
            "generated_tokens=%d response_tokens=%d output_preview=%s",
            sample_id,
            turn_count,
            finish_type,
            len(new_tokens),
            len(response_token_ids),
            _preview_text(raw_response),
            extra=_log_extra(
                sample_id,
                evaluation,
                turn=turn_count,
                finish_type=finish_type,
                generated_tokens=len(new_tokens),
                response_tokens=len(response_token_ids),
            ),
        )
        if finish_type == "abort":
            final_status = "ABORTED"
            aborted = True
            logger.warning(
                "[retail_agent_generate] model_aborted sample_id=%s turn=%d response_tokens=%d",
                sample_id,
                turn_count,
                len(response_token_ids),
                extra=_log_extra(sample_id, evaluation, turn=turn_count, response_tokens=len(response_token_ids)),
            )
            break
        if truncated or finish_type == "length":
            final_status = "TRUNCATED"
            truncated = True
            logger.warning(
                "[retail_agent_generate] model_truncated sample_id=%s turn=%d finish_type=%s "
                "response_tokens=%d max_response_tokens=%d",
                sample_id,
                turn_count,
                finish_type,
                len(response_token_ids),
                max_response_tokens,
                extra=_log_extra(
                    sample_id,
                    evaluation,
                    turn=turn_count,
                    finish_type=finish_type,
                    response_tokens=len(response_token_ids),
                    max_response_tokens=max_response_tokens,
                ),
            )
            break

        if not tool_calls:
            final_status = "COMPLETED"
            final_text = (normal_text or raw_response).strip()
            terminated = True
            logger.debug(
                "[retail_agent_generate] no_tool_call sample_id=%s turn=%d output_preview=%s",
                sample_id,
                turn_count,
                _preview_text(final_text),
                extra=_log_extra(sample_id, evaluation, turn=turn_count),
            )
            break

        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_arguments = tool_call.get("arguments") or {}
        tool_result = _execute_tool(tool_name, tool_arguments)
        output_tools.append(_tool_record(tool_name, tool_arguments, tool_result))
        if tool_name == "submit_resolution":
            submitted_via_tool = True
        logger.debug(
            "[retail_agent_generate] tool_call sample_id=%s turn=%d name=%s argument_keys=%s "
            "result_chars=%d tool_call_count=%d",
            sample_id,
            turn_count,
            tool_name,
            ",".join(sorted(tool_arguments)) or "-",
            len(tool_result),
            len(output_tools),
            extra=_log_extra(
                sample_id,
                evaluation,
                turn=turn_count,
                tool_name=tool_name,
                argument_keys=sorted(tool_arguments),
                result_chars=len(tool_result),
                tool_call_count=len(output_tools),
            ),
        )
        logger.debug(
            "[retail_agent_generate] tool_result sample_id=%s turn=%d name=%s result_preview=%s",
            sample_id,
            turn_count,
            tool_name,
            _preview_text(tool_result),
            extra=_log_extra(sample_id, evaluation, turn=turn_count, tool_name=tool_name),
        )

        tool_message = {
            "role": "tool",
            "tool_call_id": assistant_msg["tool_calls"][0]["id"],
            "name": tool_name,
            "content": tool_result or "",
        }
        messages.append(tool_message)
        obs_tokens, obs_mask = _get_token_delta(tokenizer, messages, tool_schemas)
        if len(response_token_ids) + len(obs_tokens) > max_response_tokens:
            available = max_response_tokens - len(response_token_ids)
            obs_tokens = obs_tokens[:available]
            obs_mask = obs_mask[:available]
            final_status = "TRUNCATED"
            truncated = True
        response_token_ids.extend(obs_tokens)
        loss_mask.extend(obs_mask)
        rollout_log_probs.extend([0.0] * len(obs_tokens))
        if truncated:
            logger.warning(
                "[retail_agent_generate] observation_truncated sample_id=%s turn=%d name=%s "
                "response_tokens=%d max_response_tokens=%d",
                sample_id,
                turn_count,
                tool_name,
                len(response_token_ids),
                max_response_tokens,
                extra=_log_extra(
                    sample_id,
                    evaluation,
                    turn=turn_count,
                    tool_name=tool_name,
                    response_tokens=len(response_token_ids),
                    max_response_tokens=max_response_tokens,
                ),
            )
            break
    else:
        final_status = "TRUNCATED"
        truncated = True
        logger.warning(
            "[retail_agent_generate] max_tool_turns_reached sample_id=%s max_tool_turns=%d response_tokens=%d",
            sample_id,
            max_tool_turns,
            len(response_token_ids),
            extra=_log_extra(
                sample_id,
                evaluation,
                max_tool_turns=max_tool_turns,
                response_tokens=len(response_token_ids),
            ),
        )

    if final_text is None:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                final_text = (message.get("content") or "") or ""
                break
        final_text = final_text or ""

    n_assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

    output_text = final_text or _final_output_text("".join(response_parts))

    # Score the trajectory here so retail_reward.custom_rm (the thin shim used
    # by Slime's --custom_rm_path) just returns sample.reward verbatim.
    reward_dict = retail_reward.score_retail(
        output_text or "",
        expected_actions=metadata.get("expected_actions") or {},
        expected_amounts=metadata.get("expected_amounts") or {},
        expected_resolution=metadata.get("expected_resolution") or "",
        expected_tools=metadata.get("expected_tools") or [],
        order_id=metadata.get("order_id"),
        target_items=metadata.get("target_items") or [],
        tool_calls=[{"name": t["name"], "args": t.get("arguments") or {}} for t in output_tools],
        n_tool_calls=len(output_tools),
        n_assistant_turns=n_assistant_turns,
        submitted_via_tool=submitted_via_tool,
    )

    sample.prompt = ""
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = "".join(response_parts) or final_text
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.reward = reward_dict
    _set_status(sample, final_status)

    sample.output_text = output_text
    sample.output_tools = output_tools
    metadata = dict(sample.metadata or {})
    metadata["input_prompt"] = _prompt_to_log_string(original_prompt)
    metadata["input_messages"] = input_messages
    metadata["conversation_trace"] = _conversation_trace(messages)
    metadata["final_response"] = output_text
    metadata["output_text"] = output_text
    metadata["output_tools"] = output_tools
    metadata["tool_call_count"] = len(output_tools)
    metadata["n_assistant_turns"] = n_assistant_turns
    metadata["submitted_via_tool"] = bool(submitted_via_tool)
    metadata["episode_terminated"] = bool(terminated)
    metadata["episode_truncated"] = bool(truncated)
    metadata["episode_aborted"] = bool(aborted)
    metadata["custom_generate_evaluation"] = bool(evaluation)
    sample.metadata = metadata

    if len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"loss_mask length {len(sample.loss_mask)} != response_length {sample.response_length}")
    if len(sample.rollout_log_probs) != sample.response_length:
        raise ValueError(
            f"rollout_log_probs length {len(sample.rollout_log_probs)} != response_length {sample.response_length}"
        )

    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    logger.info(
        "[retail_agent_generate] sample_finish sample_id=%s status=%s turns=%d response_tokens=%d "
        "tool_call_count=%d elapsed_ms=%d",
        sample_id,
        final_status,
        turn_count,
        sample.response_length,
        len(output_tools),
        elapsed_ms,
        extra=_log_extra(
            sample_id,
            evaluation,
            status=final_status,
            turns=turn_count,
            response_tokens=sample.response_length,
            tool_call_count=len(output_tools),
            elapsed_ms=elapsed_ms,
        ),
    )

    return sample
