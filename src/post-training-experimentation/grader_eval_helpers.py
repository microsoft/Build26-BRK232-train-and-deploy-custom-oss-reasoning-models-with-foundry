from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable


BASE_DIR = Path(__file__).resolve().parent
GRADER_PATH = BASE_DIR / "retail_grader_rft_tools_v3.py"
SAMPLES_PATH = BASE_DIR / "local_eval_samples_10.jsonl"

GRADER_FALLBACK_LINE = (
    '        v = v or sample.get("final_response") or sample.get("response") '
    'or (sample.get("metadata") or {}).get("output_text")\n'
)
COMMENTED_GRADER_FALLBACK_LINE = "        # " + GRADER_FALLBACK_LINE.lstrip()


def parse_structured_value(value):
    if isinstance(value, (dict, list, tuple)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{(":
        return value

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def normalize_metadata(row: dict) -> dict:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = parse_structured_value(metadata)
    if not isinstance(metadata, dict):
        metadata = {}
    return {key: parse_structured_value(value) for key, value in metadata.items()}


def load_jsonl(path: Path = SAMPLES_PATH) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_sample(row: dict) -> dict:
    return row


def build_item(row: dict) -> dict:
    metadata = normalize_metadata(row)
    return {
        "expected_resolution": metadata.get("expected_resolution") or row.get("expected_resolution") or "",
        "expected_tools": metadata.get("expected_tools") or row.get("expected_tools") or [],
        "expected_actions": metadata.get("expected_actions") or row.get("expected_actions") or {},
        "expected_amounts": metadata.get("expected_amounts") or row.get("expected_amounts") or {},
        "order_id": metadata.get("order_id") or row.get("order_id"),
        "target_items": metadata.get("target_items") or row.get("target_items") or [],
        "messages": metadata.get("input_messages")
        or metadata.get("conversation_trace")
        or parse_structured_value(row.get("prompt"))
        or [],
        "tools": metadata.get("tools") or [],
    }


def source_with_generated_text_fallback(source: str, enabled: bool) -> str:
    if enabled:
        if GRADER_FALLBACK_LINE in source:
            return source
        if COMMENTED_GRADER_FALLBACK_LINE in source:
            return source.replace(COMMENTED_GRADER_FALLBACK_LINE, GRADER_FALLBACK_LINE, 1)
    else:
        if COMMENTED_GRADER_FALLBACK_LINE in source:
            return source
        if GRADER_FALLBACK_LINE in source:
            return source.replace(GRADER_FALLBACK_LINE, COMMENTED_GRADER_FALLBACK_LINE, 1)

    state = "enabled" if enabled else "commented"
    raise ValueError(f"Could not render grader fallback line as {state}.")


def load_grader_namespace(fixed: bool, grader_path: Path = GRADER_PATH) -> dict:
    source = Path(grader_path).read_text(encoding="utf-8")
    source = source_with_generated_text_fallback(source, enabled=fixed)
    namespace = {"__name__": "retail_grader_compare"}
    exec(compile(source, str(grader_path), "exec"), namespace)
    return namespace


def load_grade_variant(fixed: bool, grader_path: Path = GRADER_PATH) -> Callable[[dict, dict], float]:
    namespace = load_grader_namespace(fixed=fixed, grader_path=grader_path)
    return namespace["grade"]


def get_sample_row(sample_idx: int, samples_path: Path = SAMPLES_PATH) -> dict:
    for row in load_jsonl(samples_path):
        if row.get("sample_idx") == sample_idx:
            return row
    raise ValueError(f"No sample found for sample_idx={sample_idx}.")


def extracted_output_text(fixed: bool, row: dict) -> str:
    namespace = load_grader_namespace(fixed=fixed)
    return namespace["_extract_output_text"](build_sample(row), build_item(row))


def parsed_action_lines(fixed: bool, row: dict) -> list[dict]:
    namespace = load_grader_namespace(fixed=fixed)
    output_text = namespace["_extract_output_text"](build_sample(row), build_item(row))
    return namespace["parse_action_lines"](output_text)


def score_rows(grader_func: Callable[[dict, dict], float], samples_path: Path = SAMPLES_PATH) -> list[dict]:
    results = []
    for index, row in enumerate(load_jsonl(samples_path)):
        results.append({
            "index": index,
            "sample_idx": row.get("sample_idx", index),
            "group_index": row.get("group_index"),
            "scenario_id": row.get("scenario_id") or normalize_metadata(row).get("scenario_id") or "",
            "score": grader_func(build_sample(row), build_item(row)),
        })
    return results


def average_score(results: list[dict]) -> float:
    return sum(result["score"] for result in results) / len(results) if results else 0.0


def print_score_report(title: str, results: list[dict], baseline: list[dict] | None = None) -> None:
    print(title)
    print(f"Average reward: {average_score(results):.3f}")

    if baseline is None:
        print("sample_idx  scenario                              score")
        for result in results:
            print(
                f"{result['sample_idx']:<10} "
                f"{result['scenario_id'][:36]:<36} "
                f"{result['score']:.3f}"
            )
        return

    print("sample_idx  scenario                              original  fixed   delta")
    for original, fixed in zip(baseline, results):
        delta = fixed["score"] - original["score"]
        print(
            f"{fixed['sample_idx']:<10} "
            f"{fixed['scenario_id'][:36]:<36} "
            f"{original['score']:<9.3f} "
            f"{fixed['score']:<7.3f} "
            f"{delta:+.3f}"
        )