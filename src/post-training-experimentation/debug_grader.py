import json
from pathlib import Path

from retail_grader_rft_tools_v3 import grade


SAMPLES_PATH = Path(__file__).with_name("local_eval_samples_10.jsonl")


def fabricated_item():
    # Minimal ground-truth item that mimics the dataset fields consumed by grade().
    return {
        "order_id": "ORD-TEST-001",
        "target_items": ["SKU-TEST-123"],
        "expected_actions": {
            "SKU-TEST-123": {"action": "refund", "reason": "damaged"},
        },
        "expected_amounts": {"SKU-TEST-123_refund": 24.50},
        "expected_resolution": "Action: refund for SKU-TEST-123 (reason: damaged). Amount: $24.50.",
        "expected_tools": [
            "get_order_details",
            "check_resolution_policy",
            "calculate_resolution",
            "submit_resolution",
        ],
        # The grader can fall back to the item transcript when the sample lacks
        # top-level output_text, which is the masking behavior demonstrated below.
        "messages": [
            {"role": "user", "content": "I need help with ORD-TEST-001."},
            {
                "role": "assistant",
                "content": "Policy: approved\nAction: refund for SKU-TEST-123 (reason: damaged)\nAmount: $24.50",
            },
        ],
        "tools": [
            {"name": "get_order_details"},
            {"name": "check_resolution_policy"},
            {"name": "calculate_resolution"},
            {"name": "submit_resolution"},
        ],
    }


def happy_path_sample():
    # Canonical online-grading shape: generated text and tool calls live directly
    # on the sample passed to grade().
    return {
        "output_text": "Policy: approved\nAction: refund for SKU-TEST-123 (reason: damaged)\nAmount: $24.50",
        "output_tools": [
            {"name": "get_order_details", "args": {"order_id": "ORD-TEST-001"}},
            {"name": "check_resolution_policy", "args": {"order_id": "ORD-TEST-001"}},
            {
                "name": "calculate_resolution",
                "args": {
                    "order_id": "ORD-TEST-001",
                    "items": [{"item_id": "SKU-TEST-123", "action": "refund"}],
                },
            },
            {
                "name": "submit_resolution",
                "args": {
                    "order_id": "ORD-TEST-001",
                    "resolution_summary": "Refund SKU-TEST-123 for damaged item.",
                },
            },
        ],
    }


def generated_bad_sample():
    # Keep the workflow correct so the demo isolates the answer-text extraction issue.
    output_tools = happy_path_sample()["output_tools"]
    final_response = "Action: deny for SKU-TEST-999 (reason: not eligible)."
    return {
        # Simulate generated rollout exports that store the answer under
        # final_response/metadata instead of top-level output_text.
        "final_response": final_response,
        "output_tools": output_tools,
        "conversation_trace": [
            {"role": "user", "content": "I need help with ORD-TEST-001."},
            {"role": "assistant", "content": final_response},
        ],
        "metadata": {
            "output_text": final_response,
            "output_tools": output_tools,
        },
    }


def generated_good_sample():
    # Same logical result as happy_path_sample(), but shaped like a generated rollout.
    sample = happy_path_sample()
    return {
        "output_text": sample["output_text"],
        "final_response": sample["output_text"],
        "output_tools": sample["output_tools"],
        "conversation_trace": [
            {"role": "user", "content": "I need help with ORD-TEST-001."},
            {"role": "assistant", "content": sample["output_text"]},
        ],
        "metadata": {
            "output_text": sample["output_text"],
            "output_tools": sample["output_tools"],
        },
    }


def grader_sample_from_generated(sample):
    # Normalize rollout exports into the direct fields grade() reads first.
    metadata = sample.get("metadata") or {}
    return {
        "output_text": sample.get("output_text") or sample.get("final_response") or metadata.get("output_text") or "",
        "output_tools": sample.get("output_tools") or metadata.get("output_tools") or [],
    }


def run_happy_path():
    # Baseline: a correctly shaped, correct answer should receive full credit.
    item = fabricated_item()
    sample = happy_path_sample()

    score = grade(sample, item)
    print(f"Happy path score: {score}")
    return score


def run_generated_shape_probe():
    # Build one bad generated answer and score it with the current grader.
    item = fabricated_item()
    generated_sample = generated_bad_sample()

    current_score = grade(generated_sample, item)
    print(f"Generated bad sample score with current grader: {current_score}")

    normalized_sample = grader_sample_from_generated(generated_sample)
    normalized_score = grade(normalized_sample, item)
    print(f"Generated bad sample score after external normalization: {normalized_score}")
    if current_score == normalized_score:
        print("Generated-field fallback is enabled in _extract_output_text().")
    else:
        print("Generated-field fallback is disabled; the generated answer can be masked by item fallback data.")

    # A correctly generated answer still earns full credit with the generated-rollout shape.
    fixed_sample = generated_good_sample()
    fixed_score = grade(fixed_sample, item)
    print(f"Fixed generated sample score: {fixed_score}")
    return current_score, normalized_score, fixed_score


def run_local_eval_samples():
    rows = []
    with SAMPLES_PATH.open(encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            item = sample.get("metadata") or {}
            score = grade(sample, item)
            rows.append((sample["sample_idx"], sample["scenario_id"], score))

    avg = round(sum(score for _, _, score in rows) / len(rows), 3)
    print(f"Local eval average: {avg}")
    for sample_idx, scenario_id, score in rows:
        print(f"  {sample_idx}: {scenario_id}: {score}")
    return avg, rows


if __name__ == "__main__":
    happy_score = run_happy_path()
    current_score, normalized_score, fixed_score = run_generated_shape_probe()
    local_avg, _ = run_local_eval_samples()

    if happy_score != 1.0 or normalized_score >= 0.5 or fixed_score != 1.0 or local_avg <= 0.5:
        raise SystemExit("Unexpected grader demo scores")