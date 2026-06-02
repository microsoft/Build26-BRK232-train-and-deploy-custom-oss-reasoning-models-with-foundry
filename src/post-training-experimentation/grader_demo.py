from __future__ import annotations

from grader_eval_helpers import (
    build_item,
    build_sample,
    extracted_output_text,
    get_sample_row,
    load_grade_variant,
    parsed_action_lines,
    print_score_report,
    score_rows,
)


def grader() -> list[dict]:
    results = score_rows(load_grade_variant(fixed=False))
    print_score_report("Original grader scores", results)
    return results


def grader_fixed() -> list[dict]:
    original_results = score_rows(load_grade_variant(fixed=False))
    fixed_results = score_rows(load_grade_variant(fixed=True))
    print_score_report("Fixed grader scores", fixed_results, baseline=original_results)
    print_improvement_example(sample_idx=10)
    return fixed_results


def print_improvement_example(sample_idx: int = 0) -> None:
    row = get_sample_row(sample_idx)
    original_grade = load_grade_variant(fixed=False)
    fixed_grade = load_grade_variant(fixed=True)
    original_score = original_grade(build_sample(row), build_item(row))
    fixed_score = fixed_grade(build_sample(row), build_item(row))
    original_text = extracted_output_text(fixed=False, row=row)
    fixed_text = extracted_output_text(fixed=True, row=row)
    original_actions = parsed_action_lines(fixed=False, row=row)
    fixed_actions = parsed_action_lines(fixed=True, row=row)

    print("\nDetailed improvement example")
    print(f"sample_idx={row.get('sample_idx')} scenario={row.get('scenario_id')}")
    print(f"Original score: {original_score:.3f}")
    print("Original grader sees:")
    print(original_text if original_text else "<empty text>")
    print(f"Original parsed actions: {original_actions}")
    print(f"Fixed score:    {fixed_score:.3f}")
    print("Fixed grader sees:")
    print(fixed_text if fixed_text else "<empty text>")
    print(f"Fixed parsed actions: {fixed_actions}")
    print(
        "Cause: the original extractor only reads sample['output_text'], but this "
        "row stores its correct generated answer in final_response. The fixed "
        "extractor reads final_response, so the grader can parse the refund action, "
        "item, reason, and amount and award full decision credit."
    )


if __name__ == "__main__":
    grader()
    print()
    grader_fixed()