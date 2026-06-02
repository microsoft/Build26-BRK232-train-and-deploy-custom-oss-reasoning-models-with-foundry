# Debug Grader Demo

## Purpose

Show one bug clearly: a generated answer can be ignored if it is not stored in the fields the grader expects.

The grader expects:

- `sample["output_text"]`
- `sample["output_tools"]`

If `sample["output_text"]` is missing, this grader can fall back to `item["messages"]`. That may accidentally score the reference answer instead of the generated answer.

## Files

Open these two files:

- `debug.py`
- `retail_grader_rft_tools_v3.py`

## Run

Run from the workspace root:

```powershell
python.exe debug.py
```

Expected output:

```text
Happy path score: 1.0
Generated bad sample score before output_text mapping: 1.0
Generated bad sample score after output_text mapping: 0.39
Issue: generated rollout samples can put the final answer in final_response/metadata['output_text']; grade() reads top-level sample['output_text'], so raw grading can fall back to item reference data instead of scoring the generated answer.
Fixed generated sample score: 1.0
```

## What To Show

1. Start in `run_happy_path()`.
   - The sample has top-level `output_text` and `output_tools`.
   - The score is `1.0`.

2. Go to `generated_bad_sample()`.
   - The generated answer is wrong.
   - It denies `SKU-TEST-999` instead of refunding `SKU-TEST-123`.
   - It does not have top-level `output_text`.

3. Step into `grade(generated_sample, item)`.
   - The grader does not find `sample["output_text"]`.
   - It falls back to the assistant message in `item["messages"]`.
   - The bad generated answer is masked, so the score is `1.0`.

4. Step through `grader_sample_from_generated()`.
   - It moves `final_response` or `metadata["output_text"]` into top-level `output_text`.
   - Now the grader scores the generated answer.
   - The bad generated answer scores `0.39`.

5. End with `generated_good_sample()`.
   - A correctly shaped good generated sample scores `1.0`.

## Code Proof

The fallback is in `retail_grader_rft_tools_v3.py`:

```python
def _extract_output_text(sample: dict, item: dict) -> str:
    if isinstance(sample, dict):
        output_text = sample.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

    if isinstance(item, dict):
        messages = item.get("messages") or []
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return message.get("content")

    return ""
```

The bad generated sample stores the wrong answer outside top-level `output_text`:

```python
def generated_bad_sample():
    final_response = "Action: deny for SKU-TEST-999 (reason: not eligible)."
    return {
        "final_response": final_response,
        "output_tools": happy_path_sample()["output_tools"],
        "metadata": {"output_text": final_response},
    }
```

The fix is to normalize before grading:

```python
def grader_sample_from_generated(sample):
    metadata = sample.get("metadata") or {}
    return {
        "output_text": sample.get("output_text")
        or sample.get("final_response")
        or metadata.get("output_text")
        or "",
        "output_tools": sample.get("output_tools")
        or metadata.get("output_tools")
        or [],
    }
```

Use it like this:

```python
generated_sample = generated_bad_sample()

masked_score = grade(generated_sample, item)

normalized_sample = grader_sample_from_generated(generated_sample)
actual_score = grade(normalized_sample, item)
```

## Breakpoints

Set breakpoints in this order:

1. `debug.py:114`
    - `masked_score = grade(generated_sample, item)`
    - Shows the bad generated sample being passed directly to `grade()`.

2. `retail_grader_rft_tools_v3.py:617`
    - `output_text = _extract_output_text(sample, item)`
    - Step into this call.

3. `retail_grader_rft_tools_v3.py:179`
    - `v = sample.get("output_text")`
    - Shows that the bad generated sample has no top-level `output_text`.

4. `retail_grader_rft_tools_v3.py:188`
    - `msgs = item.get("messages") or []`
    - Shows the fallback to the reference transcript.

5. `retail_grader_rft_tools_v3.py:194`
    - `return c`
    - Shows the grader returning the assistant message from `item["messages"]`.

6. `debug.py:117`
    - `bad_sample = grader_sample_from_generated(generated_sample)`
    - Shows the normalization fix.

7. `debug.py:118`
    - `bad_score = grade(bad_sample, item)`
    - Shows the same bad generated answer scoring `0.39` after normalization.

8. Optional good-sample proof:
    - `debug.py:122` - `fixed_sample = grader_sample_from_generated(generated_good_sample())`
    - `debug.py:123` - `fixed_score = grade(fixed_sample, item)`

## Takeaway

The grader can tell a good answer from a bad answer only if it reads the generated answer. Normalize generated samples into top-level `output_text` and `output_tools` before calling `grade()`.