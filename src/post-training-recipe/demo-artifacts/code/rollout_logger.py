"""
Custom rollout log function that dumps every rollout sample to a JSONL file.

Each line is a self-contained JSON object with prompt, response, reward,
and metadata — ready for dashboards (Streamlit, Gradio, Panel, etc.).

Usage:
    --custom-rollout-log-function-path examples.gsm8k_rollout_logger.rollout_logger.log_rollout_data
    --custom-eval-rollout-log-function-path examples.gsm8k_rollout_logger.rollout_logger.log_eval_rollout_data

The output directory defaults to ./rollout_logs/ and can be overridden
by setting the ROLLOUT_LOG_DIR environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

try:
    import mlflow  # type: ignore[import-not-found]
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

LOG_DIR = Path(os.environ.get("ROLLOUT_LOG_DIR", "rollout_logs"))


def _ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_to_str(prompt) -> str:
    """Normalise prompt to a plain string for the log file."""
    if isinstance(prompt, list):
        # chat-template style: [{"role": ..., "content": ...}, ...]
        return json.dumps(prompt, ensure_ascii=False)
    return "" if prompt is None else str(prompt)


def _sample_prompt_to_str(sample) -> str:
    prompt = _prompt_to_str(getattr(sample, "prompt", ""))
    if prompt:
        return prompt

    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("input_prompt", "original_prompt", "prompt_messages"):
            value = metadata.get(key)
            if value not in (None, ""):
                return _prompt_to_str(value)
    return prompt


def _sample_metadata(sample) -> dict:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _metadata_to_log(metadata: dict) -> dict:
    return {key: str(value) for key, value in metadata.items()}


def _safe_reward(reward) -> float | dict | None:
    """Return a JSON-serialisable reward value."""
    if isinstance(reward, (int, float)):
        return reward
    if isinstance(reward, dict):
        return {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in reward.items()}
    return None


# ---------------------------------------------------------------------------
# Train rollout logger
# ---------------------------------------------------------------------------

def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """
    Dump every sample from a training rollout to a JSONL file.

    Returns False so the default metric logging still runs.
    """
    _ensure_dir()
    path = LOG_DIR / f"train_rollout_{rollout_id:06d}.jsonl"

    records = []
    for idx, s in enumerate(samples):
        metadata = _sample_metadata(s)
        record = {
            "rollout_id": rollout_id,
            "sample_idx": idx,
            "group_index": s.group_index,
            "prompt": _sample_prompt_to_str(s),
            "conversation_trace": metadata.get("conversation_trace") or metadata.get("input_messages", []),
            "output_tools": metadata.get("output_tools", []),
            "final_response": metadata.get("final_response") or metadata.get("output_text") or s.response,
            "response": s.response,
            "label": s.label,
            "reward": _safe_reward(s.reward),
            "status": s.status.value,
            "response_length": s.response_length,
            "effective_response_length": s.effective_response_length,
            "removed": s.remove_sample,
            "metadata": _metadata_to_log(metadata),
            "rollout_time_sec": round(rollout_time, 2),
            "timestamp": time.time(),
        }
        records.append(record)

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"[rollout_logger] Wrote {len(records)} samples -> {path}")

    if _MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(path), artifact_path="rollout_logs/train")
            logger.info(f"[rollout_logger] Logged artifact to mlflow: {path.name}")
        except Exception as exc:
            logger.warning(f"[rollout_logger] mlflow artifact upload failed: {exc}")

    return False


# ---------------------------------------------------------------------------
# Eval rollout logger
# ---------------------------------------------------------------------------
    
def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    """
    Dump every sample from an evaluation rollout to a JSONL file.

    `data` is a dict keyed by dataset name, each value containing
    'rewards', optionally 'samples', 'truncated', etc.

    Returns False so the default metric logging still runs.
    """
    _ensure_dir()
    path = LOG_DIR / f"eval_rollout_{rollout_id:06d}.jsonl"

    records = []
    for dataset_name, dataset_data in data.items():
        rewards = dataset_data.get("rewards", [])
        samples = dataset_data.get("samples", [])
        truncated = dataset_data.get("truncated", [])

        for idx, reward in enumerate(rewards):
            record = {
                "rollout_id": rollout_id,
                "dataset": dataset_name,
                "sample_idx": idx,
                "reward": _safe_reward(reward),
                "truncated": truncated[idx] if idx < len(truncated) else None,
                "timestamp": time.time(),
            }
            # If full Sample objects are available, add prompt/response
            if idx < len(samples):
                s = samples[idx]
                metadata = _sample_metadata(s)
                record["prompt"] = _sample_prompt_to_str(s)
                record["conversation_trace"] = metadata.get("conversation_trace") or metadata.get("input_messages", [])
                record["output_tools"] = metadata.get("output_tools", [])
                record["final_response"] = metadata.get("final_response") or metadata.get("output_text") or s.response
                record["response"] = s.response
                record["label"] = s.label
                record["status"] = s.status.value
                record["response_length"] = s.response_length
                record["metadata"] = _metadata_to_log(metadata)

            records.append(record)

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"[rollout_logger] Wrote {len(records)} eval samples -> {path}")

    if _MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(path), artifact_path="rollout_logs/eval")
            logger.info(f"[rollout_logger] Logged artifact to mlflow: {path.name}")
        except Exception as exc:
            logger.warning(f"[rollout_logger] mlflow artifact upload failed: {exc}")

    return False
