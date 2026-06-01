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

import json
import logging
import os
import time
from math import isnan
from pathlib import Path

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

LOG_DIR = Path(os.environ.get("ROLLOUT_LOG_DIR", "rollout_logs"))

# When RETAIL_VERBOSE_LOGS != "true"/"1", only emit the four customer-facing
# metrics surfaced in the demo notebook:
#   1. eval/reward_mean             (aggregate across all eval datasets)
#   2. train/response_length_mean
#   3. train/entropy_loss
#   4. train/kl_loss
# All other train/* and eval/* scalars are silently dropped. The JSONL rollout
# dumps and artifact uploads are unaffected — only AzureML scalar logging is
# filtered. The rollout browser (Streamlit dashboard) keeps full data regardless
# of this flag.
_VERBOSE_LOGS = os.environ.get("RETAIL_VERBOSE_LOGS", "").strip().lower() in ("1", "true", "yes")
_ALLOWED_METRICS_EXACT = {
    "eval/reward_mean",
    "train/response_length_mean",
    "train/entropy_loss",
    "train/kl_loss",
}

def _metric_allowed(name: str) -> bool:
    if _VERBOSE_LOGS:
        return True
    return name in _ALLOWED_METRICS_EXACT


# ---------------------------------------------------------------------------
# AzureML logger (lazy, no-op when not running inside an AzureML job)
# ---------------------------------------------------------------------------

class _AzureMLLogger:
    """Minimal AzureML run-context logger for slime rollout metrics.

    Mirrors the verl AzureMLLogger pattern: log scalar metrics via
    ``Run.log(name, value, step=...)`` and upload artifact files via
    ``Run.upload_file(...)``. Silently no-ops when azureml-core is not
    installed or when the job is running in OfflineRun mode.
    """

    def __init__(self):
        self.run = None
        self.parent = None
        self.logged = set()
        try:
            from azureml.core.run import Run  # type: ignore[import-untyped]
            run = Run.get_context()
            if run is None or "OfflineRun" in getattr(run, "id", ""):
                logger.info("[azureml_logger] OfflineRun detected; AzureML logging disabled.")
                return
            self.run = run
            self.parent = self._find_pipeline_parent(run)
            logger.info(f"[azureml_logger] initialized for run id={run.id}")
        except ImportError:
            logger.info("[azureml_logger] azureml-core not available; AzureML logging disabled.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[azureml_logger] init failed: {exc}")

    @staticmethod
    def _find_pipeline_parent(run):
        try:
            parent = run.parent
            child = None
            while parent is not None and getattr(parent, "type", "") in (
                "PipelineRun", "StepRun", "FineTuneRun", "finetunerun"
            ):
                child = parent
                parent = parent.parent
            return child
        except Exception:  # noqa: BLE001
            return None

    def log_metric(self, name: str, value, step: int | None = None) -> None:
        if self.run is None or not isinstance(value, (int, float)) or isnan(float(value)):
            return
        if not _metric_allowed(name):
            return
        key = (step, name)
        if key in self.logged:
            return
        try:
            self.run.log(name, value, description=name, step=step)
            if self.parent is not None:
                try:
                    self.parent.log(name, value, description=name, step=step)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"[azureml_logger] parent log failed: {exc}")
            self.logged.add(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[azureml_logger] log failed for {name}: {exc}")

    def upload_artifact(self, local_path: Path, remote_name: str) -> None:
        if self.run is None:
            return
        try:
            self.run.upload_file(name=remote_name, path_or_stream=str(local_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[azureml_logger] upload_file failed for {remote_name}: {exc}")


_AZUREML = _AzureMLLogger()


# ---------------------------------------------------------------------------
# Slime metric bridge (wandb / TensorBoard -> AzureML)
# ---------------------------------------------------------------------------

def init_slime_metric_bridge(_args) -> None:
    """Bridge slime's wandb/tensorboard log path to AzureML.

    Slime computes ``train/kl_loss`` / ``train/entropy_loss`` / grad_norm /
    pg_loss etc. inside the training Megatron actors and routes them through
    ``slime.utils.logging_utils.log()`` -> ``wandb.log()`` (when
    ``--use-wandb``) or TensorBoard. With ``WANDB_MODE=offline`` (which every
    Foundry job sets) those values never escape the worker container.

    Hook this function via
    ``--custom-megatron-init-path rollout_logger.init_slime_metric_bridge``
    and slime will call it inside every training actor at init time. We then
    wrap ``logging_utils.log`` so every metric it forwards to wandb / TB is
    also forwarded to ``_AZUREML.log_metric`` (filtered by ``_metric_allowed``,
    so only the 4-metric customer-facing set lands when verbose_logs=False).

    ``step_key`` defaults to ``"train/step"``; we read that value from each
    incoming metrics dict and use it as the AzureML metric step so the
    dashboard plots align with slime's wandb step axis.
    """
    try:
        from slime.utils import logging_utils as _lu  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"[azureml_bridge] slime.utils.logging_utils import failed: {exc}; skipping bridge"
        )
        return

    if getattr(_lu, "_azureml_bridge_installed", False):
        return

    _orig_log = _lu.log

    def _bridged_log(args, metrics, step_key: str):
        # Always delegate to the original wandb/TB path first.
        try:
            _orig_log(args, metrics, step_key)
        finally:
            try:
                step_val = metrics.get(step_key) if isinstance(metrics, dict) else None
                step_int = int(step_val) if isinstance(step_val, (int, float)) else None
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if k == step_key:
                            continue
                        _AZUREML.log_metric(k, v, step=step_int)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[azureml_bridge] log_metric mirror failed: {exc}")

    _lu.log = _bridged_log
    _lu._azureml_bridge_installed = True
    logger.info("[azureml_bridge] slime.utils.logging_utils.log -> AzureML mirror installed")


def _ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_to_str(prompt) -> str:
    """Normalise prompt to a plain string for the log file."""
    if isinstance(prompt, list):
        # chat-template style: [{"role": ..., "content": ...}, ...]
        return json.dumps(prompt, ensure_ascii=False)
    return str(prompt)


def _safe_reward(reward) -> float | dict | None:
    """Return a JSON-serialisable reward value."""
    if isinstance(reward, (int, float)):
        return reward
    if isinstance(reward, dict):
        return {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in reward.items()}
    return None


def _scalar_reward(reward) -> float | None:
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        for key in ("reward", "score", "value", "total"):
            v = reward.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


# ---------------------------------------------------------------------------
# Rollout-browser field extractors
#
# The Streamlit dashboard's "Conversation trace browser" reads these
# per-sample fields out of each JSONL record:
#   * prompt            - the rendered input prompt (string form)
#   * conversation_trace- full multi-turn message list (list[dict])
#   * output_tools      - tool calls the model emitted (list[dict])
#   * final_response    - the model's final assistant text
#   * response          - raw response token string from slime
#   * label             - reward label
#   * status            - sample status
#   * response_length   - token count
#   * metadata          - the rest of sample.metadata (structured fields
#                         preserved, unknown values stringified)
#
# retail_generate.py populates `sample.metadata` with the upstream
# `conversation_trace`, `input_messages`, `output_tools`, `final_response`,
# `output_text`, `expected_tools`, etc. The helpers below mirror what the
# dashboard's `get_*` helpers look for, so a record written here is
# directly browsable in dashboard.py without further normalization.
# ---------------------------------------------------------------------------

_STRUCTURED_METADATA_KEYS = (
    "conversation_trace",
    "input_messages",
    "input_prompt",
    "output_tools",
    "expected_tools",
    "final_response",
    "output_text",
    "tool_call_count",
)


def _is_jsonable(value) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _sample_metadata(s) -> dict:
    """Return sample.metadata as a plain dict (empty when absent)."""
    raw = getattr(s, "metadata", None)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except Exception:  # noqa: BLE001
        return {}


def _sample_prompt_to_str(s) -> str:
    """Prefer the full input-messages chat from metadata; fall back to s.prompt."""
    metadata = _sample_metadata(s)
    candidates = (
        metadata.get("input_messages"),
        metadata.get("conversation_trace"),
        metadata.get("input_prompt"),
        getattr(s, "prompt", None),
    )
    for value in candidates:
        if value is None or value == "":
            continue
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False, default=str)
        return str(value)
    return ""


def _metadata_to_log(metadata: dict) -> dict:
    """JSON-serialisable copy of sample.metadata.

    Structured keys consumed by the dashboard (conversation_trace,
    input_messages, output_tools, etc.) are passed through unchanged so
    they remain parseable lists/dicts. Other JSON-safe values pass through
    as-is; anything else falls back to ``str()``.
    """
    out: dict = {}
    for k, v in metadata.items():
        key = str(k)
        if v is None or isinstance(v, (bool, int, float)):
            out[key] = v
            continue
        if key in _STRUCTURED_METADATA_KEYS or isinstance(v, (list, dict)):
            if _is_jsonable(v):
                out[key] = v
            else:
                try:
                    out[key] = json.loads(json.dumps(v, ensure_ascii=False, default=str))
                except Exception:  # noqa: BLE001
                    out[key] = str(v)
            continue
        if isinstance(v, str):
            out[key] = v
            continue
        out[key] = str(v)
    return out


def _enrich_sample_record(record: dict, s) -> None:
    """Add rollout-browser fields (prompt/trace/tools/final_response/metadata) to record.

    Also backfills `group_index`, `effective_response_length`, and `removed`
    from the sample so eval records — which previously only carried reward and
    truncated — expose the same shape as train records to the dashboard
    (dashboard.py groups by `group_index` and reads these for both splits).
    """
    metadata = _sample_metadata(s)
    record["prompt"] = _sample_prompt_to_str(s)
    record["conversation_trace"] = (
        metadata.get("conversation_trace") or metadata.get("input_messages") or []
    )
    record["output_tools"] = metadata.get("output_tools", [])
    record["final_response"] = (
        metadata.get("final_response")
        or metadata.get("output_text")
        or getattr(s, "response", "")
    )
    record["response"] = getattr(s, "response", "")
    record["label"] = getattr(s, "label", None)
    status = getattr(s, "status", None)
    record["status"] = status.value if hasattr(status, "value") else status
    record["response_length"] = getattr(s, "response_length", None)
    record["effective_response_length"] = getattr(s, "effective_response_length", None)
    record["group_index"] = getattr(s, "group_index", None)
    record["removed"] = bool(getattr(s, "remove_sample", False))
    # Dashboard reads these as top-level columns too (with metadata fallback).
    expected_tools = metadata.get("expected_tools")
    if expected_tools is not None:
        record["expected_tools"] = expected_tools
    record["metadata"] = _metadata_to_log(metadata)


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
    rewards: list[float] = []
    removed_count = 0
    response_lens: list[int] = []
    for idx, s in enumerate(samples):
        record = {
            "rollout_id": rollout_id,
            "sample_idx": idx,
            "group_index": s.group_index,
            "reward": _safe_reward(s.reward),
            "effective_response_length": s.effective_response_length,
            "removed": s.remove_sample,
            "rollout_time_sec": round(rollout_time, 2),
            "timestamp": time.time(),
        }
        _enrich_sample_record(record, s)
        records.append(record)
        r = _scalar_reward(s.reward)
        if r is not None:
            rewards.append(r)
        if getattr(s, "remove_sample", False):
            removed_count += 1
        if getattr(s, "response_length", None):
            response_lens.append(int(s.response_length))

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

    if rewards:
        mean_r = sum(rewards) / len(rewards)
        _AZUREML.log_metric("train/reward_mean", mean_r, step=rollout_id)
        _AZUREML.log_metric("train/reward_max", max(rewards), step=rollout_id)
        _AZUREML.log_metric("train/reward_min", min(rewards), step=rollout_id)
    _AZUREML.log_metric("train/num_samples", len(samples), step=rollout_id)
    _AZUREML.log_metric("train/removed_samples", removed_count, step=rollout_id)
    _AZUREML.log_metric("train/rollout_time_sec", float(rollout_time), step=rollout_id)
    if response_lens:
        _AZUREML.log_metric(
            "train/response_length_mean",
            sum(response_lens) / len(response_lens),
            step=rollout_id,
        )
    if isinstance(rollout_extra_metrics, dict):
        for k, v in rollout_extra_metrics.items():
            if isinstance(v, (int, float)):
                _AZUREML.log_metric(f"train/{k}", v, step=rollout_id)
    _AZUREML.upload_artifact(path, f"rollout_logs/train/{path.name}")

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
    per_dataset_rewards: dict[str, list[float]] = {}
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
            # If full Sample objects are available, add the rollout-browser
            # fields (prompt, conversation_trace, output_tools, final_response,
            # response, label, status, response_length, metadata).
            if idx < len(samples):
                _enrich_sample_record(record, samples[idx])

            records.append(record)
            r = _scalar_reward(reward)
            if r is not None:
                per_dataset_rewards.setdefault(dataset_name, []).append(r)

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

    # Flat eval scalars — no dataset segment. When multiple eval datasets are
    # configured their rewards are pooled into a single aggregate so the
    # AzureML Studio Metrics tab shows one clean `eval/reward_mean` curve.
    # Per-dataset breakdowns remain available in the JSONL rollout dumps
    # consumed by the rollout browser dashboard.
    all_eval_rewards: list[float] = []
    for rewards in per_dataset_rewards.values():
        if rewards:
            all_eval_rewards.extend(rewards)

    if all_eval_rewards:
        _AZUREML.log_metric(
            "eval/reward_mean",
            sum(all_eval_rewards) / len(all_eval_rewards),
            step=rollout_id,
        )
        _AZUREML.log_metric("eval/reward_max", max(all_eval_rewards), step=rollout_id)
        _AZUREML.log_metric("eval/reward_min", min(all_eval_rewards), step=rollout_id)
        _AZUREML.log_metric("eval/num_samples", len(all_eval_rewards), step=rollout_id)

    if isinstance(extra_metrics, dict):
        for k, v in extra_metrics.items():
            if isinstance(v, (int, float)):
                _AZUREML.log_metric(f"eval/{k}", v, step=rollout_id)
    _AZUREML.upload_artifact(path, f"rollout_logs/eval/{path.name}")

    return False
