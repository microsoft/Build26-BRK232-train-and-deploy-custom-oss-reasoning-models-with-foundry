# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Slime training entrypoint for Retail RFT on Azure ML.
This module prepares model sources, runtime packages, and the Ray job command.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("slime-rft-retail")

for _name in (
    "ray", "ray.serve", "ray.rllib", "ray.tune", "ray.data",
    "ray.workflow", "ray.autoscaler", "ray._private",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

def detect_nvlink():
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status", "-i", "0"],
            capture_output=True, text=True, timeout=10,
        )
        return "active" in result.stdout.lower()
    except Exception:
        return False


def configure_hf_cache_dirs(hf_home_override: str | None = None):
    if hf_home_override:
        hf_home = Path(hf_home_override).resolve()
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    work_dir = Path(os.getenv("SLIME_WORK_DIR", Path.cwd())).resolve()
    hf_home = Path(os.getenv("HF_HOME", work_dir / "hf_cache")).resolve()
    hf_hub_cache = Path(os.getenv("HF_HUB_CACHE", hf_home / "hub")).resolve()
    transformers_cache = Path(
        os.getenv("TRANSFORMERS_CACHE", hf_home / "transformers")
    ).resolve()

    for cache_dir in (hf_home, hf_hub_cache, transformers_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    logger.info(f"Hugging Face cache root: {hf_home}")
    logger.info(f"Hugging Face hub cache: {hf_hub_cache}")
    logger.info(f"Transformers cache:    {transformers_cache}")

    return hf_home, hf_hub_cache, transformers_cache


def resolve_checkpoint_source(
    *,
    checkpoint_path,
    model_id,
    hf_hub_cache,
    role,
):
    if checkpoint_path:
        logger.info(f"Using provided {role} checkpoint path: {checkpoint_path}")
        return checkpoint_path

    if not model_id:
        raise ValueError(
            f"Provide either --{role}_checkpoint or --{role}_model_id "
            f"(or set SLIME_{role.upper()}_MODEL_ID)."
        )

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    logger.info(f"Downloading {role} model from Hugging Face: {model_id}")
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(hf_hub_cache),
        token=token or None,
    )
    logger.info(f"{role.capitalize()} model ready at: {local_path}")
    return local_path


def merge_lora_into_base(*, base_path, lora_path, output_dir):
    """Merge a PEFT LoRA adapter into the base model and reuse cached output when present."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_marker = output_dir / "config.json"
    if merged_marker.exists() and any(output_dir.glob("*.safetensors")):
        logger.info(f"LoRA-merged checkpoint already present at {output_dir}; reusing.")
        return output_dir

    logger.info(f"Merging LoRA adapter {lora_path} into base {base_path} -> {output_dir}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Ensure peft is importable. Install peft==0.13.2 explicitly to
    # /tmp because:
    #   * we always need 0.13.2 here (newer eagerly-imports TE → cuDNN
    #     symbol crash on this image), so a user-site peft of any other
    #     version is a hazard;
    #   * we already rm -rf'd any user-site peft in
    #     install_custom_pip_packages(), so user-site is empty.
    import subprocess as _sp, sys as _sys
    target = "/tmp/_peft_install"
    if not Path(target).exists() or not any(Path(target).glob("peft/__init__.py")):
        logger.info(f"Installing peft==0.13.2 to {target} …")
        _sp.run(
            [_sys.executable, "-m", "pip", "install", "--quiet",
             "--target", target, "--no-deps", "peft==0.13.2"],
            check=True, capture_output=True, text=True, timeout=600,
        )
    if target not in _sys.path:
        _sys.path.insert(0, target)
    import importlib
    importlib.invalidate_caches()
    # Clear failed imports so Python sees the isolated peft install.
    for k in list(_sys.modules):
        if k == "peft" or k.startswith("peft."):
            del _sys.modules[k]
    from peft import PeftModel  # type: ignore  # noqa: E501
    logger.info(f"peft loaded from {PeftModel.__module__}")

    base = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(str(base_path))
    peft_model = PeftModel.from_pretrained(base, str(lora_path))
    merged = peft_model.merge_and_unload()
    logger.info("LoRA merge complete; saving merged model …")
    merged.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(str(output_dir))

    # Free up host RAM before slime/Ray spawn the actor processes.
    del base, peft_model, merged
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Merged checkpoint saved at {output_dir}")
    return output_dir


def build_slime_command(args):
    """Build the Slime train_async.py command for Retail agentic RL."""
    cmd = [
        "python3", "/opt/slime/train_async.py",
        "--actor-num-nodes", str(args.actor_num_nodes),
        "--actor-num-gpus-per-node", str(args.actor_gpus_per_node),
        # Keep rollout engines on their own GPU pool for async throughput.
        "--rollout-num-gpus", str(args.rollout_num_gpus),
        "--update-weights-interval", str(args.update_weights_interval),
        "--start-rollout-id", str(args.start_rollout_id),
        "--megatron-to-hf-mode", "bridge",
        "--hf-checkpoint", args.hf_checkpoint,
        "--ref-load", args.ref_checkpoint,
        "--save", args.output_dir,
        "--save-interval", str(args.save_interval),
        "--save-hf", f"{args.model_output}/step_{{rollout_id}}",

        # Keep architecture-specific flags outside the shared command when possible.
        "--seq-length", str(args.seq_length),
        "--tokenizer-type", "NullTokenizer",
        "--bf16",
        # Keep these keys aligned with the uploaded Retail JSONL schema.
        "--prompt-data", args.train_data,
        "--input-key", args.input_key,
        "--metadata-key", args.metadata_key,
        "--apply-chat-template",
        "--rollout-shuffle",
        # Use the Retail reward shim so custom_generate and Slime agree on score shape.
        "--custom-rm-path", args.custom_rm_path,
        # Extract the scalar score because rollout metrics cannot round a dict.
        "--reward-key", "score",
        "--eval-reward-key", "score",
        "--num-rollout", str(args.num_rollout),
        "--rollout-batch-size", str(args.rollout_batch_size),
        "--n-samples-per-prompt", str(args.n_samples_per_prompt),
        "--rollout-max-response-len", str(args.rollout_max_response_len),
        "--rollout-temperature", str(args.rollout_temperature),
        "--global-batch-size", str(args.global_batch_size),
        "--balance-data",
        "--eval-interval", str(args.eval_interval),
        "--eval-prompt-data", "grader_score", args.eval_data,
        "--n-samples-per-eval-prompt", str(args.n_samples_per_eval_prompt),
        "--eval-max-response-len", str(args.eval_max_response_len),
        "--eval-top-p", str(args.eval_top_p),
        "--eval-temperature", str(args.eval_temperature),
        "--skip-eval-before-train",
        "--tensor-model-parallel-size", str(args.tensor_parallel),
        "--sequence-parallel",
        "--pipeline-model-parallel-size", str(args.pipeline_parallel),
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", str(args.recompute_num_layers),
        "--use-dynamic-batch-size",
        "--use-dynamic-global-batch-size",
        "--max-tokens-per-gpu", str(args.max_tokens_per_gpu),
        "--train-memory-margin-bytes", "2147483648",
        "--advantage-estimator", args.advantage_estimator,
        "--use-kl-loss",
        "--kl-loss-coef", str(args.kl_loss_coef),
        "--kl-loss-type", args.kl_loss_type,
        "--entropy-coef", str(args.entropy_coef),
        "--eps-clip", str(args.eps_clip),
        "--eps-clip-high", str(args.eps_clip_high),
        "--optimizer", "adam",
        "--lr", str(args.lr),
        "--lr-decay-style", args.lr_decay_style,
        "--weight-decay", str(args.weight_decay),
        "--adam-beta1", str(args.adam_beta1),
        "--adam-beta2", str(args.adam_beta2),
        "--rollout-num-gpus-per-engine", str(args.rollout_num_gpus_per_engine),
        "--sglang-mem-fraction-static", str(args.sglang_mem_fraction),
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--attention-backend", "flash",
        "--custom-rollout-log-function-path", "rollout_logger.log_rollout_data",
        "--custom-eval-rollout-log-function-path", "rollout_logger.log_eval_rollout_data",
        # Bridge slime's wandb/TB log path to AzureML so train/kl_loss and
        # train/entropy_loss reach the Foundry portal Metrics tab (filtered
        # by ROLLOUT_VERBOSE_LOGS — only the 4 customer-facing scalars land
        # when verbose_logs=False).
        "--custom-megatron-init-path", "rollout_logger.init_slime_metric_bridge",
    ]

    if args.qk_layernorm:
        cmd.append("--qk-layernorm")

    if args.add_qkv_bias:
        cmd.append("--add-qkv-bias")

    # Add the Retail multi-turn generator after the shared Slime args.
    if args.custom_generate_path:
        cmd.extend(["--custom-generate-function-path", args.custom_generate_path])

    # Optional DAPO filtering spends rollout budget on prompts with signal.
    if getattr(args, "over_sampling_batch_size", None):
        cmd.extend(["--over-sampling-batch-size", str(args.over_sampling_batch_size)])
    if getattr(args, "dynamic_sampling_filter_path", None):
        cmd.extend(["--dynamic-sampling-filter-path", args.dynamic_sampling_filter_path])

    # Let model scripts override defaults so this entrypoint can support multiple Qwen shapes.
    if args.model_script:
        cmd.extend(_load_model_args(args.model_script))

    return cmd


def _load_model_args(model_script: str) -> list[str]:
    """Load model architecture arguments from a Slime model script."""
    import shlex
    script_path = Path(model_script).resolve()
    if not script_path.exists():
        logger.warning(f"--model_script {script_path} not found; skipping arch overrides")
        return []
    cmd_str = (
        f'set -e; source "{script_path}"; '
        f'printf "%s\\n" "${{MODEL_ARGS[@]}}"'
    )
    try:
        out = subprocess.run(
            ["bash", "-c", cmd_str],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to source {script_path}: {e.stderr}")
        return []
    args_list = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    logger.info(f"Loaded {len(args_list)} model args from {script_path}")
    return args_list


def run_training(args):
    """Submit Slime training to the existing Ray cluster."""
    # Log Ray cluster status
    try:
        result = subprocess.run(
            ["ray", "status"],
            capture_output=True, text=True, timeout=30,
        )
        logger.info(f"Ray cluster status:\n{result.stdout}")
        if result.stderr.strip():
            logger.warning(f"Ray status stderr:\n{result.stderr}")
    except Exception as e:
        logger.warning(f"Failed to get Ray status: {e}")

    has_nvlink = detect_nvlink()
    logger.info(f"NVLink detected: {has_nvlink}")

    # Include the code-dataset directory so custom RM modules are importable
    code_dataset_dir = str(Path(__file__).resolve().parent)

    aml_env_vars = {k: v for k, v in os.environ.items() if k.startswith("AZUREML")}
    for key in ("PROJECT_ENDPOINT", "MANAGED_IDENTITY_CLIENT_ID",
                "RETAIL_USER_LLM", "RETAIL_USER_LLM_TEMPERATURE",
                "RETAIL_MAX_TURNS", "RETAIL_SOLO_MODE",
                "ROLLOUT_VERBOSE_LOGS"):
        val = os.environ.get(key, "")
        if val:
            aml_env_vars[key] = val

    runtime_env = {
        "working_dir": code_dataset_dir,
        # Canary slime-310:build-demo-26-1 ships compatible numpy + transformer_engine
        # baked in, so no per-actor pip install is needed. `pip_check=False` and
        # `packages: []` keep the runtime_env block in place (for the
        # `--extra-index-url` channel registration if a future image needs it)
        # without forcing any installs.
        "pip": {
            "packages": [],
            "pip_check": False,
        },
        "env_vars": {
            **aml_env_vars,
            "PYTHONPATH": f"{code_dataset_dir}:/opt/retail:/opt/Megatron-LM/:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1" if has_nvlink else "0",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "WANDB_MODE": "offline",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ROLLOUT_LOG_DIR": args.rollout_log_dir,
            "RAY_TEMP_DIR": args.ray_temp,
            "RAY_TMPDIR": args.ray_temp,
        }
    }

    cmd = build_slime_command(args)

    # Let Ray auto-discover the cluster via the RAY_ADDRESS env var
    os.environ["RAY_ADDRESS"] = "auto"
    ray_cmd = [
        "ray", "job", "submit",
        f"--runtime-env-json={json.dumps(runtime_env)}",
        "--",
    ] + cmd

    logger.info(f"Submitting SLIME training to Ray cluster at {os.environ['RAY_ADDRESS']}:")
    logger.info(f"  {' '.join(cmd[:10])} ...")

    process = subprocess.Popen(
        ray_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(line, flush=True)

    return_code = process.wait()
    logger.info(f"Training exited with code {return_code}")
    return return_code


def fix_cudnn_path():
    try:
        import nvidia.cudnn
        cudnn_file = getattr(nvidia.cudnn, "__file__", None)
        if cudnn_file:
            cudnn_lib = os.path.join(os.path.dirname(cudnn_file), "lib")
        else:
            raise ValueError("nvidia.cudnn.__file__ is None")
    except (ImportError, AttributeError, ValueError):
        cudnn_lib = "/opt/micromamba/envs/slime/lib/python3.10/site-packages/nvidia/cudnn/lib"

    if os.path.isdir(cudnn_lib):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cudnn_lib not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{ld}"
            logger.info(f"cuDNN lib added to LD_LIBRARY_PATH: {cudnn_lib}")
    else:
        logger.warning(f"cuDNN lib not found at {cudnn_lib}")


def parse_args():
    p = argparse.ArgumentParser(description="SLIME Retail Retail Agent RFT on Azure ML")

    p.add_argument(
        "--hf_checkpoint",
        default=None,
        help="HF model checkpoint path (mounted input or local path)",
    )
    p.add_argument(
        "--ref_checkpoint",
        default=None,
        help="Reference checkpoint path (mounted input or local path)",
    )
    p.add_argument(
        "--hf_model_id",
        default=os.getenv("SLIME_HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        help="Hugging Face model ID to download when --hf_checkpoint is not provided",
    )
    p.add_argument(
        "--ref_model_id",
        default=os.getenv("SLIME_REF_MODEL_ID"),
        help="Reference model ID; defaults to --hf_model_id",
    )

    p.add_argument("--train_data", required=True, help="Training JSONL (train_retail.jsonl)")
    p.add_argument("--eval_data", required=True, help="Eval JSONL (val_retail.jsonl)")
    p.add_argument("--model_data", default=None, help="Model data directory (mounted input)")
    p.add_argument("--output_dir", default=None, help="Output/save directory")
    p.add_argument("--rollout_log_dir", default=None, help="Rollout log directory")
    p.add_argument("--model_output", default=None, help="Final model checkpoint directory")
    p.add_argument("--ray_temp", default=None, help="Ray temp directory")
    p.add_argument(
        "--hf_home_dir",
        default=None,
        help="Override HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE. "
             "Use a shared mount visible across Ray nodes (e.g. an output uri_folder).",
    )

    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--num_nodes", type=int, default=2)

# Defaults match the smaller Qwen shape unless submit_job overrides them.
    p.add_argument("--hidden_size", type=int, default=3584)
    p.add_argument("--num_attention_heads", type=int, default=28)
    p.add_argument("--num_layers", type=int, default=28)
    p.add_argument("--ffn_hidden_size", type=int, default=18944)
    p.add_argument("--norm_epsilon", type=float, default=1e-06)
    p.add_argument("--rotary_base", type=int, default=1000000)
    p.add_argument("--num_query_groups", type=int, default=4)
    p.add_argument("--kv_channels", type=int, default=128,
                   help="Per-head dimension (head_dim from HF config)")
    p.add_argument("--seq_length", type=int, default=32768)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--qk_layernorm", action="store_true",
                   help="Pass --qk-layernorm to Megatron/SLIME, required for Qwen3 models")
    p.add_argument("--add_qkv_bias", action="store_true",
                   help="Pass --add-qkv-bias to Megatron/SLIME, required by some Qwen2.x models")
    p.add_argument(
        "--model_script", default=None,
        help=(
            "Path to a slime model script (e.g. "
            "/opt/slime/scripts/models/qwen3.5-35B-A3B.sh) — its MODEL_ARGS "
            "array is bash-sourced and appended to the slime training "
            "command, so we can swap models without editing this script."
        ),
    )
    p.add_argument(
        "--sft_lora_path", default=None,
        help=(
            "Optional: path to an SFT LoRa adapter folder (mounted as a "
            "Foundry asset input).  When set, the script merges the LoRA "
            "into the base HF model and uses the merged checkpoint as "
            "--hf_checkpoint (slime needs a full HF model)."
        ),
    )

    # Keep these keys aligned with the uploaded Retail JSONL schema.
    p.add_argument("--input_key", default="input")
    p.add_argument("--metadata_key", default="metadata")
    p.add_argument("--custom_rm_path", default="retail_reward.custom_rm")
    p.add_argument("--custom_generate_path",
                   default="retail_generate.custom_generate",
                   help="Custom multi-turn generate function for retail")
    p.add_argument("--num_rollout", type=int, default=200)
    p.add_argument("--rollout_batch_size", type=int, default=4)
    p.add_argument("--n_samples_per_prompt", type=int, default=4)
    p.add_argument("--rollout_max_response_len", type=int, default=4096)
    p.add_argument("--rollout_temperature", type=float, default=0.8)
    p.add_argument("--global_batch_size", type=int, default=16)
    p.add_argument("--save_interval", type=int, default=10)

    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--n_samples_per_eval_prompt", type=int, default=1)  # 20 val tasks × 1 sample = 20 eval rollouts
    p.add_argument("--eval_max_response_len", type=int, default=4096)
    p.add_argument("--eval_top_p", type=float, default=0.95)
    p.add_argument("--eval_temperature", type=float, default=0.8,
                   help="Sampling temperature used for eval rollouts. "
                        "Defaults to 0.8 (same as train); set to 0.3 to "
                        "get a less stark exploration gap vs greedy.")

    p.add_argument("--tensor_parallel", type=int, default=4)
    p.add_argument("--pipeline_parallel", type=int, default=1)
    p.add_argument("--recompute_num_layers", type=int, default=2)
    p.add_argument("--max_tokens_per_gpu", type=int, default=8192)

    # Algorithm
    p.add_argument("--advantage_estimator", default="grpo")
    p.add_argument("--kl_loss_coef", type=float, default=0.01)
    p.add_argument("--kl_loss_type", default="low_var_kl")
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--eps_clip_high", type=float, default=0.28)

    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--lr_decay_style", default="constant")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)

    p.add_argument("--rollout_num_gpus_per_engine", type=int, default=4)
    p.add_argument("--sglang_mem_fraction", type=float, default=0.85)

    p.add_argument("--rollout_num_gpus", type=int, default=None)
    p.add_argument("--update_weights_interval", type=int, default=8)
    p.add_argument("--start_rollout_id", type=int, default=0)

# Pass optional rollout-filter knobs through without teaching this wrapper their internals.
    p.add_argument("--over_sampling_batch_size", type=int, default=None,
                   help="DAPO-style: oversample prompts and filter by reward std")
    p.add_argument("--dynamic_sampling_filter_path", type=str, default=None,
                   help="slime filter module path (e.g. "
                        "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)")

    return p.parse_args()


def check_disk_availability():
    try:
        result = subprocess.run(
            ["df", "-h"],
            capture_output=True, text=True, timeout=10,
        )
        logger.info(f"Disk availability (df -h):\n{result.stdout}")
    except subprocess.TimeoutExpired:
        logger.warning("Timeout expired while checking disk availability.")
    except Exception as e:
        logger.error(f"Error while checking disk availability: {e}")


def verify_retail_install():
    """Verify that Retail tools are importable and the bundled database loads."""
    # Check both image-baked and uploaded code paths before import.
    import sys as _sys
    from pathlib import Path as _Path
    for cand in ("/opt/retail", str(_Path(__file__).resolve().parent)):
        if cand and _Path(cand).exists() and cand not in _sys.path:
            _sys.path.insert(0, cand)
    try:
        import retail_tools
        logger.info(
            f"retail_tools loaded: {len(retail_tools.TOOL_SCHEMAS)} schemas, "
            f"{len(retail_tools.TOOL_FUNCTIONS)} functions"
        )
        db = retail_tools._load_db()
        logger.info(
            f"retail_db: {len(db['orders'])} orders, "
            f"{len(db['customers'])} customers, {len(db['products'])} products"
        )
    except ImportError as e:
        logger.error(f"retail_tools NOT IMPORTABLE: {e}")
        raise
    except Exception as e:
        logger.error(f"retail DB load failed: {e}")
        raise


def install_custom_pip_packages():
    """Install runtime packages missing from the base Slime image."""
    # Foundry home dirs persist, so remove stale user-site packages that shadow the image stack.
    user_site = Path("/home/aiscuser/.local/lib/python3.10/site-packages")
    if user_site.exists():
        for pat in ("huggingface_hub*", "transformers*", "peft*"):
            for victim in user_site.glob(pat):
                logger.info(f"Removing stale user-site shadow: {victim}")
                subprocess.run(["rm", "-rf", str(victim)], check=False)

    # Let pip use user-site for additive packages because the base venv is root-owned.

    # Pin numpy because Megatron still asserts the 1.x ABI at startup.
    logger.info("Pinning numpy<2 for Megatron compat...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "numpy<2"],
            check=True, capture_output=True, text=True, timeout=300,
        )
        import numpy as _np
        import importlib
        importlib.reload(_np)
        logger.info(f"numpy={_np.__version__}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"numpy pin failed: {e.stderr[:400]}")
    except Exception as e:
        logger.warning(f"numpy pin unexpected error: {e}")

    # transformer_engine is baked into the canary slime-310:build-demo-26-1 image
    # (cu12-compatible wheels for both the `transformer-engine` metapackage and
    # the `transformer-engine-cu12` backend). Skip the runtime pip install.

    required_packages = [
        ("azureml-core", "azureml.core"),
        ("openai", "openai"),
        ("huggingface-hub", "huggingface_hub"),
        ("azure-ai-projects>=1.0.0,<2.0.0", "azure.ai.projects"),
        ("azure-identity>=1.23.0,<2.0.0", "azure.identity"),
        # peft is needed for the optional --sft_lora_path warm-start
        # (merge_lora_into_base loads PeftModel).
        ("peft", "peft"),
    ]
    # Retail tools are pure in-process Python, so optional deps stay empty.
    optional_packages: list = []

    for package_name, module_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            logger.info(f"Installing {package_name}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", package_name],
                check=True,
                capture_output=True, text=True,
            )
            import importlib
            importlib.invalidate_caches()

    for package_name, module_name, extra_flags in optional_packages:
        try:
            __import__(module_name)
            continue
        except ImportError:
            pass
        logger.info(f"Installing optional {package_name}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *extra_flags, package_name],
            check=False,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            import importlib
            importlib.invalidate_caches()
            logger.info(f"Installed optional {package_name} successfully.")
        else:
            logger.warning(
                f"Optional package {package_name} failed to install "
                f"(exit={result.returncode}). Continuing with reward fallback. "
                f"stderr: {result.stderr.strip()[:500]}"
            )



def launch_rollout_dashboard(args):
    code_dataset_dir = str(Path(__file__).resolve().parent)
    DASHBOARD_PORT = 8501
    USER_COMMAND = (
    "set -euxo pipefail; "
    "export ROLLOUT_LOG_DIR=\"" + args.rollout_log_dir + "\"; "
    "python -m pip install --upgrade pip; "
    "grep -vE '^(ipython|ipywidgets|jupyter)' \"" + code_dataset_dir + "/requirements.txt\" > /tmp/dashboard-requirements.txt; "
    "echo '[dashboard-debug] filtered requirements:'; cat /tmp/dashboard-requirements.txt; "
    "python -m pip install --user -r /tmp/dashboard-requirements.txt; "
    "mkdir -p /tmp/dashboard-logs; "
    "echo \"[dashboard-debug] starting streamlit on 0.0.0.0:" + str(DASHBOARD_PORT) + "\"; "
    "nohup python -m streamlit run \"" + code_dataset_dir + "/dashboard.py\" "
    "--server.address 0.0.0.0 "
    "--server.port " + str(DASHBOARD_PORT) + " "
    "--server.headless true "
    "--server.enableCORS false "
    "--server.enableXsrfProtection false "
    "--server.enableWebsocketCompression false "
    "--browser.gatherUsageStats false "
    "> /tmp/dashboard-logs/streamlit.log 2>&1 & "
    "STREAMLIT_PID=$!; "
    "echo \"[dashboard-debug] streamlit pid=$STREAMLIT_PID\"; "
    "sleep 5; ps -p $STREAMLIT_PID || (echo '[dashboard-debug] streamlit died early'; cat /tmp/dashboard-logs/streamlit.log; exit 1); "
    )
    try:
        result = subprocess.run(
            ["bash", "-c", USER_COMMAND],
            capture_output=True, text=True, timeout=30,
        )
        logger.info(f"Dashboard launch output:\n{result.stdout}")
        if result.stderr.strip():
            logger.warning(f"Dashboard launch stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("Timeout expired while launching dashboard.")
    except Exception as e:
        logger.error(f"Error while launching dashboard: {e}")



def main():
    check_disk_availability()

    # Install missing runtime packages before Slime imports its stack.
    install_custom_pip_packages()

    # Fail fast if neither the image nor uploaded code exposes Retail tools.
    verify_retail_install()

    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path.cwd() / "output")
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"--output_dir not set; defaulting to {args.output_dir}")

    if args.rollout_log_dir is None:
        args.rollout_log_dir = str(Path(args.output_dir) / "rollout_logs")
        os.makedirs(args.rollout_log_dir, exist_ok=True)
        logger.info(f"--rollout_log_dir not set; defaulting to {args.rollout_log_dir}")

    if args.ray_temp is None:
        args.ray_temp = str(Path(args.output_dir) / "ray_temp")
        os.makedirs(args.ray_temp, exist_ok=True)
        logger.info(f"--ray_temp not set; defaulting to {args.ray_temp}")

    if args.model_output is None:
        args.model_output = str(Path(args.output_dir) / "model")
        os.makedirs(args.model_output, exist_ok=True)
        logger.info(f"--model_output not set; defaulting to {args.model_output}")

    args.ref_model_id = args.ref_model_id or args.hf_model_id

    # Split GPUs between actor and rollout pools for async mode.
    total_gpus = args.num_gpus * args.num_nodes

    if args.rollout_num_gpus is None:
        if args.num_nodes >= 2:
            rollout_nodes = args.num_nodes // 2
            args.rollout_num_gpus = rollout_nodes * args.num_gpus
        else:
            args.rollout_num_gpus = max(
                args.rollout_num_gpus_per_engine,
                args.num_gpus // 2,
            )
        logger.info(
            f"--rollout_num_gpus not set; defaulting to {args.rollout_num_gpus} "
            f"(of {total_gpus} total GPUs)"
        )

    actor_total_gpus = total_gpus - args.rollout_num_gpus
    if args.num_nodes >= 2:
        actor_nodes = actor_total_gpus // args.num_gpus
        args.actor_gpus_per_node = args.num_gpus
        args.actor_num_nodes = max(1, actor_nodes)
    else:
        args.actor_gpus_per_node = actor_total_gpus
        args.actor_num_nodes = 1

    if args.actor_gpus_per_node < args.tensor_parallel:
        raise ValueError(
            f"Not enough GPUs for actor: actor_gpus_per_node={args.actor_gpus_per_node} "
            f"< tensor_parallel={args.tensor_parallel}"
        )
    logger.info(
        f"Async GPU layout: {args.actor_num_nodes} actor node(s) × "
        f"{args.actor_gpus_per_node} GPUs + {args.rollout_num_gpus} rollout GPUs"
    )

    # Resolve checkpoints after cache dirs exist so Ray workers share downloads.
    if args.model_data:
        logger.info(f"Using mounted model data: {args.model_data}")
        args.hf_checkpoint = args.hf_checkpoint or args.model_data
        args.ref_checkpoint = args.ref_checkpoint or args.model_data
    else:
        _, hf_hub_cache, _ = configure_hf_cache_dirs(args.hf_home_dir)
        args.hf_checkpoint = resolve_checkpoint_source(
            checkpoint_path=args.hf_checkpoint,
            model_id=args.hf_model_id,
            hf_hub_cache=hf_hub_cache,
            role="hf",
        )
        args.ref_checkpoint = resolve_checkpoint_source(
            checkpoint_path=args.ref_checkpoint,
            model_id=args.ref_model_id,
            hf_hub_cache=hf_hub_cache,
            role="ref",
        )

    # Merge the SFT adapter only into the actor so KL stays anchored to the base model.
    if getattr(args, "sft_lora_path", None):
        merged_path = merge_lora_into_base(
            base_path=args.hf_checkpoint,
            lora_path=args.sft_lora_path,
            output_dir=Path(args.hf_home_dir or "./hf_cache") / "merged_actor",
        )
        logger.info(f"Using SFT-merged checkpoint for actor: {merged_path}")
        args.hf_checkpoint = str(merged_path)

    logger.info("=" * 60)
    logger.info("SLIME Retail Retail Agent RFT on Azure ML — ASYNC MODE")
    logger.info("=" * 60)
    logger.info(f"HF Model ID:       {args.hf_model_id or 'n/a'}")
    logger.info(f"Ref Model ID:      {args.ref_model_id or 'n/a'}")
    logger.info(f"Model Data:        {args.model_data or 'n/a (using HF download)'}")
    logger.info(f"HF Checkpoint:     {args.hf_checkpoint}")
    logger.info(f"Ref Checkpoint:    {args.ref_checkpoint}")
    logger.info(f"Train Data:        {args.train_data}")
    logger.info(f"Eval Data:         {args.eval_data}")
    logger.info(f"Output Dir:        {args.output_dir}")
    logger.info(f"Model Output:      {args.model_output}")
    logger.info(f"Ray Temp:          {args.ray_temp}")
    logger.info(f"Rollout Log Dir:   {args.rollout_log_dir}")
    logger.info(f"Custom RM:         {args.custom_rm_path}")
    logger.info(f"Custom Generate:   {args.custom_generate_path}")
    logger.info(f"GPUs/node:         {args.num_gpus}")
    logger.info(f"Actor nodes:       {args.actor_num_nodes}")
    logger.info(f"Actor GPUs/node:   {args.actor_gpus_per_node}")
    logger.info(f"Nodes:             {args.num_nodes}")
    logger.info(f"Rollout GPUs:      {args.rollout_num_gpus}")
    logger.info(f"Wt sync interval:  {args.update_weights_interval}")
    logger.info("=" * 60)

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["WANDB_MODE"] = "offline"

    launch_rollout_dashboard(args)

    # Patch library lookup before Ray launches worker processes.
    fix_cudnn_path()

    # Submit to the Ray cluster that Foundry already started for the job.
    return_code = run_training(args)

    logger.info("=" * 60)
    logger.info(f"Training finished with exit code {return_code}")
    logger.info("=" * 60)

    sys.exit(return_code)


if __name__ == "__main__":
    main()
