"""
Foundry RFT submission setup for the Retail post-purchase task.
This module builds the CommandJob body and uploads run artifacts when requested.
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import helpers as fh  # noqa: E402

PROJECT_ENDPOINT = (
    "https://foundry-training-pilot.services.ai.azure.com"
    "/api/projects/foundry-training-pilot-proj"
)
API_VERSION = "2026-01-15-preview"
PILOT_UAI = (
    "/subscriptions/72c03bf3-4e69-41af-9532-dfcdc3eefef4"
    "/resourcegroups/shared-finetuning-rg"
    "/providers/microsoft.managedidentity"
    "/userassignedidentities/fdp-training-pilot-umi"
)
PILOT_UAI_CLIENT_ID = "9959f773-bc1b-48c5-8546-b80100d2cf18"
COMPUTE_CLUSTER = "Atlas100"
GPU_COMPUTE_ID = (
    "/subscriptions/72c03bf3-4e69-41af-9532-dfcdc3eefef4"
    "/resourcegroups/computeinstance-e2e"
    "/providers/microsoft.cognitiveservices"
    "/accounts/ragarg-wc-res"
    f"/computes/{COMPUTE_CLUSTER}"
)
INSTANCE_TYPE = "Singularity.ND96am_A100_v4-n1"
STORAGE_CONNECTION_NAME = "fdptrainingpilot"

JOB_NAME_PREFIX = "Retail-RFT-Qwen14B"

# Warm-start from the SFT LoRa so GRPO starts from the demo policy.
# No project-specific default — the notebook supplies the SFT adapter URI per
# submission via `submit_job(sft_lora_uri=...)`. Override here only if you
# want a fallback for the `python submit_job.py` CLI path.
SFT_LORA_DATASET_ID = None
ENVIRONMENT_ID = "fdpcommandbtestcanary.azurecr.io/azureml/slime-310:build-demo-26-1"
# Z-derived build-demo-26-1 image — bakes in compatible numpy + transformer_engine,
# so no runtime pip pins are needed anywhere (neither at the launcher layer
# nor per-Ray-actor in retail_slime_train.py). The container's Singularity
# bootstrapper needs to know which registry to pull capability sidecars
# from; we set AZUREML_CR_BOOTSTRAPPER_CONFIG_OVERRIDE below in
# _build_body_from_ids.

HF_MODEL_ID = "Qwen/Qwen3-14B"
HF_HOME = "./hf_cache"

# Split GPUs between actor and rollout pools to keep async training saturated.
N_NODES = 4
NUM_GPUS = 8
ROLLOUT_NUM_GPUS = 16  # Reserve half the cluster for rollout workers.
ROLLOUT_NUM_GPUS_PER_ENGINE = 8  # Use full-node engines to avoid cross-node tensor parallelism.
SGLANG_MEM_FRACTION = 0.85
TENSOR_PARALLEL = 8              # Match one tensor-parallel group per actor node.
MAX_TOKENS_PER_GPU = 1024        # Keep microbatches small enough for 14B context windows.

# Use the tuned GRPO recipe from the strongest prior run.
GLOBAL_BATCH_SIZE = 32
LEARNING_RATE = "6e-7"
NUM_ROLLOUTS = 200  # Cover the compact dataset several times without overlong jobs.
ROLLOUT_BATCH_SIZE = 4
N_SAMPLES_PER_PROMPT = 8  # Keep GRPO groups large enough for stable advantages.
EPS_CLIP_HIGH = "0.40"   # Asymmetric DAPO clip upper bound (default in trainer is 0.28).
ROLLOUT_MAX_RESPONSE_LEN = 1536
ROLLOUT_TEMPERATURE = "0.7"  # Lower temperature keeps tool-use rollouts easier to grade.

# Use light KL regularization so the SFT warm-start can still explore.
KL_LOSS_COEF = "0.03"
ENTROPY_COEF = "0.0"

# Keep default metrics focused unless a debug run needs the full set.
VERBOSE_LOGS = False


def build_request_body(upload: bool = True):
    """Build the Foundry CommandJob request body for a Retail RFT run."""
    gpu = fh.configure_gpu_layout(
        N_NODES, NUM_GPUS,
        ROLLOUT_NUM_GPUS, ROLLOUT_NUM_GPUS_PER_ENGINE, SGLANG_MEM_FRACTION,
        TENSOR_PARALLEL, MAX_TOKENS_PER_GPU,
    )

    cur_dir = ROOT
    if upload:
        dataset_version = str(int(datetime.now().timestamp() * 1000))
        print(f"Uploading code + data (dataset_version={dataset_version}) ...")
        train_dataset_id = fh.upload_dataset(
            cur_dir / "demo-artifacts" / "data",
            dataset_name="retail-train-data",
            dataset_version=dataset_version,
            project_endpoint=PROJECT_ENDPOINT,
            connection_name=STORAGE_CONNECTION_NAME,
        )
        print(f"Train data : {train_dataset_id}")
        code_dataset_id = fh.upload_dataset(
            cur_dir / "demo-artifacts" / "code",
            dataset_name="retail-code",
            dataset_version=dataset_version,
            project_endpoint=PROJECT_ENDPOINT,
            connection_name=STORAGE_CONNECTION_NAME,
        )
        print(f"Code       : {code_dataset_id}")
    else:
        train_dataset_id = "azureai://placeholder/retail-train-data/dryrun"
        code_dataset_id  = "azureai://placeholder/retail-code/dryrun"
    model_dataset_id = code_dataset_id  # Reuse a harmless input because weights come from Hugging Face.

    fh.IDENTITY_UAI = PILOT_UAI
    fh.DEFAULT_IDENTITY_CLIENT_ID = PILOT_UAI_CLIENT_ID
    fh.FOUNDRY_STORAGE_CONNECTION_NAME = STORAGE_CONNECTION_NAME
    fh.COMPUTE_CONFIG_BY_CLUSTER[COMPUTE_CLUSTER] = {
        "computeId": GPU_COMPUTE_ID,
        "resources": {
            "instanceType": INSTANCE_TYPE,
            "instanceCount": N_NODES,
            "properties": {
                "AISuperComputer": {
                    "interactive": False,
                    "slaTier": "Premium",
                    "imageVersion": "",
                    "scalePolicy": {
                        "autoScaleIntervalInSec": 120,
                        "maxInstanceTypeCount": N_NODES,
                        "minInstanceTypeCount": N_NODES,
                    },
                },
            },
        },
    }

    # Reuse the tau-bench helper because it only templates command args and env vars.
    body = fh.build_retail_request_body(
        gpu=gpu,
        job_name_prefix=JOB_NAME_PREFIX,
        compute_cluster=COMPUTE_CLUSTER,
        environment_id=ENVIRONMENT_ID,
        train_dataset_id=train_dataset_id,
        code_dataset_id=code_dataset_id,
        model_dataset_id=model_dataset_id,
        global_batch_size=GLOBAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_rollouts=NUM_ROLLOUTS,
        rollout_batch_size=ROLLOUT_BATCH_SIZE,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        rollout_max_response_len=ROLLOUT_MAX_RESPONSE_LEN,
        rollout_temperature=ROLLOUT_TEMPERATURE,
        hf_model_id=HF_MODEL_ID,
        hf_home=HF_HOME,
        project_endpoint=PROJECT_ENDPOINT,
        managed_identity_client_id=PILOT_UAI_CLIENT_ID,
        # Keep legacy helper inputs harmless because Retail has no user simulator.
        retail_solo_mode="true",
        retail_max_turns="10",
        retail_user_llm="unused",
        retail_user_llm_temperature="0.0",
        sft_lora_dataset_id=SFT_LORA_DATASET_ID,
    )

    # Drop unused LiteLLM settings so the container environment stays unambiguous.
    envs = body["properties"]["environmentVariables"]
    for k in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"):
        envs.pop(k, None)

    # Prefer the image-baked Retail tools, while keeping uploaded code as a fallback.
    envs["PYTHONPATH"] = "/opt/retail:" + envs.get("PYTHONPATH", "/opt/Megatron-LM/")
    # Gate the rollout_logger AzureML metric stream. When false (default) only
    # eval/reward_mean, train/response_length_mean, train/entropy_loss, and
    # train/kl_loss reach the Studio Metrics tab.
    envs["RETAIL_VERBOSE_LOGS"] = "true" if VERBOSE_LOGS else "false"
    # Back-compat alias kept for older custom hooks that may still read it.
    envs["ROLLOUT_VERBOSE_LOGS"] = "true" if VERBOSE_LOGS else "false"
    # Patch helper defaults because this recipe ships Retail-named data files.
    import re as _re
    cmd = body["properties"]["command"]
    cmd = _re.sub(r"train_retail\.jsonl", "retail_train.jsonl", cmd)
    cmd = _re.sub(r"val_retail\.jsonl",   "retail_val.jsonl",   cmd)
    # Route the helper command to the Retail entrypoint and callbacks.
    cmd = _re.sub(r"retail_slime_train\.py", "retail_slime_train.py", cmd)
    cmd = _re.sub(r"retail_generate\.custom_generate", "retail_generate.custom_generate", cmd)
    cmd = _re.sub(r"retail_reward\.custom_rm",        "retail_reward.custom_rm",       cmd)
    # Avoid a stale model input because the job downloads weights from HF.
    cmd = _re.sub(r'\s*--model_data\s+"[^"]*"', "", cmd)

    # Append knobs the shared helper does not expose yet.
    cmd += f" --kl_loss_coef {KL_LOSS_COEF} --entropy_coef {ENTROPY_COEF}"
    cmd += " --recompute_num_layers 16"

    # Source the model script first so Slime gets its expected defaults.
    cmd += " --model_script /opt/slime/scripts/models/qwen3-14B.sh"

    # Keep explicit Qwen3-14B shape flags as a fallback for older images.
    cmd += (
        " --hidden_size 5120"
        " --num_attention_heads 40"
        " --num_layers 40"
        " --ffn_hidden_size 17408"
        " --num_query_groups 8"
        " --vocab_size 151936"
        " --qk_layernorm"
    )

    # Make eval deterministic so dashboard changes reflect policy changes.
    cmd += " --eval_temperature 0.0"
    cmd += " --eval_top_p 1.0"
    cmd += " --n_samples_per_eval_prompt 2"
    cmd += " --eval_interval 5"

    # Skip zero-advantage prompt groups to spend rollouts on useful gradients.
    cmd += " --dynamic_sampling_filter_path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    cmd += " --over_sampling_batch_size 8"

    # Asymmetric DAPO clip — let positive-advantage updates take bigger upside
    # steps. Default in retail_slime_train.py is 0.28; matches zv-rft-14b-v6c2-mofd.
    cmd += f" --eps_clip_high {EPS_CLIP_HIGH}"

    # Bound trajectories so long tool loops cannot exceed Slime sequence length.
    envs["RETAIL_MAX_TRAJ_TOKENS"] = "16384"
    envs["RETAIL_MAX_TURNS"]       = "10"
    envs["RETAIL_ENV_STEP_TIMEOUT"] = "30"
    envs["RETAIL_DOMAIN"]      = "retail"  # Preserve the helper contract for downstream tags.

    # Canary slime-310:build-demo-26-1 is pulled from the fdpcommandbtestcanary
    # ACR; tell the Singularity bootstrapper which registry+prefix to fetch
    # capability sidecars from.
    envs["AZUREML_CR_BOOTSTRAPPER_CONFIG_OVERRIDE"] = json.dumps({
        "capabilities_registry": {
            "registry": {
                "url": "fdpcommandbtestcanary.azurecr.io",
                "username": None,
                "password": None,
            },
            "repo_prefix": "cr2026051502_singularity_bootstrapper",
            "regional_tag_prefix": False,
        }
    })
    envs["SINGULARITY_SIDECAR_CONSOLIDATION"] = "false"

    body["properties"]["command"] = cmd
    body["properties"]["inputs"].pop("model_dataset", None)

    # Expose the in-job Streamlit dashboard ("Foundry Rollout Browser",
    # launched as a sidecar from retail_slime_train.py on port 8501) as a
    # custom Foundry job service so the AISC compute fabric routes a public
    # endpoint to it. Once the head node starts the streamlit process, the
    # service goes from NotStarted -> Running and the portal exposes the URL.
    body["properties"].setdefault("services", {})
    body["properties"]["services"]["foundry-rollout-browser"] = {
        "jobServiceType": "Custom",
        "port": 8501,
        "endpoint": "",
        "status": "NotStarted",
        "errorMessage": None,
        "properties": {"requiredPort": "8501"},
        "nodes": None,
    }

    body["properties"]["tags"] = {
        "scenario": "retail-rft",
        "domain": "retail",
        "variant": "image-bake",
        "agent": HF_MODEL_ID,
        "submittedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return body, train_dataset_id, code_dataset_id, model_dataset_id


def main():
    body, _train_id, _code_id, _model_id = build_request_body(upload=True)
    created = fh.submit_job_via_sdk(
        body,
        project_endpoint=PROJECT_ENDPOINT,
        job_name_prefix=JOB_NAME_PREFIX,
    )
    print(f"JOB_ID: {created.name}")
    portal = getattr(created, "foundry_portal_url", None)
    if portal:
        # SDK currently emits /build/train/jobs/<name> but the working portal route is /build/train/custom-jobs/<name>.
        portal = portal.replace("/build/train/jobs/", "/build/train/custom-jobs/")
        print(f"Portal: {portal}")
    return created


if __name__ == "__main__":
    main()
