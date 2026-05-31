"""Submit a Zava SFT / async-GRPO Foundry training job.

Recovered from the actual payload used to submit `zv-rft-32b-sft-h-mf2y`
(and `zv-rft-32b-sft-a-3nwl`). The original ad-hoc submission was lost, so
this file captures it as a reproducible script.

Usage:
    python submit_sft.py --cluster h100
    python submit_sft.py --cluster a100
    python submit_sft.py --cluster h100 --name-prefix zv-rft-32b-sft-h
"""
from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

PROJECT_ENDPOINT = (
    "https://foundry-training-pilot.services.ai.azure.com"
    "/api/projects/foundry-training-pilot-proj"
)
API_VERSION = "2026-01-15-preview"

SUBSCRIPTION = "72c03bf3-4e69-41af-9532-dfcdc3eefef4"
COMPUTE_RG = "computeinstance-e2e"
COMPUTE_ACCOUNT = "ragarg-wc-res"
UMI = (
    f"/subscriptions/{SUBSCRIPTION}/resourcegroups/shared-finetuning-rg"
    "/providers/microsoft.managedidentity/userassignedidentities/fdp-training-pilot-umi"
)

CLUSTERS = {
    "h100": {
        "name": "testfoundrywcusclustergpu",
        "instance_type": "Singularity.ND96r_H100_v5",
        "name_prefix": "zv-rft-32b-sft-h",
        "display": "zv-rft-32b-sft-h-n4",
    },
    "a100": {
        "name": "testfoundrywcusclustera100",
        "instance_type": "Singularity.ND96amsr_A100_v4",
        "name_prefix": "zv-rft-32b-sft-a",
        "display": "zv-rft-32b-sft-a-n4",
    },
}

# Foundry dataset URIs currently in use (bump versions when you re-upload).
DATASETS = {
    "train_dataset": "azureai://accounts/foundry-training-pilot/projects/foundry-training-pilot-proj/data/zava-train-data/versions/1780023586082",
    "code_dataset":  "azureai://accounts/foundry-training-pilot/projects/foundry-training-pilot-proj/data/zava-code/versions/1780023586082",
    "sft_lora_dataset": "azureai://accounts/foundry-training-pilot/projects/foundry-training-pilot-proj/data/zava-sft-qwen3-32b-a100-checkpoints-1779968060624/versions/20260528113423611",
}

# ENV_IMAGE = "mcr.microsoft.com/azureml/curated/slime-pytorch-2.9-cuda12.8:3" # have dependency issues.
ENV_IMAGE = "fdpcommandbtestcanary.azurecr.io/azureml/slime-310:9-candidate-v5-cp2-trace1"
ENV_VARS = {
    "AZUREML_CR_BOOTSTRAPPER_CONFIG_OVERRIDE": json.dumps({
        "capabilities_registry": {
            "registry": {"url": "fdpcommandbtestcanary.azurecr.io", "username": None, "password": None},
            "repo_prefix": "cr2026051502_singularity_bootstrapper",
            "regional_tag_prefix": False,
        }
    }),
    "SINGULARITY_SIDECAR_CONSOLIDATION": "false",
    "PYTHONPATH": "/opt/zava:/opt/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "SLIME_HF_MODEL_ID": "Qwen/Qwen3-32B",
    "SLIME_REF_MODEL_ID": "Qwen/Qwen3-32B",
    "HF_HOME": "./hf_cache",
    "HF_HUB_CACHE": "./hf_cache/hub",
    "TRANSFORMERS_CACHE": "./hf_cache/transformers",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PROJECT_ENDPOINT": PROJECT_ENDPOINT,
    "MANAGED_IDENTITY_CLIENT_ID": "9959f773-bc1b-48c5-8546-b80100d2cf18",
    "TAUBENCH_MAX_TURNS": "10",
    "TAUBENCH_SOLO_MODE": "true",
    "TAUBENCH_USER_LLM": "unused",
    "TAUBENCH_USER_LLM_TEMPERATURE": "0.0",
    "ZAVA_MAX_TRAJ_TOKENS": "16384",
    "ZAVA_MAX_TURNS": "10",
    "ZAVA_ENV_STEP_TIMEOUT": "30",
    "TAUBENCH_DOMAIN": "zava",
}

# Training command. ${{inputs.*}} / ${{outputs.*}} are Foundry placeholders
# that the runtime substitutes before launching bash.
COMMAND = (
    'python "${{inputs.code_dataset}}/zava_slime_train.py"'
    ' --train_data "${{inputs.train_dataset}}/zava_train.jsonl"'
    ' --eval_data "${{inputs.train_dataset}}/zava_val.jsonl"'
    ' --output_dir "${{outputs.checkpoints}}"'
    ' --rollout_log_dir "${{outputs.rollouts}}"'
    ' --model_output "${{outputs.model_output}}"'
    ' --ray_temp "${{outputs.ray_temp}}"'
    ' --hf_home_dir "${{outputs.hf_cache}}"'
    " --num_gpus 8 --num_nodes 4"
    " --n_samples_per_prompt 4 --rollout_batch_size 2"
    " --rollout_max_response_len 1536 --rollout_temperature 0.8"
    " --global_batch_size 16 --max_tokens_per_gpu 512"
    " --sglang_mem_fraction 0.85 --rollout_num_gpus 16"
    " --tensor_parallel 8 --num_rollout 369 --lr 5e-7"
    ' --sft_lora_path "${{inputs.sft_lora_dataset}}"'
    " --kl_loss_coef 0.05 --entropy_coef 0.0 --recompute_num_layers 16"
    " --model_script /opt/slime/scripts/models/qwen3-32B.sh"
    " --hidden_size 5120 --num_attention_heads 40 --num_layers 64"
    " --ffn_hidden_size 27648 --num_query_groups 8 --vocab_size 151936"
    " --qk_layernorm --eval_temperature 0.3 --n_samples_per_eval_prompt 1"
)


def _suffix(n: int = 4) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def token() -> str:
    out = subprocess.check_output(
        ["az", "account", "get-access-token", "--resource", "https://ai.azure.com",
         "--query", "accessToken", "-o", "tsv"],
        shell=True,
    )
    return out.decode().strip()


def build_payload(cluster_key: str, instance_count: int = 4) -> dict:
    c = CLUSTERS[cluster_key]
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    asset_v = f"{version}"
    asset_base = f"taubench-{version}"

    return {
        "properties": {
            "description": "Taubench retail agent: async GRPO fine-tuning with env reward",
            "tags": {
                "env": "PILOT",
                "scenario": "zava-rft",
                "domain": "zava",
                "variant": "image-bake",
                "agent": "Qwen/Qwen3-32B",
            },
            "properties": {},  # Foundry rejects azureml.* / _azureml.* user keys
            "displayName": c["display"],
            "computeId": (
                f"/subscriptions/{SUBSCRIPTION}/resourcegroups/{COMPUTE_RG}"
                f"/providers/microsoft.cognitiveservices/accounts/{COMPUTE_ACCOUNT}"
                f"/computes/{c['name']}"
            ),
            "experimentName": "Default",
            "isArchived": False,
            "jobType": "Command",
            "resources": {
                "instanceCount": instance_count,
                "instanceType": c["instance_type"],
                "properties": {
                    "AISuperComputer": {
                        "interactive": False,
                        "slaTier": "Premium",
                        "imageVersion": "",
                        "scalePolicy": {
                            "autoScaleIntervalInSec": 120,
                            "maxInstanceTypeCount": instance_count,
                            "minInstanceTypeCount": instance_count,
                        },
                    }
                },
                "shmSize": "2g",
            },
            "command": COMMAND,
            "environmentImageReference": ENV_IMAGE,
            "inputs": {
                k: {"uri": v, "mode": "ReadOnlyMount", "jobInputType": "uri_folder"}
                for k, v in DATASETS.items()
            },
            "outputs": {
                "model_output":  {"assetName": f"{asset_base}-model",       "assetVersion": asset_v, "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
                "checkpoints":   {"assetName": f"{asset_base}-checkpoints", "assetVersion": asset_v, "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
                "rollouts":      {"assetName": f"{asset_base}-rollouts",    "assetVersion": asset_v, "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
                "ray_temp":      {"assetName": f"{asset_base}-ray-temp",    "assetVersion": asset_v, "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
                "hf_cache":      {"assetName": f"{asset_base}-hf-cache",    "assetVersion": asset_v, "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
            },
            "distribution": {
                "distributionType": "Ray",
                "port": 6379,
                "includeDashboard": True,
                "headNodeAdditionalArgs": "",
                "workerNodeAdditionalArgs": "",
            },
            "environmentVariables": dict(ENV_VARS),
            "userAssignedIdentityId": UMI,
        }
    }


def submit(name: str, payload: dict) -> dict:
    url = f"{PROJECT_ENDPOINT}/jobs/{name}?api-version={API_VERSION}"
    tok = token()
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}\n{e.read().decode()[:3000]}")
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", choices=list(CLUSTERS), required=True)
    ap.add_argument("--name", default=None, help="explicit job name (else auto-generated)")
    ap.add_argument("--instance-count", type=int, default=4)
    args = ap.parse_args()

    c = CLUSTERS[args.cluster]
    name = args.name or f"{c['name_prefix']}-{_suffix()}"
    payload = build_payload(args.cluster, args.instance_count)
    print(f"Submitting {name} on {c['name']} ({args.instance_count}x {c['instance_type']})")
    resp = submit(name, payload)
    print(f"OK -> {resp.get('name')}  status={resp.get('properties', {}).get('status')}")
    print(
        f"Portal: https://eastus2euap.ai.azure.com/nextgen/r/csA7805pQa-VMt_Nw-7-9A,"
        "shared-finetuning-rg,,foundry-training-pilot,foundry-training-pilot-proj"
        f"/build/train/jobs/{name}/details"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
