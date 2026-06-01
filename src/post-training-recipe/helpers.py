"""helpers.py — SDK-based helpers for the Foundry RFT demo notebook.

External dependencies (public packages only):
    pip install -r requirements.txt

Sections
--------
Constants        Foundry project, compute, identity, tag defaults.
Dataset          upload_dataset()                     (SDK)
Job submission   body_to_command_job() · submit_job_via_sdk()   (SDK)
GPU Placement    GPULayout · configure_gpu_layout()
Job Setup        build_retail_request_body()
"""

from __future__ import annotations

import os
import random
import string
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Mapping, NamedTuple

# ──────────────────────────────────────────────────────────────────────────── Constants
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default) or default

FOUNDRY_PROJECT_ENDPOINT: str = _env(
    "FOUNDRY_TRAININGJOB__PROJECT_SCOPED_ENDPOINT",
    "https://fdp-command-job-canary.services.ai.azure.com"
    "/api/projects/fdp-command-job-UMI-CANARY-proj",
)

FOUNDRY_STORAGE_CONNECTION_NAME: str = _env(
    "FOUNDRY_TRAININGJOB__STORAGE_CONNECTION_NAME",
    "shjondhalews748870471c2lmuh",
)

COMPUTE_CLUSTER_GPU: str = "testfoundrywcusclustergpu"

COMPUTE_CLUSTER_CPU: str = "testfoundrywcusclustercpu"

COMPUTE_CLUSTER_A100: str = "Atlas100"

_SUBSCRIPTION_ID = _env(
    "FOUNDRY_TRAININGJOB__SUBSCRIPTION_ID",
    "72c03bf3-4e69-41af-9532-dfcdc3eefef4",
)

_RG_COMPUTE = "computeinstance-e2e"

_ACCOUNT_COMPUTE = "ragarg-wc-res"

_GPU_COMPUTE_ID: str = _env(
    "FOUNDRY_TRAININGJOB__GPU_COMPUTE_ID",
    (
        f"/subscriptions/{_SUBSCRIPTION_ID}/resourcegroups/{_RG_COMPUTE}"
        f"/providers/microsoft.cognitiveservices/accounts/{_ACCOUNT_COMPUTE}"
        f"/computes/{COMPUTE_CLUSTER_GPU}"
    ),
)

_CPU_COMPUTE_ID: str = _env(
    "FOUNDRY_TRAININGJOB__COMPUTE_ID",
    (
        f"/subscriptions/{_SUBSCRIPTION_ID}/resourcegroups/{_RG_COMPUTE}"
        f"/providers/microsoft.cognitiveservices/accounts/{_ACCOUNT_COMPUTE}"
        f"/computes/{COMPUTE_CLUSTER_CPU}"
    ),
)

_A100_COMPUTE_ID: str = _env(
    "FOUNDRY_TRAININGJOB__SMI_COMPUTE_ID",
    (
        f"/subscriptions/{_SUBSCRIPTION_ID}/resourcegroups/{_RG_COMPUTE}"
        f"/providers/microsoft.cognitiveservices/accounts/{_ACCOUNT_COMPUTE}"
        f"/computes/{COMPUTE_CLUSTER_A100}"
    ),
)

_AISUPERCOMPUTER: dict[str, str] = {
    "imageVersion": "",
    "slaTier": "Premium",
    "priority": "high",
}

COMPUTE_CONFIG_BY_CLUSTER: dict[str, dict[str, Any]] = {
    COMPUTE_CLUSTER_GPU: {
        "computeId": _GPU_COMPUTE_ID,
        "resources": {
            "instanceCount": 1,
            "instanceType": "Singularity.ND96r_H100_v5",
            "properties": {"AISuperComputer": _AISUPERCOMPUTER},
        },
    },
    COMPUTE_CLUSTER_CPU: {
        "computeId": _CPU_COMPUTE_ID,
        "resources": {
            "instanceCount": 1,
            "instanceType": "Singularity.D4_v3",
            "properties": {"AISuperComputer": _AISUPERCOMPUTER},
        },
    },
    COMPUTE_CLUSTER_A100: {
        "computeId": _A100_COMPUTE_ID,
        "resources": {
            "instanceCount": 1,
            "instanceType": _env(
                "FOUNDRY_TRAININGJOB__SMI_INSTANCE_TYPE",
                "Singularity.ND96am_A100_v4-n1",
            ),
            "properties": {"AISuperComputer": _AISUPERCOMPUTER},
        },
    },
}

IDENTITY_UAI: str = _env(
    "FOUNDRY_TRAININGJOB__IDENTITY_UAI",
    (
        f"/subscriptions/{_SUBSCRIPTION_ID}"
        f"/resourceGroups/shared-finetuning-rg"
        "/providers/Microsoft.ManagedIdentity"
        "/userAssignedIdentities/fdp-command-job-test-mi-canary"
    ),
)

DEFAULT_JOB_ENV_VARS: dict[str, str] = {
    "AZUREML_CR_BOOTSTRAPPER_CONFIG_OVERRIDE": (
        '{"capabilities_registry": {"registry": {"url": "fdpcommandbtestcanary.azurecr.io",'
        ' "username": null, "password": null},'
        ' "repo_prefix": "cr2026051502_singularity_bootstrapper",'
        ' "regional_tag_prefix": false}}'
    ),
    "SINGULARITY_SIDECAR_CONSOLIDATION": "false",
}

CANARY_TAGS: dict[str, str] = {"env": "CANARY"}

DEFAULT_IDENTITY_CLIENT_ID: Final[str] = "9959f773-bc1b-48c5-8546-b80100d2cf18"

def upload_dataset(
    local_path: str | Path,
    *,
    dataset_name: str,
    dataset_version: str,
    project_endpoint: str = FOUNDRY_PROJECT_ENDPOINT,
    connection_name: str = FOUNDRY_STORAGE_CONNECTION_NAME,
) -> str:
    """Upload a local file or folder to Foundry storage and return its asset URI.

    Uses the ``azure-ai-projects`` SDK (``AIProjectClient``) to perform the
    upload.  The asset URI returned is suitable for use as a job input
    ``uri`` value.

    Parameters
    ----------
    local_path:
        Absolute or relative path to the local file or directory to upload.
    dataset_name:
        Foundry dataset asset name (e.g. ``"demo-train-data"``).
    dataset_version:
        Version string applied to the created dataset asset.  Use a unique
        value (e.g. a millisecond timestamp) to avoid collisions.
    project_endpoint:
        Project-scoped Foundry endpoint URL.  Defaults to
        ``FOUNDRY_PROJECT_ENDPOINT``.
    connection_name:
        Storage connection name registered in the Foundry project.  Defaults
        to ``FOUNDRY_STORAGE_CONNECTION_NAME``.

    Returns
    -------
    str
        Foundry asset URI of the uploaded dataset
        (e.g. ``azureai://accounts/.../data/<name>/versions/<version>``).

    Raises
    ------
    ValueError
        If ``local_path`` does not exist or the service response lacks an ID.
    """
    from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
    from azure.identity import AzureCliCredential  # type: ignore[import-untyped]

    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"local_path does not exist: {path}")

    credential = AzureCliCredential(process_timeout=30)
    with AIProjectClient(endpoint=project_endpoint, credential=credential) as client:
        if path.is_file():
            dataset = client.datasets.upload_file(
                name=dataset_name,
                version=dataset_version,
                file_path=str(path),
                connection_name=connection_name,
            )
        else:
            dataset = client.datasets.upload_folder(
                name=dataset_name,
                version=dataset_version,
                folder=str(path),
                connection_name=connection_name,
            )

    dataset_id: str | None = getattr(dataset, "id", None)
    if not dataset_id:
        raise ValueError("Service response did not include a dataset id.")
    return dataset_id

def _job_name_suffix(length: int = 4) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=length))

def body_to_command_job(body: Mapping[str, Any]):
    """Translate the recipe's internal Foundry REST body dict into an
    ``azure.ai.projects.models.CommandJob`` model.

    The recipe expresses the job in Foundry's REST shape (camelCase
    ``properties`` block); ``CommandJob`` takes flat snake_case kwargs.
    """
    from azure.ai.projects.models import CommandJob, JobResourceConfiguration  # type: ignore[import-untyped]

    props = body["properties"]
    resources_dict = props.get("resources") or {}
    resources = JobResourceConfiguration(
        instance_count=resources_dict.get("instanceCount"),
        instance_type=resources_dict.get("instanceType"),
        properties=resources_dict.get("properties") or {},
    )
    cj_kwargs: dict = {
        "command": props.get("command"),
        "compute": props.get("computeId"),
        "description": props.get("description"),
        "display_name": props.get("displayName"),
        "environment_image_reference": props.get("environmentImageReference"),
        "environment_variables": props.get("environmentVariables") or {},
        "inputs": props.get("inputs") or {},
        "outputs": props.get("outputs") or {},
        "resources": resources,
        "tags": props.get("tags") or {},
        "user_assigned_identity_id": props.get("userAssignedIdentityId"),
        # Foundry job services (e.g. ``foundry-rollout-browser`` Streamlit
        # sidecar on port 8501). Must be forwarded explicitly because
        # ``CommandJob`` does not pull through unknown fields from the body
        # dict — without this line the registration in
        # ``submit_job.py:_build_body_from_ids`` is silently dropped and the
        # portal never surfaces the public endpoint.
        "services": props.get("services") or None,
    }
    if props.get("distribution"):
        cj_kwargs["distribution"] = props["distribution"]
    if props.get("properties"):
        cj_kwargs["properties"] = props["properties"]
    cj_kwargs = {k: v for k, v in cj_kwargs.items() if v is not None}
    return CommandJob(**cj_kwargs)

def submit_job_via_sdk(
    request_body: Mapping[str, Any],
    *,
    project_endpoint: str,
    job_name_prefix: str = "job",
    job_name: str | None = None,
):
    """Submit a Foundry Command Job using the ``azure-ai-projects`` SDK.

    Uses ``AIProjectClient.beta.jobs.create_or_update`` with
    ``DefaultAzureCredential``. Returns the created ``CommandJob`` (which
    carries ``.name``, ``.id``, ``.foundry_portal_url``, ``.services``).
    """
    from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
    from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]

    effective_name = job_name or f"{job_name_prefix}-{_job_name_suffix()}"
    cmd_job = body_to_command_job(request_body)
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as client,
    ):
        return client.beta.jobs.create_or_update(name=effective_name, job=cmd_job)

class GPULayout(NamedTuple):
    """Validated async Ray GPU placement configuration.

    User-supplied fields
    --------------------
    n_nodes                    Total number of Ray nodes requested from Foundry.
    num_gpus                   GPUs per physical node (e.g. 8 for H100/A100).
    rollout_num_gpus           Total GPUs reserved for rollout inference.
                               Must equal ``rollout_nodes × num_gpus``.
    rollout_num_gpus_per_engine  GPUs per SGLang engine (intra-engine tensor parallelism).
    sglang_mem_fraction        Fraction of GPU memory dedicated to the SGLang KV-cache.
                               Rollout nodes are fully dedicated in async mode, so this
                               can be set higher (e.g. 0.85) than in sync mode.
    tensor_parallel            Tensor-parallel degree on actor nodes.
                               Must evenly divide ``actor_nodes × num_gpus``.
    max_tokens_per_gpu         Sequence-parallel token budget per GPU on actor nodes.

    Derived fields (computed by configure_gpu_layout)
    --------------------------------------------------
    rollout_nodes       Number of nodes assigned to rollout (= rollout_num_gpus // num_gpus).
    actor_nodes         Number of nodes assigned to the actor (= n_nodes - rollout_nodes).
    dp_degree           Data-parallel degree on actor nodes
                        (= actor_nodes × num_gpus / tensor_parallel).
    sglang_engine_count Number of SGLang inference engines on the rollout node(s)
                        (= rollout_num_gpus / rollout_num_gpus_per_engine).
    """

    # User-supplied
    n_nodes: int
    num_gpus: int
    rollout_num_gpus: int
    rollout_num_gpus_per_engine: int
    sglang_mem_fraction: float
    tensor_parallel: int
    max_tokens_per_gpu: int
    # Derived
    rollout_nodes: int
    actor_nodes: int
    dp_degree: int
    sglang_engine_count: int

def configure_gpu_layout(
    n_nodes: int,
    num_gpus: int,
    rollout_num_gpus: int,
    rollout_num_gpus_per_engine: int,
    sglang_mem_fraction: float,
    tensor_parallel: int,
    max_tokens_per_gpu: int,
) -> GPULayout:
    """Validate the async Ray GPU placement config, print a layout summary, and return a GPULayout.

    In **async GRPO** mode the Ray cluster is divided into two whole-node roles:

    * **Actor nodes** — run FSDP + ZeRO training with tensor parallelism.
    * **Rollout nodes** — run SGLang inference engines for response generation.

    Three invariants are checked; ``AssertionError`` is raised if any fails,
    so misconfiguration is caught before a job submission is attempted:

    1. At least one actor node must remain after reserving rollout nodes.
    2. ``tensor_parallel`` must evenly divide the total actor GPU count.
    3. ``rollout_num_gpus_per_engine`` must evenly divide ``rollout_num_gpus``.

    Parameters
    ----------
    n_nodes:
        Total number of Ray nodes to request from Foundry.
    num_gpus:
        GPUs per physical node (typically 8 for H100 / A100 nodes).
    rollout_num_gpus:
        Total GPUs dedicated to rollout inference.
        Must equal an integer multiple of ``num_gpus`` (whole-node assignment).
    rollout_num_gpus_per_engine:
        GPUs per SGLang inference engine (intra-engine tensor parallelism).
        Must evenly divide ``rollout_num_gpus``.
    sglang_mem_fraction:
        Fraction of GPU memory allocated to the SGLang KV-cache on rollout nodes.
    tensor_parallel:
        Tensor-parallel degree applied to actor nodes.
        Must evenly divide ``actor_nodes × num_gpus``.
    max_tokens_per_gpu:
        Sequence-parallel token budget per GPU on the actor node(s).

    Returns
    -------
    GPULayout
        Immutable record containing both the user-supplied values and all
        derived quantities (actor_nodes, dp_degree, sglang_engine_count, …).
    """
    rollout_nodes = rollout_num_gpus // num_gpus
    actor_nodes = n_nodes - rollout_nodes
    actor_gpus_total = actor_nodes * num_gpus
    dp_degree = actor_gpus_total // tensor_parallel
    sglang_engine_count = rollout_num_gpus // rollout_num_gpus_per_engine

    assert actor_nodes > 0, (
        f"No actor nodes: N_NODES={n_nodes}, rollout_nodes={rollout_nodes}. "
        "Reduce ROLLOUT_NUM_GPUS or increase N_NODES."
    )
    assert actor_gpus_total % tensor_parallel == 0, (
        f"TENSOR_PARALLEL={tensor_parallel} must divide actor_gpus_total={actor_gpus_total}."
    )
    assert rollout_num_gpus % rollout_num_gpus_per_engine == 0, (
        f"ROLLOUT_NUM_GPUS_PER_ENGINE={rollout_num_gpus_per_engine} must divide "
        f"ROLLOUT_NUM_GPUS={rollout_num_gpus}."
    )

    print(
        f"Async GPU layout: {actor_nodes} actor node ({actor_gpus_total} GPUs, "
        f"TP={tensor_parallel} DP={dp_degree}) + "
        f"{rollout_nodes} rollout node ({rollout_num_gpus} GPUs, "
        f"{sglang_engine_count}\u00d7TP{rollout_num_gpus_per_engine} SGLang engines)"
    )
    for node_idx in range(n_nodes):
        if node_idx < actor_nodes:
            tp_tags = "  ".join(f"T{g % tensor_parallel}" for g in range(num_gpus))
            print(
                f"  Node {node_idx} [ACTOR  ]  {tp_tags}  "
                f"(TP={tensor_parallel}, DP rank {node_idx}  FSDP+ZeRO)"
            )
        else:
            eng_tags = "  ".join(
                f"E{g // rollout_num_gpus_per_engine}" for g in range(num_gpus)
            )
            print(
                f"  Node {node_idx} [ROLLOUT]  {eng_tags}  "
                f"({sglang_engine_count} SGLang engines, mem={sglang_mem_fraction})"
            )

    return GPULayout(
        n_nodes=n_nodes,
        num_gpus=num_gpus,
        rollout_num_gpus=rollout_num_gpus,
        rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
        sglang_mem_fraction=sglang_mem_fraction,
        tensor_parallel=tensor_parallel,
        max_tokens_per_gpu=max_tokens_per_gpu,
        rollout_nodes=rollout_nodes,
        actor_nodes=actor_nodes,
        dp_degree=dp_degree,
        sglang_engine_count=sglang_engine_count,
    )

def build_retail_request_body(
    *,
    gpu: GPULayout,
    train_dataset_id: str,
    code_dataset_id: str,
    model_dataset_id: str,
    job_name_prefix: str,
    compute_cluster: str,
    environment_id: str,
    global_batch_size: int,
    learning_rate: str,
    num_rollouts: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    rollout_max_response_len: int,
    rollout_temperature: str,
    hf_model_id: str,
    hf_home: str,
    project_endpoint: str,
    managed_identity_client_id: str | None = None,
    retail_user_llm: str = "",
    retail_user_llm_temperature: str = "0.7",
    retail_max_turns: str = "30",
    retail_solo_mode: str = "true",
    sft_lora_dataset_id: str | None = None,
) -> dict:
    """Assemble the Foundry Command Job request body for retail RL training.

    Similar to ``build_request_body`` but adapted for retail:
    - Calls ``retail_slime_train.py`` instead of ``slime_rl_train.py``
    - Uses retail environment variables instead of grader/judge vars
    - Retail data files (train_retail.jsonl, val_retail.jsonl)
    """
    if managed_identity_client_id is None:
        managed_identity_client_id = DEFAULT_IDENTITY_CLIENT_ID
    timestamp = int(datetime.now().timestamp() * 1000)
    hf_hub_cache = f"{hf_home}/hub"
    transformers_cache = f"{hf_home}/transformers"

    command_parts = [
        'python "${{inputs.code_dataset}}/retail_slime_train.py"',
        '--train_data "${{inputs.train_dataset}}/train_retail.jsonl"',
        '--eval_data "${{inputs.train_dataset}}/val_retail.jsonl"',
        '--model_data "${{inputs.model_dataset}}"',
        '--output_dir "${{outputs.checkpoints}}"',
        '--rollout_log_dir "${{outputs.rollouts}}"',
        '--model_output "${{outputs.model_output}}"',
        '--ray_temp "${{outputs.ray_temp}}"',
        '--hf_home_dir "${{outputs.hf_cache}}"',
        f"--num_gpus {gpu.num_gpus}",
        f"--num_nodes {gpu.n_nodes}",
        f"--n_samples_per_prompt {n_samples_per_prompt}",
        f"--rollout_batch_size {rollout_batch_size}",
        f"--rollout_max_response_len {rollout_max_response_len}",
        f"--rollout_temperature {rollout_temperature}",
        f"--global_batch_size {global_batch_size}",
        f"--max_tokens_per_gpu {gpu.max_tokens_per_gpu}",
        f"--sglang_mem_fraction {gpu.sglang_mem_fraction}",
        f"--rollout_num_gpus {gpu.rollout_num_gpus}",
        f"--tensor_parallel {gpu.tensor_parallel}",
        f"--num_rollout {num_rollouts}",
        f"--lr {learning_rate}",
    ]
    if sft_lora_dataset_id:
        command_parts.append('--sft_lora_path "${{inputs.sft_lora_dataset}}"')
    command = " ".join(command_parts)

    compute_config = deepcopy(COMPUTE_CONFIG_BY_CLUSTER[compute_cluster])
    compute_config["resources"]["instanceCount"] = gpu.n_nodes

    env_vars = {
        **deepcopy(DEFAULT_JOB_ENV_VARS),
        "PYTHONPATH": "/opt/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "SLIME_HF_MODEL_ID": hf_model_id,
        "SLIME_REF_MODEL_ID": hf_model_id,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "TRANSFORMERS_CACHE": transformers_cache,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PROJECT_ENDPOINT": project_endpoint,
        "MANAGED_IDENTITY_CLIENT_ID": managed_identity_client_id,
        # Retail-specific environment variables
        "RETAIL_MAX_TURNS": retail_max_turns,
        "RETAIL_SOLO_MODE": retail_solo_mode,
    }

    # Only set user LLM vars if a user simulator is configured
    if retail_user_llm:
        env_vars["RETAIL_USER_LLM"] = retail_user_llm
        env_vars["RETAIL_USER_LLM_TEMPERATURE"] = retail_user_llm_temperature

    return {
        "properties": {
            "jobType": "Command",
            "description": "Retail retail agent: async GRPO fine-tuning with env reward",
            "displayName": f"{job_name_prefix}-n{gpu.n_nodes}",
            "command": command,
            "environmentImageReference": environment_id,
            **compute_config,
            "userAssignedIdentityId": IDENTITY_UAI,
            "properties": {
                "_azureml.LogTrainingMetricsToAzMon": "true",
            },
            "environmentVariables": env_vars,
            "inputs": {
                "train_dataset": {
                    "jobInputType": "uri_folder",
                    "uri": train_dataset_id,
                    "mode": "ReadOnlyMount",
                },
                "model_dataset": {
                    "jobInputType": "uri_folder",
                    "uri": model_dataset_id,
                    "mode": "ReadOnlyMount",
                },
                "code_dataset": {
                    "jobInputType": "uri_folder",
                    "uri": code_dataset_id,
                    "mode": "ReadOnlyMount",
                },
                **(
                    {"sft_lora_dataset": {
                        "jobInputType": "uri_folder",
                        "uri": sft_lora_dataset_id,
                        "mode": "ReadOnlyMount",
                    }} if sft_lora_dataset_id else {}
                ),
            },
            "outputs": {
                "model_output": {
                    "assetName": f"retail-model-{timestamp}",
                    "jobOutputType": "safetensors_model",
                    "mode": "ReadWriteMount",
                },
                "checkpoints": {
                    "assetName": f"retail-checkpoints-{timestamp}",
                    "jobOutputType": "uri_folder",
                    "mode": "ReadWriteMount",
                },
                "rollouts": {
                    "assetName": f"retail-rollouts-{timestamp}",
                    "jobOutputType": "uri_folder",
                    "mode": "ReadWriteMount",
                },
                "ray_temp": {
                    "assetName": f"retail-ray-temp-{timestamp}",
                    "jobOutputType": "uri_folder",
                    "mode": "ReadWriteMount",
                },
                "hf_cache": {
                    "assetName": f"retail-hf-cache-{timestamp}",
                    "jobOutputType": "uri_folder",
                    "mode": "ReadWriteMount",
                },
            },
            "distribution": {
                "distributionType": "Ray",
                "port": 6379,
                "address": None,
                "includeDashboard": "True",
                "headNodeAdditionalArgs": "",
                "workerNodeAdditionalArgs": "",
            },
            "tags": {
                **deepcopy(CANARY_TAGS),
                "scenario": "retail-retail-agent",
                "trainDatasetId": train_dataset_id,
                "codeDatasetId": code_dataset_id,
                "modelDatasetId": model_dataset_id,
            },
        }
    }
