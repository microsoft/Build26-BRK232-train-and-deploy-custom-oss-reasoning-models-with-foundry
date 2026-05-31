"""Helper module for the Retail RFT demo notebook.

Hides path setup, submission, and log-streaming plumbing so the notebook cells
stay one-liners.

Usage in notebook:
    from slime_rl_setup import ENV, setup_env, submit_job, stream_logs
    setup_env(project_endpoint="https://<account>.services.ai.azure.com/api/projects/<project>")
    submit_job(cluster_name="<your-compute-cluster>")
    stream_logs()

KERNEL / DEPENDENCY REQUIREMENTS
--------------------------------
``submit_job()`` uses ``azure-ai-projects==2.3.0a20260525001`` (prerelease)
+ ``azure-identity``. Both require Python 3.11+.

Install:
    pip install --pre azure-ai-projects==2.3.0a20260525001 azure-identity \
        --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple

Auth: ``az login`` to the tenant that owns your Foundry project.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Populated by setup_env(); imported elsewhere
ENV: SimpleNamespace = SimpleNamespace()


# ──────────────────── 1. Environment setup ───────────────────────────────────────────────────
def setup_env(
    project_endpoint: str,
    storage_connection_name: str | None = None,
    managed_identity_uai: str | None = None,
    managed_identity_client_id: str | None = None,
    verbose: bool = True,
) -> SimpleNamespace:
    """Discover paths relative to this notebook directory and validate them.

    Parameters
    ----------
    project_endpoint : str, required
        Full Foundry project endpoint URL. Format:
        ``https://<account>.services.ai.azure.com/api/projects/<project>``.
        Copy this from your Foundry project's Overview page. The endpoint is
        monkey-patched onto the recipe at submit time so uploads + job
        creation + log streaming all hit your project (not the recipe's
        baked-in default).
    storage_connection_name : str, optional
        Name of the Foundry workspace connection pointing at the storage
        account the project MSI can write to. Required unless your project
        already has access to the recipe's baked-in pilot storage account.
        Find this in Foundry portal → Management center → Connections.
    managed_identity_uai : str, optional
        Full ARM resource ID of the user-assigned managed identity the
        training container should run as. Format:
        ``/subscriptions/<sub>/resourcegroups/<rg>/providers/microsoft.managedidentity/userassignedidentities/<name>``.
        Required unless your project already has access to the recipe's
        baked-in pilot UAI.
    managed_identity_client_id : str, optional
        Client (application) ID of the same managed identity above. Required
        together with ``managed_identity_uai``.

    Assumes layout::

        <notebook-dir>/
            slime_rl_setup.py        (this file)
            post-training-recipe/    (RFT recipe — submit_job.py, demo-artifacts/, helpers.py)
    """
    if not project_endpoint or not str(project_endpoint).strip() or "<" in str(project_endpoint):
        raise ValueError(
            "setup_env requires a real project_endpoint URL — format: "
            "'https://<account>.services.ai.azure.com/api/projects/<project>'. "
            "Copy it from your Foundry project's Overview page."
        )
    if not str(project_endpoint).startswith("https://") or "/api/projects/" not in str(project_endpoint):
        raise ValueError(
            f"project_endpoint does not look like a Foundry project URL: {project_endpoint!r}. "
            "Expected 'https://<account>.services.ai.azure.com/api/projects/<project>'."
        )
    project_endpoint = str(project_endpoint).rstrip("/")
    # Derive project name (last URL segment) for display only.
    project_name = project_endpoint.rsplit("/", 1)[-1]

    # Reject obvious placeholder values for the optional overrides.
    for name, val in (
        ("storage_connection_name", storage_connection_name),
        ("managed_identity_uai", managed_identity_uai),
        ("managed_identity_client_id", managed_identity_client_id),
    ):
        if val is not None and ("<" in str(val) or not str(val).strip()):
            raise ValueError(f"{name} looks like a placeholder: {val!r}")

    # UAI + client ID must travel together — Azure rejects one without the other.
    if bool(managed_identity_uai) ^ bool(managed_identity_client_id):
        raise ValueError(
            "managed_identity_uai and managed_identity_client_id must be set together "
            "(or both left as None to keep the recipe's pilot defaults)."
        )

    nb_dir = Path.cwd().resolve()
    e = SimpleNamespace(
        nb_dir=nb_dir,
        recipe_dir=nb_dir / "post-training-recipe",
        project_endpoint=project_endpoint,
        project=project_name,
        storage_connection_name=storage_connection_name,
        managed_identity_uai=managed_identity_uai,
        managed_identity_client_id=managed_identity_client_id,
        job_id=None,
    )
    if not e.recipe_dir.is_dir():
        raise FileNotFoundError(f"[slime_rl_setup] missing recipe: {e.recipe_dir}")

    global ENV
    ENV = e
    if verbose:
        print(f"NB_DIR    : {e.nb_dir}")
        print(f"Recipe    : {e.recipe_dir}")
        print(f"Project   : {e.project}")
        print(f"Endpoint  : {e.project_endpoint}")
        if e.storage_connection_name:
            print(f"Storage   : {e.storage_connection_name}")
        if e.managed_identity_uai:
            print(f"UAI       : {e.managed_identity_uai}")
    return e


# ──────────────────── 2. Echo key submit parameters ───────────────────────────────────────────
KEY_NEEDLES = (
    "JOB_NAME_PREFIX", "NUM_ROLLOUTS", "MODEL_DATASET_ID", "SFT_LORA_DATASET_ID",
    "HF_MODEL_ID", "COMPUTE_CLUSTER", "N_NODES",
    "ROLLOUT_BATCH_SIZE", "N_SAMPLES_PER_PROMPT", "GLOBAL_BATCH_SIZE",
    "LEARNING_RATE", "KL_LOSS_COEF", "MAX_TOKENS_PER_GPU",
)


def show_submit_params() -> None:
    """Echo the most important constants from submit_job.py so we can verify
    the recipe before submission."""
    submit = ENV.recipe_dir / "submit_job.py"
    text = submit.read_text(encoding="utf-8")
    for needle in KEY_NEEDLES:
        for line in text.splitlines():
            if line.lstrip().startswith(needle):
                print(line.rstrip())
                break


# ──────────────────── 3. Submit the job via azure-ai-projects SDK ─────────────────────────────
_PARAM_GROUPS = {
    "sampling": {
        "rollout_temperature":      "ROLLOUT_TEMPERATURE",
        "rollout_max_response_len": "ROLLOUT_MAX_RESPONSE_LEN",
        "n_samples_per_prompt":     "N_SAMPLES_PER_PROMPT",
        "rollout_batch_size":       "ROLLOUT_BATCH_SIZE",
    },
    "training": {
        "learning_rate":     "LEARNING_RATE",
        "global_batch_size": "GLOBAL_BATCH_SIZE",
        "num_rollouts":      "NUM_ROLLOUTS",
        "kl_loss_coef":      "KL_LOSS_COEF",
        "entropy_coef":      "ENTROPY_COEF",
    },
    "rollout": {
        "rollout_num_gpus":            "ROLLOUT_NUM_GPUS",
        "rollout_num_gpus_per_engine": "ROLLOUT_NUM_GPUS_PER_ENGINE",
        "sglang_mem_fraction":         "SGLANG_MEM_FRACTION",
        "max_tokens_per_gpu":          "MAX_TOKENS_PER_GPU",
        "tensor_parallel":             "TENSOR_PARALLEL",
    },
}


def _apply_overrides(recipe_submit, group_label: str, overrides: dict | None) -> None:
    if not overrides:
        return
    mapping = _PARAM_GROUPS[group_label]
    applied = []
    for key, value in overrides.items():
        if key not in mapping:
            valid = ", ".join(sorted(mapping))
            raise ValueError(
                f"Unknown {group_label} parameter '{key}'. Valid keys: {valid}"
            )
        const_name = mapping[key]
        prev = getattr(recipe_submit, const_name, None)
        setattr(recipe_submit, const_name, value)
        applied.append(f"{const_name}: {prev!r} -> {value!r}")
    print(f"[{group_label}] " + "; ".join(applied))


def submit_job(
    cluster_name: str,
    warm_start: bool = True,
    sft_lora_uri: str | None = None,
    sampling: dict | None = None,
    training: dict | None = None,
    rollout: dict | None = None,
    verbose_logs: bool = False,
) -> str | None:
    """Submit the training job via
    ``azure-ai-projects.AIProjectClient.beta.training.jobs``.

    Parameters
    ----------
    cluster_name : str, required
        Name of the compute cluster you have access to in your Foundry
        project. No default is assumed — you must pass this explicitly.
        Submission fails with a clear Azure error if the cluster does not
        exist (or your identity has no access to it).
    warm_start : bool, default True
        If True, seed the RFT run from an existing SFT LoRA adapter. You must
        provide its asset URI via ``sft_lora_uri``. If False, perform a
        cold-start RFT run directly on the base model with no SFT seed (and
        ``sft_lora_uri`` is ignored).
    sft_lora_uri : str, optional
        The Foundry dataset URI of the SFT LoRA checkpoint to warm-start
        from (e.g. ``azureai://accounts/<acct>/projects/<proj>/data/<name>/versions/<ver>``).
        Required when ``warm_start=True``. Produce one by running the
        companion SFT notebook (``../post-training-sft-recipe/retail_sft_submit.ipynb``)
        and copying the resulting checkpoint asset URI.
    sampling : dict, optional
        Per-rollout sampling overrides. Recognised keys:
        ``rollout_temperature``, ``rollout_max_response_len``,
        ``n_samples_per_prompt``, ``rollout_batch_size``.
    training : dict, optional
        GRPO training overrides. Recognised keys:
        ``learning_rate``, ``global_batch_size``, ``num_rollouts``,
        ``kl_loss_coef``, ``entropy_coef``.
    rollout : dict, optional
        Rollout-engine / compute overrides. Recognised keys:
        ``rollout_num_gpus``, ``rollout_num_gpus_per_engine``,
        ``sglang_mem_fraction``, ``max_tokens_per_gpu``, ``tensor_parallel``.
    verbose_logs : bool, default False
        When False (default), the container only emits four customer-facing
        metrics: ``eval/reward_mean``, ``train/response_length_mean``,
        ``train/entropy_loss``, ``train/kl_loss``. Set True to surface the
        full debug metric set (per-dataset eval stats, reward min/max,
        rollout-time, all extra training metrics, etc.).

    Steps:
        1. Apply the optional overrides on the recipe.
        2. Call the recipe's ``build_request_body()`` to upload code + train
           datasets and assemble the job spec.
        3. Convert the spec into a ``CommandJob`` and submit through the SDK.
    """
    import importlib
    import secrets
    sys.path.insert(0, str(ENV.recipe_dir))
    # Reload helpers FIRST so the renamed surface (`build_retail_request_body`,
    # previously `build_taubench_request_body`) propagates through the kernel's
    # module cache. submit_job binds `fh = helpers` at import time; if helpers
    # is stale, the bound reference still resolves attributes from the old
    # module body.
    import helpers as _fh_reload  # type: ignore[import-not-found]
    importlib.reload(_fh_reload)
    import submit_job as recipe_submit  # type: ignore[import-not-found]
    importlib.reload(recipe_submit)  # Reset monkey-patched module constants to on-disk defaults so per-call overrides don't leak between cells.

    # Route uploads + submission + streaming to the project the user passed to
    # setup_env() instead of the recipe's baked-in pilot endpoint.
    if not getattr(ENV, "project_endpoint", None):
        raise RuntimeError(
            "submit_job() requires setup_env(project_endpoint=...) to be called first."
        )
    recipe_submit.PROJECT_ENDPOINT = ENV.project_endpoint
    print(f"Project endpoint: {recipe_submit.PROJECT_ENDPOINT}")

    # Same propagation for the project-scoped storage connection + container
    # identity. Without these, dataset upload fails (project MSI can't reach
    # the pilot storage) and the container fails to start (UAI not in tenant).
    if ENV.storage_connection_name:
        recipe_submit.STORAGE_CONNECTION_NAME = ENV.storage_connection_name
        print(f"Storage connection: {recipe_submit.STORAGE_CONNECTION_NAME}")
    if ENV.managed_identity_uai:
        recipe_submit.PILOT_UAI = ENV.managed_identity_uai
        recipe_submit.PILOT_UAI_CLIENT_ID = ENV.managed_identity_client_id
        print(f"Managed identity: {recipe_submit.PILOT_UAI_CLIENT_ID}")

    # Cluster is mandatory — fail fast with a clear message instead of silently
    # routing the job to whatever cluster happens to be baked into the recipe.
    if not cluster_name or not str(cluster_name).strip() or "<" in str(cluster_name):
        raise ValueError(
            "cluster_name is required — pass the name of a Foundry compute "
            "cluster you have access to (e.g. cluster_name=\"my-a100-cluster\"). "
            "If the cluster does not exist or you lack access, the SDK submit "
            "call will surface an Azure error."
        )

    recipe_submit.COMPUTE_CLUSTER = cluster_name
    # GPU_COMPUTE_ID was composed at import time from the recipe default; rebuild it.
    recipe_submit.GPU_COMPUTE_ID = (
        recipe_submit.GPU_COMPUTE_ID.rsplit("/computes/", 1)[0]
        + f"/computes/{cluster_name}"
    )
    print(f"Compute cluster: {cluster_name}")

    if warm_start:
        if not sft_lora_uri or not str(sft_lora_uri).strip():
            raise ValueError(
                "warm_start=True requires sft_lora_uri — pass the Foundry "
                "asset URI of your SFT LoRA checkpoint "
                "(e.g. 'azureai://accounts/<acct>/projects/<proj>/data/"
                "<name>/versions/<ver>'). Produce one by running the "
                "companion SFT notebook (../post-training-sft-recipe/"
                "retail_sft_submit.ipynb), or pass warm_start=False "
                "to cold-start from the base model."
            )
        recipe_submit.SFT_LORA_DATASET_ID = sft_lora_uri
        print(f"Warm start: seeding from SFT LoRA {sft_lora_uri}")
    else:
        if sft_lora_uri:
            print("Note: sft_lora_uri is ignored when warm_start=False.")
        recipe_submit.SFT_LORA_DATASET_ID = None
        print("Cold start: no SFT LoRA seed will be used.")

    _apply_overrides(recipe_submit, "sampling", sampling)
    _apply_overrides(recipe_submit, "training", training)
    _apply_overrides(recipe_submit, "rollout",  rollout)

    recipe_submit.VERBOSE_LOGS = bool(verbose_logs)
    print(f"Verbose logs: {recipe_submit.VERBOSE_LOGS}")

    print("Preparing job spec (uploads code + train datasets)...")
    body, _train_id, _code_id, _model_id = recipe_submit.build_request_body(upload=True)

    # Use the shared helpers.submit_job_via_sdk so the recipe CLI and notebook
    # paths use the exact same azure-ai-projects SDK call.
    import helpers as fh  # type: ignore[import-not-found]
    suffix = secrets.token_hex(2)
    job_name = f"{recipe_submit.JOB_NAME_PREFIX}-{suffix}"
    print(f"Submitting via azure-ai-projects SDK as: {job_name}")
    try:
        created = fh.submit_job_via_sdk(
            body,
            project_endpoint=recipe_submit.PROJECT_ENDPOINT,
            job_name=job_name,
        )
    except ImportError as e:
        raise ImportError(
            "azure-ai-projects (>=2.3.0a) and azure-identity are required. "
            "Install with: pip install --pre azure-ai-projects==2.3.0a20260525001 "
            "azure-identity --extra-index-url "
            "https://pkgs.dev.azure.com/azure-sdk/public/_packaging/"
            "azure-sdk-for-python/pypi/simple"
        ) from e
    ENV.job_id = job_name
    print(f"\nJOB_ID: {job_name}")
    foundry_portal_url = getattr(created, "foundry_portal_url", None)
    if foundry_portal_url:
        # SDK currently emits /build/train/jobs/<name> but the working portal route is /build/train/custom-jobs/<name>.
        foundry_portal_url = foundry_portal_url.replace("/build/train/jobs/", "/build/train/custom-jobs/")
        print(f"Portal: {foundry_portal_url}")
    return job_name


# ──────────────────── 4. Stream job logs via azure-ai-projects SDK ────────────────────────────
def stream_logs(job_id: str | None = None) -> None:
    """Tail the running job's log files to stdout until the job reaches a
    terminal state, using ``AIProjectClient.beta.jobs.stream()``.

    Parameters
    ----------
    job_id : str, optional
        The job name to tail. Defaults to the JOB_ID returned by the most
        recent ``submit_job()`` call (stored on ``ENV.job_id``).

    Raises
    ------
    RuntimeError
        If the job ends in a failed state (re-raised by the SDK).
    """
    job_id = job_id or ENV.job_id
    if not job_id:
        print("No JOB_ID set. Pass job_id= or run submit_job() first.")
        return

    sys.path.insert(0, str(ENV.recipe_dir))
    import submit_job as recipe_submit  # type: ignore[import-not-found]

    try:
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        raise ImportError(
            "azure-ai-projects (>=2.3.0a) and azure-identity are required. "
            "Install with: pip install --pre azure-ai-projects==2.3.0a20260525001 "
            "azure-identity --extra-index-url "
            "https://pkgs.dev.azure.com/azure-sdk/public/_packaging/"
            "azure-sdk-for-python/pypi/simple"
        ) from e

    # Use the project endpoint from setup_env if available, else fall back to the recipe default.
    project_endpoint = getattr(ENV, "project_endpoint", None) or recipe_submit.PROJECT_ENDPOINT
    print(f"Streaming logs for {job_id} (this blocks until the job reaches a terminal state)...")
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint,
                        credential=credential) as project_client,
    ):
        project_client.beta.jobs.stream(name=job_id)


# ──────────────────── 5. SFT LoRA pre-training step ──────────────────────────
# Reproduces the proven `zava-sft-qwen3-14b-a100-hp25` SFT-LoRA job
# (sft_tau2.py + transformers/trl/peft on a single H100 node).
# The code + data are already registered as Foundry assets under
# `zava-sft-qwen3-14b-a100-{code,data}/versions/20260530011712` in the
# Build-26-Demo-Proj project, so we reuse those URIs directly — no upload
# needed. Output `checkpoints` becomes the LoRA adapter URI that the
# subsequent RFT call consumes via `sft_lora_uri=...`.
SFT_DEFAULTS = SimpleNamespace(
    image="fdpcommandbtestcanary.azurecr.io/azureml/slime-310:9-candidate-v5-cp2-trace1",
    code_uri=("azureai://accounts/build-26-demo/projects/Build-26-Demo-Proj"
              "/data/zava-sft-qwen3-14b-a100-code/versions/20260530011712"),
    train_uri=("azureai://accounts/build-26-demo/projects/Build-26-Demo-Proj"
               "/data/zava-sft-qwen3-14b-a100-data/versions/20260530011712"),
    train_filename="zava_train_sft_v6.jsonl",
    base_model="Qwen/Qwen3-14B",
    instance_type="Singularity.ND96r_H100_v5",
    # sft_tau2.py knobs that match hp25 (only `epochs` differs by default)
    batch_size=1, grad_accum=16, lr="2e-05", max_seq_len=4096,
    lora_r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules="qwen", val_frac=0, logging_steps=2,
    save_strategy="epoch",
)


def _sft_command(epochs: float) -> str:
    """Build the bash command line for sft_tau2.py, mirroring hp25."""
    d = SFT_DEFAULTS
    pip_pkgs = (
        "'transformers>=4.51,<5' trl==0.11.4 peft==0.13.2 "
        "'datasets>=2.21,<3' 'accelerate>=1.0' 'bitsandbytes>=0.43' azureml-core"
    )
    return (
        "bash -c '"
        "pip install --quiet --upgrade pip "
        f"&& pip install --quiet {pip_pkgs} "
        "&& python \"${{inputs.code_dataset}}/sft_tau2.py\" "
        f"--train-jsonl \"${{{{inputs.train_dataset}}}}/{d.train_filename}\" "
        f"--base-model {d.base_model} "
        "--output-dir \"${{outputs.checkpoints}}\" "
        f"--epochs {epochs} --batch-size {d.batch_size} "
        f"--grad-accum {d.grad_accum} --lr {d.lr} --max-seq-len {d.max_seq_len} "
        f"--lora-r {d.lora_r} --lora-alpha {d.lora_alpha} "
        f"--lora-dropout {d.lora_dropout} --target-modules {d.target_modules} "
        f"--val-frac {d.val_frac} --logging-steps {d.logging_steps} "
        f"--save-strategy {d.save_strategy}"
        "'"
    )


def _compute_id_from_cluster(cluster_name: str) -> str:
    """Foundry compute IDs follow a fixed cog-services path; reuse the recipe's
    pattern so the user only passes the cluster short name."""
    sys.path.insert(0, str(ENV.recipe_dir))
    import submit_job as recipe_submit  # type: ignore[import-not-found]
    base = recipe_submit.GPU_COMPUTE_ID.rsplit("/computes/", 1)[0]
    return f"{base}/computes/{cluster_name}"


def submit_sft_lora(
    cluster_name: str,
    epochs: float = 1.0,
    instance_count: int = 1,
    name: str | None = None,
) -> str:
    """Submit the SFT-LoRA pre-training job (sft_tau2.py) on a single H100 node.

    Parameters
    ----------
    cluster_name : str, required
        Name of the H100-class compute cluster (e.g.
        ``"testfoundrywcusclustergpu"``) in your Foundry project.
    epochs : float, default 1.0
        Number of training epochs for the LoRA adapter. Use ``1.0`` to keep
        the demo turnaround tight; the production checkpoint used 5 epochs.
    instance_count : int, default 1
        Number of nodes. sft_tau2.py is single-node — leave at 1.
    name : str, optional
        Explicit job name. If omitted, a random 4-char suffix is appended to
        ``"zava-sft-14b-e{epochs}"``.

    Returns
    -------
    str
        The submitted job name. Pass it to ``wait_for_sft_lora()`` to block
        until completion and capture the output adapter URI.
    """
    if not cluster_name or "<" in str(cluster_name):
        raise ValueError("submit_sft_lora requires an explicit cluster_name")
    if not getattr(ENV, "project_endpoint", None):
        raise RuntimeError("Call setup_env() before submit_sft_lora().")

    import secrets
    from datetime import datetime, timezone

    job_name = name or f"zava-sft-14b-e{int(epochs)}-{secrets.token_hex(2)}"
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    asset_base = f"{job_name}-out"

    cj_props = {
        "description": f"Zava SFT-LoRA on Qwen3-14B ({epochs} epoch{'s' if epochs != 1 else ''})",
        "tags": {
            "scenario": "zava-sft-lora",
            "domain": "zava",
            "agent": SFT_DEFAULTS.base_model,
            "epochs": str(epochs),
        },
        "displayName": job_name,
        "computeId": _compute_id_from_cluster(cluster_name),
        "experimentName": "Default",
        "jobType": "Command",
        "resources": {
            "instanceCount": instance_count,
            "instanceType": SFT_DEFAULTS.instance_type,
            "properties": {
                "AISuperComputer": {
                    "interactive": False, "slaTier": "Premium", "imageVersion": "",
                    "scalePolicy": {
                        "autoScaleIntervalInSec": 120,
                        "maxInstanceTypeCount": instance_count,
                        "minInstanceTypeCount": instance_count,
                    },
                }
            },
            "shmSize": "2g",
        },
        "command": _sft_command(epochs),
        "environmentImageReference": SFT_DEFAULTS.image,
        "inputs": {
            "train_dataset": {"uri": SFT_DEFAULTS.train_uri,
                              "mode": "ReadOnlyMount", "jobInputType": "uri_folder"},
            "code_dataset":  {"uri": SFT_DEFAULTS.code_uri,
                              "mode": "ReadOnlyMount", "jobInputType": "uri_folder"},
        },
        "outputs": {
            "checkpoints": {"assetName": f"{asset_base}-checkpoints",
                            "assetVersion": version,
                            "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
            "hf_cache":    {"assetName": f"{asset_base}-hf-cache",
                            "assetVersion": version,
                            "mode": "ReadWriteMount", "jobOutputType": "uri_folder"},
        },
        "distribution": {
            "distributionType": "Ray", "port": 6379, "includeDashboard": True,
            "headNodeAdditionalArgs": "", "workerNodeAdditionalArgs": "",
        },
        "environmentVariables": {
            "AZUREML_CR_BOOTSTRAPPER_CONFIG_OVERRIDE": (
                '{"capabilities_registry": {"registry": {"url": '
                '"fdpcommandbtestcanary.azurecr.io", "username": null, '
                '"password": null}, "repo_prefix": '
                '"cr2026051502_singularity_bootstrapper", '
                '"regional_tag_prefix": false}}'
            ),
            "SINGULARITY_SIDECAR_CONSOLIDATION": "false",
            "HF_HOME": "./hf_cache",
            "HF_HUB_CACHE": "./hf_cache/hub",
            "TRANSFORMERS_CACHE": "./hf_cache/transformers",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PROJECT_ENDPOINT": ENV.project_endpoint,
        },
    }
    if getattr(ENV, "managed_identity_uai", None):
        cj_props["userAssignedIdentityId"] = ENV.managed_identity_uai
        cj_props["environmentVariables"]["MANAGED_IDENTITY_CLIENT_ID"] = (
            ENV.managed_identity_client_id or ""
        )

    body = {"properties": cj_props}
    import helpers as fh  # type: ignore[import-not-found]
    print(f"Submitting SFT-LoRA job {job_name} on {cluster_name} "
          f"({instance_count}x {SFT_DEFAULTS.instance_type}, epochs={epochs})")
    try:
        created = fh.submit_job_via_sdk(
            body, project_endpoint=ENV.project_endpoint, job_name=job_name,
        )
    except ImportError as e:
        raise ImportError(
            "azure-ai-projects (>=2.3.0a) and azure-identity are required. "
            "Install with: pip install --pre azure-ai-projects==2.3.0a20260525001 "
            "azure-identity --extra-index-url "
            "https://pkgs.dev.azure.com/azure-sdk/public/_packaging/"
            "azure-sdk-for-python/pypi/simple"
        ) from e
    ENV.sft_job_id = job_name
    print(f"\nSFT_JOB_ID: {job_name}")
    foundry_portal_url = getattr(created, "foundry_portal_url", None)
    if foundry_portal_url:
        foundry_portal_url = foundry_portal_url.replace(
            "/build/train/jobs/", "/build/train/custom-jobs/")
        print(f"Portal: {foundry_portal_url}")
    return job_name


def wait_for_sft_lora(
    job_id: str | None = None,
    poll_interval_sec: int = 60,
    max_wait_min: int = 240,
) -> str:
    """Poll the SFT job until it reaches a terminal state and return the
    output adapter URI suitable for ``submit_job(sft_lora_uri=...)``.

    Parameters
    ----------
    job_id : str, optional
        SFT job name. Defaults to the one stored by the most recent
        ``submit_sft_lora()`` call (``ENV.sft_job_id``).
    poll_interval_sec : int, default 60
        Seconds between status polls.
    max_wait_min : int, default 240
        Hard ceiling (in minutes) before raising TimeoutError.

    Returns
    -------
    str
        ``azureai://accounts/<account>/projects/<project>/data/<name>/versions/<version>``
        pointing at the SFT job's ``checkpoints`` output asset.
    """
    import time
    import requests
    from azure.identity import DefaultAzureCredential

    job_id = job_id or getattr(ENV, "sft_job_id", None)
    if not job_id:
        raise ValueError("No SFT job_id. Pass job_id= or run submit_sft_lora() first.")
    project_endpoint = ENV.project_endpoint.rstrip("/")
    # account/project derived from endpoint URL
    # https://<account>.services.ai.azure.com/api/projects/<project>
    account = project_endpoint.split("//", 1)[1].split(".")[0]
    project = project_endpoint.rsplit("/", 1)[-1]

    with DefaultAzureCredential() as cred:
        token = cred.get_token("https://ai.azure.com/.default").token

    url = (f"{project_endpoint}/jobs/{job_id}"
           "?api-version=2026-01-15-preview")
    headers = {"Authorization": f"Bearer {token}"}

    terminal = {"Completed", "Failed", "Canceled", "Cancelled"}
    deadline = time.time() + max_wait_min * 60
    last_status = None
    print(f"Polling SFT job {job_id} every {poll_interval_sec}s "
          f"(timeout {max_wait_min}m)...")
    while True:
        # Refresh token periodically (cheap; AAD token lasts ~1h).
        if time.time() % 1800 < poll_interval_sec:
            with DefaultAzureCredential() as cred:
                token = cred.get_token("https://ai.azure.com/.default").token
                headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        p = d.get("properties", {})
        status = p.get("status")
        if status != last_status:
            print(f"  status: {status}")
            last_status = status
        if status in terminal:
            if status != "Completed":
                raise RuntimeError(
                    f"SFT job {job_id} ended with status={status}. "
                    "Check the Foundry portal for the failure cause."
                )
            ckpt = (p.get("outputs") or {}).get("checkpoints") or {}
            asset_name = ckpt.get("assetName")
            asset_version = ckpt.get("assetVersion")
            if not asset_name or not asset_version:
                raise RuntimeError(
                    f"SFT job {job_id} completed but checkpoints output has "
                    f"no assetName/assetVersion: {ckpt!r}"
                )
            uri = (f"azureai://accounts/{account}/projects/{project}"
                   f"/data/{asset_name}/versions/{asset_version}")
            ENV.sft_lora_uri = uri
            print(f"\nSFT LoRA adapter URI:\n  {uri}")
            return uri
        if time.time() > deadline:
            raise TimeoutError(
                f"SFT job {job_id} did not finish within {max_wait_min} minutes "
                f"(last status={status})."
            )
        time.sleep(poll_interval_sec)