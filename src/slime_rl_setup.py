"""Helper module for the Retail RFT demo notebook.

Hides path setup, submission, and log-streaming plumbing so the notebook cells
stay one-liners.

Usage in notebook:
    from slime_rl_setup import ENV, setup_env, submit_job, job_status
    setup_env(project_endpoint="https://<account>.services.ai.azure.com/api/projects/<project>")
    submit_job(cluster_name="<your-compute-cluster>")
    job_status()   # watches job status via Foundry GET /jobs/{name}; opens portal for log content

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

    # Auto-pick the storage connection name based on the project, since Foundry
    # enforces account-scoped unique connection names: two projects under the
    # same account cannot both own a connection called "fdpcommandjobdefaultbyos-
    # connection", so Build-26-Demo-Project had to be created with the
    # "-bdp-connection" suffix. Caller can still override explicitly.
    if storage_connection_name is None:
        _KNOWN_STORAGE_CONNECTIONS = {
            "Build-26-Demo-Proj":    "fdpcommandjobdefaultbyos-connection",      # original sibling
            "Build-26-Demo-Project": "fdpcommandjobdefaultbyos-bdp-connection",  # second project on same account
        }
        if project_name in _KNOWN_STORAGE_CONNECTIONS:
            storage_connection_name = _KNOWN_STORAGE_CONNECTIONS[project_name]

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
    return "Setup is Complete."


# ──────────────────── 2. Echo key submit parameters ───────────────────────────────────────────
KEY_NEEDLES = (
    "JOB_NAME_PREFIX", "NUM_ROLLOUTS", "MODEL_DATASET_ID", "SFT_LORA_DATASET_ID",
    "HF_MODEL_ID", "COMPUTE_CLUSTER", "N_NODES",
    "ROLLOUT_BATCH_SIZE", "N_SAMPLES_PER_PROMPT", "GLOBAL_BATCH_SIZE",
    "LEARNING_RATE", "KL_LOSS_COEF", "EPS_CLIP_HIGH", "MAX_TOKENS_PER_GPU",
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
        "eps_clip_high":     "EPS_CLIP_HIGH",
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
    instance_type: str | None = None,
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
        If True, seed the RFT run from an existing SFT LoRa adapter. You must
        provide its asset URI via ``sft_lora_uri``. If False, perform a
        cold-start RFT run directly on the base model with no SFT seed (and
        ``sft_lora_uri`` is ignored).
    sft_lora_uri : str, optional
        The Foundry dataset URI of the SFT LoRa checkpoint to warm-start
        from (e.g. ``azureai://accounts/<acct>/projects/<proj>/data/<name>/versions/<ver>``).
        Required when ``warm_start=True``. Produce one by running the
        companion SFT notebook (``../post-training-sft-recipe/retail_sft_submit.ipynb``)
        and copying the resulting checkpoint asset URI.
    instance_type : str, optional
        Singularity/AISC instance type SKU (e.g.
        ``"Singularity.ND96r_H100_v5"`` or
        ``"Singularity.ND96am_A100_v4-n1"``). When omitted, the recipe
        default (``INSTANCE_TYPE`` in ``post-training-recipe/submit_job.py``)
        is used. Override this when the cluster's default SKU has zero
        Singularity quota in your account/SLA tier or when you want to
        target a specific GPU generation.
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

    if instance_type:
        if "<" in str(instance_type) or not str(instance_type).strip():
            raise ValueError(
                f"instance_type looks like a placeholder: {instance_type!r}. "
                "Pass a real Singularity SKU (e.g. 'Singularity.ND96r_H100_v5')."
            )
        recipe_submit.INSTANCE_TYPE = instance_type
        print(f"Instance type override: {instance_type}")

    if warm_start:
        if not sft_lora_uri or not str(sft_lora_uri).strip():
            raise ValueError(
                "warm_start=True requires sft_lora_uri — pass the Foundry "
                "asset URI of your SFT LoRa checkpoint "
                "(e.g. 'azureai://accounts/<acct>/projects/<proj>/data/"
                "<name>/versions/<ver>'). Produce one by running the "
                "companion SFT notebook (../post-training-sft-recipe/"
                "retail_sft_submit.ipynb), or pass warm_start=False "
                "to cold-start from the base model."
            )
        recipe_submit.SFT_LORA_DATASET_ID = sft_lora_uri
        recipe_submit.JOB_NAME_PREFIX = "Retail-SFT-RFT-Qwen14B"
        print(f"Warm start: seeding from SFT LoRa {sft_lora_uri}")
    else:
        if sft_lora_uri:
            print("Note: sft_lora_uri is ignored when warm_start=False.")
        recipe_submit.SFT_LORA_DATASET_ID = None
        recipe_submit.JOB_NAME_PREFIX = "Retail-RFT-Qwen14B"
        print("Cold start: no SFT LoRa seed will be used.")

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


# ──────────────────── 4. Watch job status via azure-ai-projects SDK ───────────────────────────
def job_status(
    job_id: str | None = None,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> str:
    """Watch a Foundry training job until it reaches a terminal state.

    Calls only ``AIProjectClient.beta.jobs.get(name=...)`` — the Foundry
    job-resource endpoint (``GET /jobs/{name}``). That endpoint does **not**
    require ``Microsoft.MachineLearningServices/workspaces/experiments/runs/read``,
    so it works for identities that are not AzureML workspace contributors.

    The SDK's ``jobs.stream()`` and ``jobs.download()`` cannot be used here:
    both go through ``/jobs/{name}/history/runs/...``, which is the AzureML
    experiments scope and requires the role above. Log file *content* is only
    reachable via the Foundry portal in that case.

    Parameters
    ----------
    job_id : str, optional
        The job name to watch. Defaults to ``ENV.job_id`` from the most
        recent ``submit_job()`` call.
    poll_seconds : float, default 30
        Interval between ``jobs.get`` calls.
    timeout_seconds : float, optional
        Stop watching after this many seconds. ``None`` waits indefinitely.

    Returns
    -------
    str
        The final job status (lower-cased), e.g. ``"completed"``.

    Raises
    ------
    RuntimeError
        If the job reaches the ``failed`` terminal state.
    """
    import time
    from datetime import datetime, timezone

    job_id = job_id or getattr(ENV, "job_id", None)
    if not job_id:
        print("No JOB_ID set. Pass job_id= or run submit_job() first.")
        return ""

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

    # Mirror the SDK's terminal-state set; kept local so we don't depend on
    # azure.ai.projects' internal _job_helper constants.
    terminal = {"completed", "failed", "canceled", "cancelled",
                "notresponding", "paused", "unknown"}

    project_endpoint = (
        getattr(ENV, "project_endpoint", None) or recipe_submit.PROJECT_ENDPOINT
    )

    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    print(f"Watching job: {job_id}")
    print(f"Endpoint:     {project_endpoint}")
    print(
        "Note: log file content is served from the AzureML experiments scope "
        "(/jobs/<name>/history/runs/...) which requires "
        "Microsoft.MachineLearningServices/workspaces/experiments/runs/read. "
        "If that role is not granted to your identity, this watcher will only "
        "print status transitions and the portal URL — open the portal to view "
        "live driver/user logs in the browser."
    )

    start = time.time()
    last_status: str | None = None
    portal_printed = False

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=project_endpoint, credential=credential) as client,
    ):
        while True:
            try:
                job = client.beta.jobs.get(name=job_id)
            except Exception as exc:  # noqa: BLE001 — surface and retry transient errors
                print(f"[{_ts()}] jobs.get failed: {exc!r} — retrying in {poll_seconds:.0f}s")
                time.sleep(poll_seconds)
                continue

            status = (getattr(job, "status", None) or "").lower() or "unknown"

            if not portal_printed:
                portal = getattr(job, "foundry_portal_url", None)
                if portal:
                    portal = portal.replace("/build/train/jobs/", "/build/train/custom-jobs/")
                    print(f"Portal (live logs): {portal}")
                portal_printed = True

            if status != last_status:
                print(f"[{_ts()}] status: {last_status or '<initial>'} → {status}")
                last_status = status

            if status in terminal:
                print(f"[{_ts()}] terminal state reached: {status}")
                if status == "failed":
                    raise RuntimeError(
                        f"Job '{job_id}' ended in 'failed' state. "
                        f"Open the portal URL above to inspect driver/user logs."
                    )
                return status

            if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
                print(
                    f"[{_ts()}] watch timed out after {timeout_seconds:.0f}s "
                    f"(current status: {status}). Job continues running on the cluster."
                )
                return status

            time.sleep(poll_seconds)


# ──────────────────── 5. SFT LoRa pre-training step ──────────────────────────

# No project-specific defaults — every URI / image / cluster value is supplied
# by the notebook, so the helper works against any Foundry project that has
# the SFT code + data uploaded as datasets and a compatible training image.


def _sft_command(
    *,
    train_filename: str,
    base_model: str,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    lr: str,
    max_seq_len: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    val_frac: float,
    logging_steps: int,
    save_strategy: str,
    extra_pip: str,
) -> str:
    """Build the bash command line for sft_retail.py."""
    # We wrap the whole pipeline in `bash -c '...'`. Pip specifiers that contain
    # shell metacharacters (`<`, `>`) MUST be single-quoted so bash treats them
    # as one literal argument. Since the outer quoting is already single, we
    # use the canonical `'\''` escape (close-quote, escaped-quote, reopen-quote)
    # — written as raw Python strings below for readability.
    q = r"'\''"  # one literal single quote inside a `'...'` block
    pip_pkgs = (
        f"{q}transformers>=4.51,<5{q} trl==0.11.4 peft==0.13.2 "
        f"{q}datasets>=2.21,<3{q} {q}accelerate>=1.0{q} {q}bitsandbytes>=0.43{q} azureml-core"
    )
    if extra_pip:
        pip_pkgs = pip_pkgs + " " + extra_pip
    return (
        "bash -c '"
        # The PTCA base image's site-packages is owned by root, but the training
        # container runs as a non-root user. Install user-locally (~/.local/...)
        # so the freshly-pinned versions take precedence over the image defaults.
        # Use `python -m pip` for the second install so we invoke the just-
        # upgraded pip from ~/.local/bin without depending on PATH ordering.
        "pip install --quiet --user --upgrade pip "
        f"&& python -m pip install --quiet --user {pip_pkgs} "
        '&& python "${{inputs.code_dataset}}/sft_retail.py" '
        f'--train-jsonl "${{{{inputs.train_dataset}}}}/{train_filename}" '
        f"--base-model {base_model} "
        '--output-dir "${{outputs.checkpoints}}" '
        f"--epochs {epochs} --batch-size {batch_size} "
        f"--grad-accum {grad_accum} --lr {lr} --max-seq-len {max_seq_len} "
        f"--lora-r {lora_r} --lora-alpha {lora_alpha} "
        f"--lora-dropout {lora_dropout} --target-modules {target_modules} "
        f"--val-frac {val_frac} --logging-steps {logging_steps} "
        f"--save-strategy {save_strategy}"
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
    *,
    cluster_name: str,
    instance_type: str,
    image: str,
    code_uri: str,
    train_uri: str,
    train_filename: str,
    base_model: str,
    epochs: float = 1.0,
    instance_count: int = 1,
    name: str | None = None,
    display_name: str = "Retail-SFT-14B",
    # Generic LoRA SFT training knobs (no project-specific defaults).
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: str = "2e-05",
    max_seq_len: int = 4096,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: str = "qwen",
    val_frac: float = 0.0,
    logging_steps: int = 2,
    save_strategy: str = "epoch",
    extra_pip: str = "",
) -> str:
    """Submit a single-node LoRA SFT job (``sft_retail.py``) and return the
    submitted job name.

    All Foundry-project / dataset / image references are required parameters
    — the helper carries no project-specific defaults so the same code works
    for any customer scenario.

    Parameters
    ----------
    cluster_name : str
        Short name of the Foundry compute cluster (e.g. ``"my-h100-cluster"``).
    instance_type : str
        Singularity/AISC instance type (e.g. ``"Singularity.ND96r_H100_v5"``).
    image : str
        Container image reference for the training environment.
    code_uri : str
        ``azureai://...`` URI of the dataset folder containing ``sft_retail.py``.
    train_uri : str
        ``azureai://...`` URI of the dataset folder containing the training
        JSONL file.
    train_filename : str
        Filename of the training JSONL within ``train_uri``.
    base_model : str
        Hugging Face model id of the base checkpoint.
    epochs : float, default 1.0
        Number of training epochs for the LoRA adapter.
    instance_count : int, default 1
        Number of compute nodes (sft_retail.py is single-node — leave at 1).
    name : str, optional
        Explicit job name. If omitted, ``"sft-lora-<hex>"`` is used.
    batch_size, grad_accum, lr, max_seq_len, lora_r, lora_alpha, lora_dropout,
    target_modules, val_frac, logging_steps, save_strategy : optional
        Generic LoRA SFT knobs passed straight to sft_retail.py; defaults are
        safe baselines for Qwen-class base models.
    extra_pip : str, optional
        Extra ``pip install`` packages (space-separated) for the bootstrap.

    Returns
    -------
    str
        The submitted job name.
    """
    for label, val in [
        ("cluster_name", cluster_name), ("instance_type", instance_type),
        ("image", image), ("code_uri", code_uri), ("train_uri", train_uri),
        ("train_filename", train_filename), ("base_model", base_model),
    ]:
        if not val or "<" in str(val):
            raise ValueError(f"submit_sft_lora requires an explicit {label}")
    if not getattr(ENV, "project_endpoint", None):
        raise RuntimeError("Call setup_env() before submit_sft_lora().")

    import secrets
    from datetime import datetime, timezone

    job_name = name or f"sft-lora-{secrets.token_hex(2)}"
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    asset_base = f"{job_name}-out"

    cj_props = {
        "description": f"LoRA SFT on {base_model} ({epochs} epoch{'s' if epochs != 1 else ''})",
        "tags": {
            "scenario": "sft-lora",
            "agent": base_model,
            "epochs": str(epochs),
        },
        "displayName": display_name,
        "computeId": _compute_id_from_cluster(cluster_name),
        "experimentName": "Default",
        "jobType": "Command",
        "resources": {
            "instanceCount": instance_count,
            "instanceType": instance_type,
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
        "command": _sft_command(
            train_filename=train_filename, base_model=base_model, epochs=epochs,
            batch_size=batch_size, grad_accum=grad_accum, lr=lr,
            max_seq_len=max_seq_len, lora_r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=target_modules,
            val_frac=val_frac, logging_steps=logging_steps,
            save_strategy=save_strategy, extra_pip=extra_pip,
        ),
        "environmentImageReference": image,
        "inputs": {
            "train_dataset": {"uri": train_uri,
                              "mode": "ReadOnlyMount", "jobInputType": "uri_folder"},
            "code_dataset":  {"uri": code_uri,
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
        "environmentVariables": {
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
          f"({instance_count}x {instance_type}, epochs={epochs})")
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


def submit_sft_lora_from_local(
    *,
    train_path: str | Path,
    val_path: str | Path | None = None,
    dataset_name: str,
    code_dir: str | Path,
    code_dataset_name: str,
    cluster_name: str,
    instance_type: str,
    image: str,
    base_model: str,
    epochs: float = 1.0,
    **submit_kwargs,
) -> str:
    """Upload local SFT data + code folders to Foundry datasets, then submit
    an SFT-LoRA job.

    Convenience wrapper around :func:`submit_sft_lora` that takes notebook-
    friendly local paths (relative to the notebook directory, or absolute)
    for both the train/val JSONL files and the training-code folder
    (containing ``sft_retail.py``). Both are uploaded as fresh
    timestamped versions of their respective Foundry datasets via
    ``helpers.upload_dataset``; the returned URIs are forwarded as
    ``train_uri`` / ``code_uri`` to :func:`submit_sft_lora`.

    Parameters
    ----------
    train_path : str | Path
        Local path to the training JSONL file (file, not folder). Relative
        paths resolve against the notebook directory (``ENV.nb_dir``).
    val_path : str | Path, optional
        Local path to a validation JSONL file. Uploaded alongside the
        training file so it is available to the training container; only
        actually consumed if ``sft_retail.py`` references it.
    dataset_name : str
        Foundry dataset asset name for the data folder. A new timestamp-
        based version is created on every call.
    code_dir : str | Path
        Local folder containing ``sft_retail.py`` (and any helper modules it
        imports). Uploaded as ``code_dataset_name``.
    code_dataset_name : str
        Foundry dataset asset name for the code folder.
    cluster_name, instance_type, image, base_model, epochs :
        Forwarded to :func:`submit_sft_lora`.
    **submit_kwargs :
        Any other keyword arguments accepted by :func:`submit_sft_lora`
        (e.g. ``lora_r``, ``lr``, ``val_frac``).

    Returns
    -------
    str
        The submitted job name (same as :func:`submit_sft_lora`).
    """
    import shutil
    import tempfile
    from datetime import datetime, timezone

    if not getattr(ENV, "project_endpoint", None):
        raise RuntimeError("Call setup_env() before submit_sft_lora_from_local().")

    nb_dir = Path(getattr(ENV, "nb_dir", Path.cwd())).resolve()

    def _resolve_file(p: str | Path, label: str) -> Path:
        rp = Path(p)
        if not rp.is_absolute():
            rp = (nb_dir / rp).resolve()
        if not rp.exists() or not rp.is_file():
            raise FileNotFoundError(f"{label} does not exist or is not a file: {rp}")
        if rp.suffix.lower() != ".jsonl":
            print(f"warning: {label} is not a .jsonl file: {rp.name}")
        return rp

    def _resolve_dir(p: str | Path, label: str) -> Path:
        rp = Path(p)
        if not rp.is_absolute():
            rp = (nb_dir / rp).resolve()
        if not rp.exists() or not rp.is_dir():
            raise FileNotFoundError(f"{label} does not exist or is not a directory: {rp}")
        return rp

    train_file = _resolve_file(train_path, "train_path")
    val_file = _resolve_file(val_path, "val_path") if val_path else None
    code_folder = _resolve_dir(code_dir, "code_dir")
    if not (code_folder / "sft_retail.py").exists():
        print(f"warning: sft_retail.py not found at top of {code_folder}")

    if str(ENV.recipe_dir) not in sys.path:
        sys.path.insert(0, str(ENV.recipe_dir))
    import helpers as fh  # type: ignore[import-not-found]
    upload_extras = {}
    if getattr(ENV, "storage_connection_name", None):
        upload_extras["connection_name"] = ENV.storage_connection_name

    # Stage data files into a fresh folder so the upload only contains the
    # requested files (avoids pulling in unrelated siblings).
    with tempfile.TemporaryDirectory(prefix="sft-upload-") as staging:
        staging_dir = Path(staging)
        shutil.copy2(train_file, staging_dir / train_file.name)
        if val_file is not None:
            shutil.copy2(val_file, staging_dir / val_file.name)

        version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        print(f"Uploading {len(list(staging_dir.iterdir()))} data file(s) to "
              f"Foundry dataset {dataset_name}:{version} ...")
        train_uri = fh.upload_dataset(
            staging_dir,
            dataset_name=dataset_name,
            dataset_version=version,
            project_endpoint=ENV.project_endpoint,
            **upload_extras,
        )
    print(f"train_uri = {train_uri}")

    code_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    print(f"Uploading code folder {code_folder} to Foundry dataset "
          f"{code_dataset_name}:{code_version} ...")
    code_uri = fh.upload_dataset(
        code_folder,
        dataset_name=code_dataset_name,
        dataset_version=code_version,
        project_endpoint=ENV.project_endpoint,
        **upload_extras,
    )
    print(f"code_uri  = {code_uri}")

    return submit_sft_lora(
        cluster_name=cluster_name,
        instance_type=instance_type,
        image=image,
        code_uri=code_uri,
        train_uri=train_uri,
        train_filename=train_file.name,
        base_model=base_model,
        epochs=epochs,
        **submit_kwargs,
    )


def wait_for_sft_lora(
    job_id: str | None = None,
    poll_interval_sec: int = 60,  # retained for back-compat; unused (no polling)
    max_wait_min: int = 240,      # retained for back-compat; unused (no polling)
) -> str:
    """Resolve the SFT job's LoRA adapter URI *immediately* without polling.

    The job's output ``checkpoints`` asset name/version is fixed at submit
    time (see ``submit_sft_lora_from_local``), so we can construct the
    ``azureai://`` URI as soon as the job is registered -- no need to wait
    for the SFT run to finish. The returned URI can be passed straight
    into ``submit_job(sft_lora_uri=...)`` so the next (RFT) job is queued
    immediately. Foundry will mount the adapter at RFT-job start time;
    if the SFT job hasn't finished writing it by then, the RFT job will
    wait at mount time or fail with a clear "asset not found" error.

    Parameters
    ----------
    job_id : str, optional
        SFT job name. Defaults to the one stored by the most recent
        ``submit_sft_lora_from_local()`` call (``ENV.sft_job_id``).
    poll_interval_sec, max_wait_min
        Accepted for backward compatibility; ignored (this function no
        longer polls). Will be removed in a future cleanup.

    Returns
    -------
    str
        ``azureai://accounts/<account>/projects/<project>/data/<name>/versions/<version>``
        pointing at the SFT job's ``checkpoints`` output asset.
    """
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
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    p = r.json().get("properties", {})
    status = p.get("status")
    ckpt = (p.get("outputs") or {}).get("checkpoints") or {}
    asset_name = ckpt.get("assetName")
    asset_version = ckpt.get("assetVersion")
    if not asset_name or not asset_version:
        raise RuntimeError(
            f"SFT job {job_id} has no checkpoints assetName/assetVersion in its "
            f"spec (status={status}): {ckpt!r}. Cannot construct adapter URI."
        )
    uri = (f"azureai://accounts/{account}/projects/{project}"
           f"/data/{asset_name}/versions/{asset_version}")
    ENV.sft_lora_uri = uri
    print(f"SFT job {job_id} status: {status} (not waiting -- proceeding to next job)")
    print(f"SFT LoRa adapter URI (resolved from job spec):\n  {uri}")
    return uri