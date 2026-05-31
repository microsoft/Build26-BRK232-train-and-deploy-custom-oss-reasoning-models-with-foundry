"""Helper module for the Zava SFT demo notebook.

Hides path setup, submission, and rollout-tail plumbing so the notebook cells
stay one-liners.

Usage in notebook::

    from slime_sft_setup import (
        ENV, setup_env, show_submit_params, submit_job, tail_rollouts,
    )
    setup_env(project="<your-foundry-project-name>")
    submit_job(cluster="h100")     # or cluster="a100"

KERNEL / DEPENDENCY REQUIREMENTS
--------------------------------
``submit_job()`` uses ``azure-ai-projects==2.3.0a20260525001`` (prerelease)
+ ``azure-identity``. Both require Python 3.11+.

Install::

    pip install --pre azure-ai-projects==2.3.0a20260525001 azure-identity \
        --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple

Auth: ``az login`` to the tenant that owns your Foundry project.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Populated by setup_env(); imported elsewhere
ENV: SimpleNamespace = SimpleNamespace()


# ── 1. Environment setup ────────────────────────────────────────────────────
def setup_env(
    project: str,
    subscription: str | None = None,
    resource_group: str | None = None,
    workspace: str | None = None,
    region: str | None = None,
    verbose: bool = True,
) -> SimpleNamespace:
    """Discover paths relative to this notebook directory and validate them.

    Parameters
    ----------
    project : str
        Foundry project name to submit against. Required &mdash; pass the name
        of the project you have access to.
    subscription, resource_group, workspace, region : str, optional
        Azure workspace coordinates used by ``tail_rollouts``. They can be
        supplied here (recommended) or passed directly to ``tail_rollouts``
        later. ``workspace`` follows the Foundry format
        ``<workspace>@<project>@AML``; ``region`` is the Azure region of
        the workspace (e.g. ``westcentralus``, ``eastus2``).

    Assumes layout::

        <notebook-dir>/
            slime_sft_setup.py       (this file)
            recipe/                  (submit_sft.py + README.md)
            reports/                 (extract_rollouts.py + per-job output dirs)
    """
    if not project:
        raise ValueError("setup_env requires an explicit project name")
    nb_dir = Path.cwd().resolve()
    e = SimpleNamespace(
        nb_dir=nb_dir,
        recipe_dir=nb_dir / "recipe",
        reports_dir=nb_dir / "reports",
        project=project,
        subscription=subscription,
        resource_group=resource_group,
        workspace=workspace,
        region=region,
        job_id=None,
    )
    for label, p in [("recipe", e.recipe_dir), ("reports", e.reports_dir)]:
        if not p.is_dir():
            raise FileNotFoundError(f"[slime_sft_setup] missing {label}: {p}")
    e.reports_dir.mkdir(parents=True, exist_ok=True)

    global ENV
    ENV = e
    if verbose:
        print(f"NB_DIR    : {e.nb_dir}")
        print(f"Recipe    : {e.recipe_dir}")
        print(f"Reports   : {e.reports_dir}")
        print(f"Project   : {e.project}")
        if e.subscription:
            print(f"Subscript.: {e.subscription}")
        if e.resource_group:
            print(f"RG        : {e.resource_group}")
        if e.workspace:
            print(f"Workspace : {e.workspace}")
        if e.region:
            print(f"Region    : {e.region}")
    return e


# ── 2. Echo key submit parameters ───────────────────────────────────────────
def _recipe_module():
    import importlib
    sys.path.insert(0, str(ENV.recipe_dir))
    import submit_sft as r  # type: ignore[import-not-found]
    importlib.reload(r)
    return r


def show_submit_params(cluster: str = "h100") -> None:
    """Echo the chosen cluster config, dataset URIs, image, and a few key
    command-line knobs from ``submit_sft.py`` so we can verify the recipe
    before submission.
    """
    r = _recipe_module()
    if cluster not in r.CLUSTERS:
        raise ValueError(f"unknown cluster {cluster!r}; choose from {list(r.CLUSTERS)}")
    c = r.CLUSTERS[cluster]
    print(f"Cluster choice    : {cluster}")
    print(f"  name            : {c['name']}")
    print(f"  instance_type   : {c['instance_type']}")
    print(f"  name_prefix     : {c['name_prefix']}")
    print(f"  display name    : {c['display']}")
    print(f"Image             : {r.ENV_IMAGE}")
    print("Datasets:")
    for k, v in r.DATASETS.items():
        print(f"  {k:18s}: {v}")
    # Surface a handful of training knobs from the command string.
    for needle in ("--num_gpus", "--num_nodes", "--tensor_parallel",
                   "--rollout_num_gpus", "--global_batch_size", "--lr",
                   "--num_rollout", "--kl_loss_coef"):
        idx = r.COMMAND.find(needle)
        if idx >= 0:
            # Show "needle value" up to next ' --' or end of string.
            tail = r.COMMAND[idx:].split(" --", 1)[0]
            print(f"  cmd: {tail}")


# ── 3. Submit the job via azure-ai-projects SDK ─────────────────────────────
def _body_to_command_job(body: dict):
    """Translate the recipe's internal config dict into an
    ``azure.ai.projects.CommandJob`` model.

    The recipe expresses the job in Foundry's REST shape (camelCase
    ``properties`` block); ``CommandJob`` takes flat snake_case kwargs.
    """
    from azure.ai.projects.models import CommandJob, JobResourceConfiguration

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
    }
    if props.get("distribution"):
        cj_kwargs["distribution"] = props["distribution"]
    if props.get("properties"):
        cj_kwargs["properties"] = props["properties"]
    cj_kwargs = {k: v for k, v in cj_kwargs.items() if v is not None}
    return CommandJob(**cj_kwargs)


def submit_job(
    cluster: str = "h100",
    instance_count: int = 4,
    name: str | None = None,
) -> str | None:
    """Submit the SFT job via
    ``azure-ai-projects.AIProjectClient.beta.training.jobs``.

    Parameters
    ----------
    cluster : str
        Cluster key defined in ``submit_sft.CLUSTERS`` (``"h100"`` or
        ``"a100"`` out of the box). Add more entries to that dict to expose
        additional clusters here.
    instance_count : int
        Number of nodes to request (default 4).
    name : str, optional
        Explicit job name. If omitted, a random 4-char suffix is appended to
        the cluster's ``name_prefix``.
    """
    import secrets

    r = _recipe_module()
    if cluster not in r.CLUSTERS:
        raise ValueError(f"unknown cluster {cluster!r}; choose from {list(r.CLUSTERS)}")
    c = r.CLUSTERS[cluster]
    job_name = name or f"{c['name_prefix']}-{secrets.token_hex(2)}"

    print("Preparing job spec...")
    payload = r.build_payload(cluster, instance_count)
    cmd_job = _body_to_command_job(payload)

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

    print(f"Submitting {job_name} on {c['name']} "
          f"({instance_count}x {c['instance_type']})")
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=r.PROJECT_ENDPOINT,
                        credential=credential) as project_client,
    ):
        created = project_client.beta.training.jobs.create_or_update(
            name=job_name, job=cmd_job,
        )
    ENV.job_id = job_name
    print(f"\nJOB_ID: {job_name}")
    foundry_portal_url = getattr(created, "foundry_portal_url", None)
    if foundry_portal_url:
        print(f"Portal: {foundry_portal_url}")
    return job_name


# ── 4. Tail rollouts via extract_rollouts.py ────────────────────────────────
def tail_rollouts(
    job_id: str | None = None,
    subscription: str | None = None,
    resource_group: str | None = None,
    workspace: str | None = None,
    region: str | None = None,
) -> None:
    """Pull rollouts + write grades.csv / train_curve.csv / eval_curve.csv.

    Workspace coordinates default to whatever was supplied to ``setup_env``;
    pass any of ``subscription``, ``resource_group``, ``workspace``,
    ``region`` here to override per call.
    """
    job_id = job_id or ENV.job_id
    if not job_id:
        print("No JOB_ID set. Pass job_id= or run submit_job() first.")
        return
    subscription = subscription or getattr(ENV, "subscription", None)
    resource_group = resource_group or getattr(ENV, "resource_group", None)
    workspace = workspace or getattr(ENV, "workspace", None)
    region = region or getattr(ENV, "region", None)
    missing = [n for n, v in [
        ("subscription", subscription),
        ("resource_group", resource_group),
        ("workspace", workspace),
        ("region", region),
    ] if not v]
    if missing:
        raise ValueError(
            "tail_rollouts is missing required workspace coordinates: "
            + ", ".join(missing)
            + ". Supply them via setup_env(...) or as keyword arguments."
        )
    extractor = ENV.reports_dir / "extract_rollouts.py"
    out_dir = ENV.reports_dir / f"out_{job_id.replace('-', '_')}"
    cmd = [sys.executable, str(extractor),
           "--job", job_id,
           "--subscription", subscription,
           "--resource-group", resource_group,
           "--workspace", workspace,
           "--region", region,
           "--out", str(out_dir)]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-2000:])
