"""Extract rollouts + grades from a Foundry RFT job.

Pulls every per-trajectory JSONL artifact (train + eval) and produces:
  - `<out>/rollouts/{train,eval}/<file>.jsonl` — raw JSONL artifacts.
  - `<out>/grades.csv`                          — flat table, one row per sample.
  - `<out>/eval_curve.csv`                      — one row per eval rollout (mean / nz / max).
  - `<out>/train_curve.csv`                     — one row per train rollout.
  - `<out>/sample_responses.jsonl`              — full prompt + response + reward dict per sample.

Works for any RFT job that uses the proven `rollout_logger.py` artifact
schema.

Usage:
    python extract_rollouts.py --job <job_name> \
        --subscription <sub-id> --resource-group <rg> \
        --workspace <workspace@project@AML> --region <region> \
        [--out <dir>] [--no-download] [--max-rollouts N]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from azure.identity import AzureCliCredential

logger = logging.getLogger("extract")


def _auth_header() -> dict:
    cred = AzureCliCredential(process_timeout=30)
    tok = cred.get_token("https://management.azure.com/.default").token
    return {"Authorization": f"Bearer {tok}"}


def _ws_base(cfg: dict) -> str:
    ws = urllib.parse.quote(cfg["workspace"], safe="")
    return (
        f"{cfg['rh_host']}/artifact/v2.0/subscriptions/{cfg['subscription']}"
        f"/resourceGroups/{cfg['resource_group']}"
        f"/providers/Microsoft.MachineLearningServices/workspaces/{ws}"
    )


def list_artifacts(cfg: dict, job: str) -> list[str]:
    """Paginated list of all artifact paths for a run."""
    base = _ws_base(cfg)
    headers = _auth_header()
    paths: list[str] = []
    ct = ""
    while True:
        url = (
            f"{base}/artifacts/ExperimentRun/dcid.{job}"
            f"?continuationToken={ct}&pageSize=500"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        for v in data.get("value", []):
            p = v.get("path")
            if p:
                paths.append(p)
        ct = data.get("continuationToken") or ""
        if not ct:
            break
    return paths


def get_sas(cfg: dict, job: str, path: str) -> str:
    """Get a SAS download URL for a single artifact."""
    base = _ws_base(cfg)
    headers = _auth_header()
    enc = urllib.parse.quote(path, safe="")
    url = f"{base}/artifacts/ExperimentRun/dcid.{job}/contentinfo?path={enc}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read()).get("contentUri", "")


def download_artifact(cfg: dict, job: str, path: str, out_path: Path) -> bool:
    """Download one artifact via SAS to disk. Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sas = get_sas(cfg, job, path)
        if not sas:
            return False
        with urllib.request.urlopen(sas, timeout=120) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"download failed for {path}: {exc}")
        return False


# ── Per-sample extraction (handles both train + eval JSONL formats) ─
def _coerce_reward(r) -> dict:
    """Train rollouts log a reward dict; eval rollouts log a scalar.
    Normalize both into a dict with 'score' as the primary metric."""
    if isinstance(r, dict):
        out = dict(r)
        out.setdefault("score", out.get("score") or 0.0)
        return out
    try:
        return {"score": float(r)}
    except (TypeError, ValueError):
        return {"score": 0.0}


def _flatten_sample(rec: dict, kind: str) -> dict:
    """Turn one raw rollout JSONL record into a flat row."""
    reward = _coerce_reward(rec.get("reward"))
    meta = rec.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (TypeError, json.JSONDecodeError):
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    resp = rec.get("response") or ""
    if isinstance(resp, list):
        resp = json.dumps(resp, ensure_ascii=False)
    return {
        "kind": kind,
        "rollout_id": rec.get("rollout_id"),
        "sample_idx": rec.get("sample_idx"),
        "group_index": rec.get("group_index"),
        # Identifiers (vary by run type — tau2 vs zava vs others)
        "task_id": meta.get("task_id"),
        "scenario_id": meta.get("scenario_id") or meta.get("task_id"),
        "order_id": meta.get("order_id"),
        "domain": meta.get("domain"),
        "difficulty": meta.get("difficulty"),
        # Scoring
        "score": reward.get("score"),
        "binary_reward": reward.get("binary_reward"),
        "db_reward": reward.get("db_reward"),
        "mean_action_reward": reward.get("mean_action_reward"),
        "n_actions": reward.get("n_actions"),
        "n_action_matches": reward.get("n_action_matches"),
        "mean_env_assertion_reward": reward.get("mean_env_assertion_reward"),
        "n_env_assertions": reward.get("n_env_assertions"),
        "mean_communicate_reward": reward.get("mean_communicate_reward"),
        "mean_nl_assertion_reward": reward.get("mean_nl_assertion_reward"),
        # Zava-specific (only present in zava runs)
        "n_amount_matches": reward.get("n_amount_matches"),
        "n_reason_matches": reward.get("n_reason_matches"),
        "n_parsed_actions": reward.get("n_parsed_actions"),
        "n_expected_actions": reward.get("n_expected_actions"),
        "n_extra_actions": reward.get("n_extra_actions"),
        # Trajectory shape
        "status": rec.get("status"),
        "response_length": rec.get("response_length"),
        "n_assistant_turns": (
            meta.get("n_assistant_turns")
            or reward.get("n_assistant_turns")
        ),
        "n_tool_calls": meta.get("n_tool_calls") or reward.get("n_tool_calls"),
        "tool_call_history": meta.get("tool_call_history"),
        "episode_terminated": meta.get("episode_terminated"),
        "episode_truncated": meta.get("episode_truncated"),
        "submitted_via_tool": meta.get("submitted_via_tool"),
        "rollout_time_sec": rec.get("rollout_time_sec"),
        "removed": rec.get("removed"),
        "response_preview": resp[:300].replace("\n", " ").replace("\r", " "),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True, help="Foundry run id (job name)")
    p.add_argument("--subscription", required=True,
                   help="Azure subscription id that owns the workspace")
    p.add_argument("--resource-group", required=True,
                   help="Resource group of the AML workspace")
    p.add_argument("--workspace", required=True,
                   help="Workspace identifier in the form <workspace>@<project>@AML")
    p.add_argument("--region", required=True,
                   help="Azure region of the workspace (e.g. westcentralus, eastus2)")
    p.add_argument("--out", default=None,
                   help="Output directory (default: ./out_<job>)")
    p.add_argument("--no-download", action="store_true",
                   help="Skip downloading raw JSONL files (just summarize)")
    p.add_argument("--max-rollouts", type=int, default=10**9,
                   help="Cap on number of train rollouts to extract")
    p.add_argument("--include-response", action="store_true",
                   help="Include full response text in sample_responses.jsonl")
    args = p.parse_args()

    cfg = {
        "rh_host": f"https://{args.region}.api.azureml.ms",
        "subscription": args.subscription,
        "resource_group": args.resource_group,
        "workspace": args.workspace,
    }
    out_dir = Path(args.out or f"out_{args.job}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rollouts" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "rollouts" / "eval").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Listing artifacts for {args.job} in {args.region}...")
    paths = list_artifacts(cfg, args.job)
    train_paths = sorted(p for p in paths if "train_rollout_" in p)
    eval_paths = sorted(p for p in paths if "eval_rollout_" in p)
    print(f"  found {len(train_paths)} train + {len(eval_paths)} eval rollouts")

    train_paths = train_paths[: args.max_rollouts]

    # ── Download ─────────────────────────────────────────────────────
    files_to_load: list[tuple[Path, str]] = []  # (local_path, kind)
    for path, kind in [(p, "train") for p in train_paths] + [(p, "eval") for p in eval_paths]:
        local = out_dir / "rollouts" / kind / Path(path).name
        if args.no_download and local.exists():
            files_to_load.append((local, kind))
            continue
        if not local.exists() or local.stat().st_size == 0:
            ok = download_artifact(cfg, args.job, path, local)
            if not ok:
                continue
        files_to_load.append((local, kind))
    print(f"  downloaded/cached {len(files_to_load)} files into {out_dir}/rollouts/")

    # ── Flatten into rows ────────────────────────────────────────────
    all_rows: list[dict] = []
    full_records: list[dict] = []
    for local, kind in files_to_load:
        try:
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = _flatten_sample(rec, kind)
                all_rows.append(row)
                if args.include_response:
                    full_records.append({
                        "kind": kind,
                        "rollout_id": rec.get("rollout_id"),
                        "sample_idx": rec.get("sample_idx"),
                        "scenario_id": row["scenario_id"],
                        "order_id": row["order_id"],
                        "score": row["score"],
                        "binary_reward": row["binary_reward"],
                        "prompt": rec.get("prompt"),
                        "response": rec.get("response"),
                        "reward": rec.get("reward"),
                        "metadata": rec.get("metadata"),
                    })
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to read {local}: {exc}")

    if not all_rows:
        print("No rollouts parsed. Either job has no artifacts yet or download failed.")
        return

    # ── Write flat grades.csv ────────────────────────────────────────
    fieldnames = list(all_rows[0].keys())
    grades_csv = out_dir / "grades.csv"
    with open(grades_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"  wrote {len(all_rows)} rows -> {grades_csv}")

    # ── Per-rollout curves ───────────────────────────────────────────
    def _curve(kind: str) -> list[dict]:
        by_rollout: dict[int, list[dict]] = {}
        for r in all_rows:
            if r["kind"] != kind:
                continue
            rid = r.get("rollout_id")
            if rid is None:
                continue
            by_rollout.setdefault(int(rid), []).append(r)
        out = []
        for rid in sorted(by_rollout):
            rs = by_rollout[rid]
            scores = [s["score"] for s in rs if s["score"] is not None]
            if not scores:
                continue
            n = len(scores)
            nonzero = sum(1 for s in scores if s > 0)
            perfect = sum(1 for s in scores if s >= 0.999)
            out.append({
                "rollout_id": rid,
                "n_samples": n,
                "mean_score": sum(scores) / n,
                "max_score": max(scores),
                "min_score": min(scores),
                "nonzero": nonzero,
                "perfect_1.0": perfect,
                "nonzero_pct": round(100 * nonzero / n, 1),
            })
        return out

    for kind in ("train", "eval"):
        rows = _curve(kind)
        if not rows:
            continue
        curve_csv = out_dir / f"{kind}_curve.csv"
        with open(curve_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  wrote {kind} curve ({len(rows)} rollouts) -> {curve_csv}")

    # ── Full prompt+response JSONL (optional) ────────────────────────
    if args.include_response:
        sr = out_dir / "sample_responses.jsonl"
        with open(sr, "w", encoding="utf-8") as f:
            for rec in full_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"  wrote {len(full_records)} responses -> {sr}")

    # ── Headline summary ─────────────────────────────────────────────
    train_curve = _curve("train")
    eval_curve = _curve("eval")
    print()
    print("=" * 60)
    print(f"Summary for {args.job}")
    print("=" * 60)
    if train_curve:
        best = max(train_curve, key=lambda r: r["mean_score"])
        last = train_curve[-1]
        print(f"  Train: {len(train_curve)} rollouts")
        print(f"    Best mean @ rollout {best['rollout_id']}: "
              f"{best['mean_score']:.3f} ({best['perfect_1.0']}/{best['n_samples']} perfect)")
        print(f"    Last mean @ rollout {last['rollout_id']}: {last['mean_score']:.3f}")
    if eval_curve:
        best = max(eval_curve, key=lambda r: r["mean_score"])
        last = eval_curve[-1]
        print(f"  Eval: {len(eval_curve)} points")
        print(f"    Best mean @ rollout {best['rollout_id']}: "
              f"{best['mean_score']:.3f} ({best['perfect_1.0']}/{best['n_samples']} perfect)")
        print(f"    Last mean @ rollout {last['rollout_id']}: {last['mean_score']:.3f}")
    print()
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
