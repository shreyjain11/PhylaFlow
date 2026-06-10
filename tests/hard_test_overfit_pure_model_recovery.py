import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CSV_FIELDS = [
    "experiment_id",
    "status",
    "created_at",
    "base_config",
    "epochs",
    "target_epoch",
    "note",
    "git_commit",
    "git_dirty_hash",
    "checkpoint_root",
    "checkpoint_dir",
    "checkpoint_path",
    "trial_dir",
    "train_log_path",
    "pure_model_rf",
    "training_runtime_s",
    "evaluation_runtime_s",
    "total_runtime_s",
    "overrides_json",
]

OVERFIT_CHECKPOINT_EVERY_EPOCHS = 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sanity overfit training job, evaluate pure-model "
            "sampling recovery RF, and record the result to JSONL and CSV."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/sanity_train.yaml",
        help="Base config file to start from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3_000,
        help="Number of overfit epochs/steps to run.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value using dotted keys, e.g. trainer.lr=0.001.",
    )
    parser.add_argument(
        "--name",
        default="sanity-overfit",
        help="Human-readable prefix for the experiment id.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note recorded with the experiment.",
    )
    parser.add_argument(
        "--results-root",
        default="analysis/overfit_pure_model_search",
        help="Where JSONL/CSV ledgers and per-trial metadata/logs are stored.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="",
        help=(
            "Root directory for trial checkpoints. Defaults to "
            "<base checkpoint dir>/overfit_pure_model_search."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for the training subprocess.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if an identical recorded experiment already exists.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Keep trainer.record enabled during training so W&B/sample logging stays on.",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        default="",
        help=(
            "Skip training and evaluate an already-produced checkpoint path. "
            "Useful for salvaging a partially completed long run."
        ),
    )
    return parser.parse_args()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must look like KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def _set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise TypeError(
                f"Cannot assign nested key {dotted_key!r}; {key!r} is not a mapping."
            )
        cursor = next_value
    cursor[keys[-1]] = value


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        _set_nested_value(updated, key, value)
    return updated


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _git_metadata(repo_root: Path) -> dict[str, str]:
    env_commit = os.environ.get("PHYLAFLOW_GIT_COMMIT", "").strip()
    env_dirty_hash = os.environ.get("PHYLAFLOW_GIT_DIRTY_HASH", "").strip()
    env_status_path = os.environ.get("PHYLAFLOW_GIT_STATUS_PATH", "").strip()
    env_status = os.environ.get("PHYLAFLOW_GIT_STATUS", "")
    if env_commit and env_dirty_hash:
        if env_status_path:
            try:
                env_status = Path(env_status_path).read_text(encoding="utf-8")
            except OSError:
                env_status = env_status or ""
        return {
            "git_commit": env_commit,
            "git_dirty_hash": env_dirty_hash,
            "git_status": env_status,
        }

    commit = _git_output(repo_root, "rev-parse", "HEAD")
    status = _git_output(repo_root, "status", "--short")
    diff = _git_output(repo_root, "diff", "--no-ext-diff", "HEAD", "--", ".")
    return {
        "git_commit": commit or "unknown",
        "git_dirty_hash": _hash_text(status + "\n" + diff),
        "git_status": status,
    }


def _dedupe_payload(
    base_config: str,
    overrides: dict[str, Any],
    git_commit: str,
    git_dirty_hash: str,
    evaluate_checkpoint: str = "",
) -> dict[str, Any]:
    payload = {
        "base_config": base_config,
        "mode": "evaluate_checkpoint" if evaluate_checkpoint else "run_overfit",
        "overrides": overrides,
        "git_commit": git_commit,
        "git_dirty_hash": git_dirty_hash,
    }
    if evaluate_checkpoint:
        payload["evaluate_checkpoint"] = evaluate_checkpoint
    return payload


def _load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _find_existing_record(
    records: list[dict[str, Any]], dedupe_key: str
) -> dict[str, Any] | None:
    for record in reversed(records):
        record_key = record.get("dedupe_key")
        if not record_key:
            record_key = _hash_text(
                _canonical_json(
                    _dedupe_payload(
                        base_config=record.get("base_config", ""),
                        overrides=record.get("overrides", {}),
                        git_commit=record.get("git_commit", "unknown"),
                        git_dirty_hash=record.get("git_dirty_hash", ""),
                    )
                )
            )
        if record_key == dedupe_key:
            return record
    return None


def _append_record(record: dict[str, Any], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    csv_row = {field: record.get(field, "") for field in CSV_FIELDS}
    needs_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(csv_row)


def _run_overfit_training(
    python_executable: str,
    repo_root: Path,
    config_path: Path,
    train_log_path: Path,
) -> int:
    command = [
        python_executable,
        "-c",
        (
            "import sys; "
            "from run.run import run_overfit; "
            "sys.argv = ['run_overfit', sys.argv[1]]; "
            "run_overfit()"
        ),
        str(config_path),
    ]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc_path and not env.get("CUDA_HOME"):
        cuda_home = str(Path(nvcc_path).resolve().parent.parent)
        env["CUDA_HOME"] = cuda_home
        env.setdefault("CUDA_PATH", cuda_home)

    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    with train_log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return result.returncode


def _find_target_checkpoint(checkpoint_root: Path, target_epoch: int) -> tuple[Path, Path]:
    expected_name = f"overfit-epoch={target_epoch}.ckpt"
    matches = sorted(
        checkpoint_root.rglob(expected_name),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if not matches:
        available = sorted(str(path) for path in checkpoint_root.rglob("*.ckpt"))
        raise FileNotFoundError(
            f"Could not find {expected_name} under {checkpoint_root}. "
            f"Available checkpoints: {available[:10]}"
        )
    checkpoint_path = matches[-1]
    return checkpoint_path.parent, checkpoint_path


def _extract_evaluation_checkpoint_context(
    checkpoint_path: Path,
) -> tuple[str, Path]:
    checkpoint_dir = checkpoint_path.parent
    if checkpoint_dir.name.startswith("202"):
        if checkpoint_dir.parent.parent.name == "overfit_pure_model_search":
            return checkpoint_dir.parent.name, checkpoint_dir.parent
        return checkpoint_dir.name, checkpoint_dir
    if checkpoint_dir.name:
        return checkpoint_dir.name, checkpoint_dir
    raise ValueError(
        f"Could not infer evaluation context from checkpoint path {checkpoint_path}."
    )


def _tail_text(path: Path, max_lines: int = 40) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _evaluate_pure_model_rf(
    config: dict[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    from tests import test_checkpoint_sampling_recovery as checkpoint_recovery

    TrainingModule = checkpoint_recovery.TrainingModule
    build_sanity_tree_pair = checkpoint_recovery._build_sanity_tree_pair
    load_checkpoint_model = checkpoint_recovery._load_checkpoint_model
    normalized_rf = checkpoint_recovery._normalized_rf

    random.seed(13)
    torch.manual_seed(13)
    device = torch.device("cpu")

    data_module, start_tree, target_tree, n_leaves = build_sanity_tree_pair(config)
    base_model = load_checkpoint_model(config, str(checkpoint_path), device)

    module = TrainingModule(
        model=copy.deepcopy(base_model),
        dataset=data_module,
        lr=config["trainer"]["lr"],
        record=False,
        epochs=1,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=None,
    ).to(device)
    module.eval()

    phyla_dim = int(config["model"]["phyla_dim"])
    phyla_embeddings = torch.zeros(
        (1, n_leaves, phyla_dim),
        dtype=torch.float32,
        device=device,
    )
    sampled_trees, _, _, _, _ = module.sample(
        [start_tree],
        phyla_embeddings=phyla_embeddings,
        num_samples=1,
        T=1.0,
        dt_base=0.02,
        max_steps=256,
        max_events=1024,
    )
    sampled_tree = sampled_trees[0]
    return {
        "pure_model_rf": normalized_rf(sampled_tree, target_tree),
        "sampled_tree": sampled_tree,
        "target_tree": target_tree,
        "start_tree": start_tree,
    }


def main() -> int:
    args = _parse_args()
    if args.epochs <= 0 or args.epochs % OVERFIT_CHECKPOINT_EVERY_EPOCHS != 0:
        raise ValueError(
            "Overfit runner checkpoints every "
            f"{OVERFIT_CHECKPOINT_EVERY_EPOCHS} epochs, so --epochs must be a "
            f"positive multiple of {OVERFIT_CHECKPOINT_EVERY_EPOCHS}. Got {args.epochs}."
        )
    repo_root = REPO_ROOT
    base_config_path = (repo_root / args.config).resolve()
    results_root = (repo_root / args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_root / "results.jsonl"
    csv_path = results_root / "results.csv"

    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    explicit_overrides = dict(_parse_override(item) for item in args.set)
    harness_overrides = dict(explicit_overrides)
    harness_overrides["trainer.epochs"] = args.epochs
    harness_overrides["trainer.record"] = bool(args.record)
    evaluate_checkpoint = (
        str(Path(args.evaluate_checkpoint).resolve()) if args.evaluate_checkpoint else ""
    )

    git_meta = _git_metadata(repo_root)
    dedupe_payload = _dedupe_payload(
        base_config=str(base_config_path),
        overrides=harness_overrides,
        git_commit=git_meta["git_commit"],
        git_dirty_hash=git_meta["git_dirty_hash"],
        evaluate_checkpoint=evaluate_checkpoint,
    )
    experiment_spec = {
        **dedupe_payload,
        "note": args.note,
    }
    dedupe_key = _hash_text(_canonical_json(dedupe_payload))
    fingerprint = _hash_text(_canonical_json(experiment_spec))
    experiment_id = f"{args.name}-{fingerprint[:12]}"

    existing_record = _find_existing_record(_load_records(jsonl_path), dedupe_key)
    if existing_record is not None and not args.force:
        print(
            _canonical_json(
                {
                    "status": "skipped_existing",
                    "experiment_id": existing_record.get("experiment_id"),
                    "pure_model_rf": existing_record.get("pure_model_rf"),
                    "record_status": existing_record.get("status"),
                }
            )
        )
        return 0 if existing_record.get("status") == "completed" else 1

    trial_config = _apply_overrides(base_config, harness_overrides)
    created_at = datetime.now(timezone.utc).isoformat()
    target_epoch = args.epochs - 1
    training_start = time.time()
    status = "training_failed"
    checkpoint_dir = ""
    checkpoint_path = ""
    pure_model_rf = None
    evaluation_runtime_s = None
    sampled_tree = ""
    target_tree = ""
    start_tree = ""
    failure_reason = ""
    train_returncode = None

    checkpoint_root = (
        Path(args.checkpoint_root).resolve()
        if args.checkpoint_root
        else Path(base_config["trainer"]["checkpoint_dir"]).resolve()
        / "overfit_pure_model_search"
    )

    if args.evaluate_checkpoint:
        resolved_checkpoint_path = Path(args.evaluate_checkpoint).resolve()
        (
            resolved_experiment_id,
            trial_checkpoint_root,
        ) = _extract_evaluation_checkpoint_context(
            resolved_checkpoint_path
        )
        if resolved_experiment_id != experiment_id:
            experiment_id = resolved_experiment_id
        checkpoint_dir = str(resolved_checkpoint_path.parent)
        checkpoint_path = str(resolved_checkpoint_path)
        trial_dir = results_root / "trials" / experiment_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.yaml"
        train_log_path = trial_dir / "train.log"
        trial_config["trainer"]["checkpoint_dir"] = str(trial_checkpoint_root)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(trial_config, handle, sort_keys=False)
        print(
            _canonical_json(
                {
                    "status": "evaluating_existing",
                    "experiment_id": experiment_id,
                    "checkpoint_path": checkpoint_path,
                    "target_epoch": target_epoch,
                }
            )
        )
        try:
            evaluation_start = time.time()
            evaluation = _evaluate_pure_model_rf(trial_config, resolved_checkpoint_path)
            evaluation_runtime_s = time.time() - evaluation_start
            pure_model_rf = float(evaluation["pure_model_rf"])
            sampled_tree = evaluation["sampled_tree"]
            target_tree = evaluation["target_tree"]
            start_tree = evaluation["start_tree"]
            status = "completed"
        except Exception as exc:
            status = "evaluation_failed"
            failure_reason = str(exc)
    else:
        trial_checkpoint_root = checkpoint_root / experiment_id
        trial_dir = results_root / "trials" / experiment_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.yaml"
        train_log_path = trial_dir / "train.log"
        trial_config["trainer"]["checkpoint_dir"] = str(trial_checkpoint_root)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(trial_config, handle, sort_keys=False)

        print(
            _canonical_json(
                {
                    "status": "starting",
                    "experiment_id": experiment_id,
                    "config": str(config_path),
                    "checkpoint_root": str(trial_checkpoint_root),
                    "target_epoch": target_epoch,
                    "overrides": harness_overrides,
                }
            )
        )

        train_returncode = _run_overfit_training(
            python_executable=args.python,
            repo_root=repo_root,
            config_path=config_path,
            train_log_path=train_log_path,
        )
        training_runtime_s = time.time() - training_start

        if train_returncode == 0:
            try:
                resolved_checkpoint_dir, resolved_checkpoint_path = _find_target_checkpoint(
                    trial_checkpoint_root,
                    target_epoch=target_epoch,
                )
                checkpoint_dir = str(resolved_checkpoint_dir)
                checkpoint_path = str(resolved_checkpoint_path)
                evaluation_start = time.time()
                evaluation = _evaluate_pure_model_rf(
                    trial_config, resolved_checkpoint_path
                )
                evaluation_runtime_s = time.time() - evaluation_start
                pure_model_rf = float(evaluation["pure_model_rf"])
                sampled_tree = evaluation["sampled_tree"]
                target_tree = evaluation["target_tree"]
                start_tree = evaluation["start_tree"]
                status = "completed"
            except Exception as exc:
                status = "evaluation_failed"
                failure_reason = str(exc)
        else:
            failure_reason = (
                f"Training subprocess exited with code {train_returncode}."
            )

    if args.evaluate_checkpoint:
        training_runtime_s = 0.0

    total_runtime_s = time.time() - training_start
    record = {
        "dedupe_key": dedupe_key,
        "fingerprint": fingerprint,
        "experiment_id": experiment_id,
        "status": status,
        "created_at": created_at,
        "base_config": str(base_config_path),
        "epochs": args.epochs,
        "target_epoch": target_epoch,
        "note": args.note,
        "git_commit": git_meta["git_commit"],
        "git_dirty_hash": git_meta["git_dirty_hash"],
        "git_status": git_meta["git_status"],
        "checkpoint_root": str(trial_checkpoint_root),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_path,
        "trial_dir": str(trial_dir),
        "train_log_path": str(train_log_path),
        "pure_model_rf": pure_model_rf,
        "training_runtime_s": training_runtime_s,
        "evaluation_runtime_s": evaluation_runtime_s,
        "total_runtime_s": total_runtime_s,
        "overrides": harness_overrides,
        "overrides_json": _canonical_json(harness_overrides),
        "sampled_tree": sampled_tree,
        "target_tree": target_tree,
        "start_tree": start_tree,
        "failure_reason": failure_reason,
        "train_log_tail": _tail_text(train_log_path),
    }
    _append_record(record, jsonl_path=jsonl_path, csv_path=csv_path)

    print(
        _canonical_json(
            {
                "status": status,
                "experiment_id": experiment_id,
                "pure_model_rf": pure_model_rf,
                "checkpoint_path": checkpoint_path,
                "results_jsonl": str(jsonl_path),
                "results_csv": str(csv_path),
                "failure_reason": failure_reason,
            }
        )
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
