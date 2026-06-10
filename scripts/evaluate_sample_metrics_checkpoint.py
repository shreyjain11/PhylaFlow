#!/usr/bin/env python
"""Run sample-metrics harness generation from a saved checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import PhylaDataModule
from model.model import return_model
from run.TrainingModule import TrainingModule
from run.run import (
    _configure_torch_runtime,
    _get_dataset_ids_from_config,
    _load_model_init_checkpoint,
    _set_global_seed,
)
from utils.metric_utils import (
    kl_divergence_topological_distributions,
    kl_divergence_tree_topology_distributions,
)


def _split_ids(config):
    ids = _get_dataset_ids_from_config(config)
    rng = random.Random(42)
    rng.shuffle(ids)
    if len(ids) < 2:
        return ids, ids
    split = int(0.8 * len(ids))
    return ids[:split], ids[split:]


def _training_module_kwargs(config, model, dataset):
    trainer_cfg = dict(config.get("trainer") or {})
    signature = inspect.signature(TrainingModule.__init__)
    kwargs = {
        name: trainer_cfg[name]
        for name in signature.parameters
        if name not in {"self", "model", "dataset"} and name in trainer_cfg
    }
    kwargs["model"] = model
    kwargs["dataset"] = dataset
    kwargs.setdefault("logger", None)
    return kwargs


def _load_full_module_state_if_available(module, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict")
    if not state_dict:
        return {
            "loaded_full_module_state": 0,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    return {
        "loaded_full_module_state": 1,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _latest_tree_dump(tree_dump_dir):
    candidates = sorted(
        Path(tree_dump_dir).glob("*_trees.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prefix_metrics(prefix, metrics):
    return {f"{prefix}{key}": value for key, value in dict(metrics).items()}


def _target_distribution_metrics(target_trees, sampled_trees, n_leaves_values):
    if not target_trees or not sampled_trees or len(target_trees) != len(sampled_trees):
        return {}
    metrics = {}
    if n_leaves_values and len(set(n_leaves_values)) == 1:
        metrics.update(
            kl_divergence_topological_distributions(
                target_trees,
                sampled_trees,
                num_leaves=int(n_leaves_values[0]),
            )
        )
    metrics.update(kl_divergence_tree_topology_distributions(target_trees, sampled_trees))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--start-mode",
        choices=("unseen", "bank"),
        default="unseen",
        help=(
            "Use metric-encoder unseen starts, or evaluate the bank/index start "
            "trees directly. Bank mode is useful for topology-stream cases whose "
            "source-case phyla mapping is tied to the original start tree."
        ),
    )
    parser.add_argument(
        "--zero-shot-random-start",
        action="store_true",
        help="Also run the zero-shot random-start block after bank/index evaluation.",
    )
    parser.add_argument("--mrbayes", action="store_true")
    parser.add_argument("--relaxed-likelihood", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_dump_dir = output_dir / "generated_trees"
    tree_dump_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.sample_config, "r", encoding="utf-8") as handle:
        sample_config = yaml.safe_load(handle)

    trainer_cfg = config.setdefault("trainer", {})
    sample_trainer_cfg = dict(sample_config.get("trainer") or {})
    data_cfg = config.setdefault("data", {})
    data_cfg["sample_metrics_config_path"] = str(args.sample_config)
    for key in (
        "sample_metrics_num_pairs",
        "sample_metrics_unseen_start_seed",
        "sample_metrics_unseen_pair_selection_mode",
        "sample_metrics_unseen_start_max_duplicate_tries",
        "sample_metrics_unseen_start_metric_encoder_path",
        "sample_metrics_branch_relaxer_checkpoint_path",
        "sample_metrics_mrbayes20k_num_starts",
        "sample_metrics_mrbayes20k_ngen",
        "sample_metrics_mrbayes20k_samplefreq",
        "sample_metrics_mrbayes20k_printfreq",
        "sample_metrics_mrbayes20k_max_workers",
        "sample_metrics_mrbayes20k_timeout_sec",
        "sample_metrics_mrbayes20k_dataset_pickle_path",
        "sample_metrics_mrbayes20k_golden_root",
        "sample_metrics_mrbayes20k_bin",
    ):
        if key in sample_trainer_cfg:
            trainer_cfg[key] = sample_trainer_cfg[key]
    trainer_cfg["record"] = False
    trainer_cfg["sample_metrics_trace_path"] = str(output_dir / "metrics.jsonl")
    trainer_cfg["sample_metrics_tree_dump_enabled"] = True
    trainer_cfg["sample_metrics_tree_dump_dir"] = str(tree_dump_dir)
    trainer_cfg["sample_metrics_checkpoint_enabled"] = False
    trainer_cfg["sample_metrics_relaxed_likelihood_enabled"] = bool(
        args.relaxed_likelihood
    )
    trainer_cfg["sample_metrics_mrbayes20k_enabled"] = bool(args.mrbayes)
    trainer_cfg["sample_metrics_unseen_start_eval"] = args.start_mode == "unseen"
    trainer_cfg["sample_metrics_zero_shot_random_start_eval"] = bool(
        args.zero_shot_random_start
    )
    trainer_cfg["sample_metrics_unseen_pair_selection_mode"] = trainer_cfg.get(
        "sample_metrics_unseen_pair_selection_mode",
        "random_bank",
    )
    if args.num_pairs is not None:
        trainer_cfg["sample_metrics_num_pairs"] = int(args.num_pairs)
    if args.label:
        trainer_cfg["wandb_name"] = str(args.label)

    effective_config_path = output_dir / "effective_config.yaml"
    with effective_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    _configure_torch_runtime()
    _set_global_seed(trainer_cfg.get("seed"))

    train_ids, test_ids = _split_ids(config)
    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    phyla_flow = return_model(config)
    phyla_flow = _load_model_init_checkpoint(phyla_flow, args.checkpoint, device)
    phyla_flow.to(device)

    module = TrainingModule(**_training_module_kwargs(config, phyla_flow, dataset))
    module.legacy_first_hit_gather_only = bool(
        trainer_cfg.get("legacy_first_hit_gather_only", False)
    )
    module_state_info = _load_full_module_state_if_available(
        module,
        args.checkpoint,
        device,
    )
    module._sample_metrics_can_write_artifacts = lambda: True
    module.to(device)
    module.eval()

    started = time.time()
    with torch.inference_mode():
        metrics = module.sample_compare_harness(train=True)
    elapsed_sec = time.time() - started

    dump_path = _latest_tree_dump(tree_dump_dir)
    rows = _read_jsonl(dump_path) if dump_path else []
    start_trees = [row.get("start_tree") for row in rows if row.get("start_tree")]
    sampled_trees = [row.get("sampled_tree") for row in rows if row.get("sampled_tree")]
    target_trees = [row.get("target_tree") for row in rows if row.get("target_tree")]
    n_leaves_values = [
        int(row["n_leaves"])
        for row in rows
        if row.get("n_leaves") is not None
    ]

    random_start_metrics = {}
    if start_trees:
        random_start_metrics.update(
            _prefix_metrics(
                "random_start_",
                module._posterior_reference_metrics(start_trees, train=True),
            )
        )
        random_start_metrics.update(
            _prefix_metrics(
                "random_start_target_",
                _target_distribution_metrics(target_trees, start_trees, n_leaves_values),
            )
        )

    sampled_target_metrics = _prefix_metrics(
        "sampled_target_",
        _target_distribution_metrics(target_trees, sampled_trees, n_leaves_values),
    )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "sample_config": str(Path(args.sample_config).resolve()),
        "effective_config": str(effective_config_path.resolve()),
        "tree_dump_jsonl": str(dump_path.resolve()) if dump_path else None,
        "elapsed_sec": elapsed_sec,
        "num_rows": len(rows),
        "module_state_info": module_state_info,
        "metrics": metrics,
        "random_start_metrics": random_start_metrics,
        "sampled_target_metrics": sampled_target_metrics,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
