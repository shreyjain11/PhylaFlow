#!/usr/bin/env python
"""Per-dataset posterior-vs-sampler KL evaluation.

This is intentionally separate from the lightweight sample-metrics harness.  The
harness pools split counts across unrelated held-out cases; this script computes
topological KL inside each held-out dataset/subset and only then averages.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from ete3 import Tree as EteTree

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
    align_numeric_leaf_labels_to_reference,
    calculate_norm_rf,
    kl_divergence_topological_distributions,
    kl_divergence_tree_topology_distributions,
)


def _split_ids(config: Dict[str, Any]) -> tuple[List[str], List[str]]:
    ids = _get_dataset_ids_from_config(config)
    rng = random.Random(42)
    rng.shuffle(ids)
    if len(ids) < 2:
        return ids, ids
    split = int(0.8 * len(ids))
    return ids[:split], ids[split:]


def _training_module_kwargs(config: Dict[str, Any], model, dataset):
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


def _load_full_module_state_if_available(module, checkpoint_path: str, device):
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


def _temporary_random_seed(seed_value: str):
    class _Context:
        def __enter__(self):
            self._state = random.getstate()
            random.seed(seed_value)

        def __exit__(self, exc_type, exc, tb):
            random.setstate(self._state)

    return _Context()


def _sample_dataset_item(dataset_split, index: int, seed: int, case_offset: int):
    # TreeDataset uses the module-level Python RNG for posterior draw, subset draw,
    # and random start construction.  Pin it here so each eval case is reproducible.
    seed_value = f"{seed}:case:{case_offset}:dataset_item:{index}"
    with _temporary_random_seed(seed_value):
        return dataset_split[int(index)]


def _prune_and_remap_tree(raw_newick: str, seq_ordering_map: Dict[str, str]) -> str:
    keep_names = set(str(name) for name in seq_ordering_map.keys())
    tree = EteTree(str(raw_newick), format=1)
    present = {str(leaf.name) for leaf in tree.iter_leaves()}
    missing = sorted(keep_names - present)
    if missing:
        raise ValueError(f"Posterior tree missing selected leaves: {missing[:5]}")
    tree.prune(sorted(keep_names, key=lambda value: int(value) if value.isdigit() else value), preserve_branch_length=True)
    for leaf in tree.iter_leaves():
        leaf.name = str(seq_ordering_map[str(leaf.name)])
    return tree.write(format=1)


def _sample_posterior_subset_trees(
    dataset_split,
    dataset_key: int | str,
    seq_ordering_map: Dict[str, str],
    count: int,
    rng: random.Random,
) -> List[str]:
    raw_trees = list(dataset_split.return_posterior_trees(dataset_key))
    if not raw_trees:
        raise ValueError(f"No posterior trees for dataset {dataset_key}")

    out: List[str] = []
    attempts = 0
    max_attempts = max(count * 10, count + 100)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        raw_tree = rng.choice(raw_trees)
        try:
            out.append(_prune_and_remap_tree(raw_tree, seq_ordering_map))
        except Exception:
            continue
    if len(out) < count:
        raise RuntimeError(
            f"Only built {len(out)} posterior subset trees for dataset {dataset_key}; "
            f"needed {count}."
        )
    return out


def _aggregate_scalar_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic))
        }
    )
    summary: Dict[str, float] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float, np.generic))
            and np.isfinite(float(row[key]))
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_median"] = float(np.median(arr))
        summary[f"{key}_min"] = float(arr.min())
        summary[f"{key}_max"] = float(arr.max())
        summary[f"{key}_p10"] = float(np.quantile(arr, 0.10))
        summary[f"{key}_p90"] = float(np.quantile(arr, 0.90))
    return summary


def _paired_rf_mean(a_trees: List[str], b_trees: List[str]) -> float:
    if not a_trees or not b_trees:
        return float("nan")
    count = min(len(a_trees), len(b_trees))
    values = [
        float(calculate_norm_rf(a_trees[idx], b_trees[idx]))
        for idx in range(count)
    ]
    return float(np.asarray(values, dtype=np.float64).mean())


def _build_pair_from_sample(sample: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    start_tree = str(sample["start_tree"])
    return {
        "start_tree": start_tree,
        "target_tree": str(sample["target_tree"]),
        "bank_group_key": sample.get("bank_group_key"),
        "dataset_id": sample.get("dataset_id", sample.get("id")),
        "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
        "max_events": int(sample.get("fixed_pair_num_events", 1024)),
        "name_mapping": sample.get("num_to_name"),
        "selected_sequences": sample.get("selected_sequences"),
        "selected_sequence_names": sample.get("selected_sequence_names"),
        "source_bank_index": int(source_index),
    }


def _evaluate_case(
    module: TrainingModule,
    dataset_split,
    source_index: int,
    case_idx: int,
    *,
    num_samples: int,
    seed: int,
    dump_trees_dir: Path | None = None,
) -> Dict[str, Any]:
    case_rng = random.Random(f"{seed}:case:{case_idx}:samples:{source_index}")
    sample = _sample_dataset_item(dataset_split, source_index, seed, case_idx)
    seq_ordering_map = {
        str(key): str(value)
        for key, value in dict(sample.get("seq_ordering_map") or {}).items()
    }
    if not seq_ordering_map:
        raise ValueError(f"Sample for index {source_index} has no seq_ordering_map")

    pair = _build_pair_from_sample(sample, source_index)
    posterior_dataset_key = pair.get("dataset_id")
    if posterior_dataset_key in {None, ""}:
        posterior_dataset_key = source_index
    posterior_trees = _sample_posterior_subset_trees(
        dataset_split,
        posterior_dataset_key,
        seq_ordering_map,
        count=int(num_samples),
        rng=case_rng,
    )

    with torch.inference_mode():
        sample_kwargs = module._build_harness_sample_kwargs(pair, train=True)
        sample_kwargs["return_trace"] = False
        sample_kwargs["trace_state_rf"] = False

        start_trees: List[str] = []
        sampled_trees: List[str] = []
        sample_times: List[float] = []
        for sample_idx in range(int(num_samples)):
            with _temporary_random_seed(
                f"{seed}:case:{case_idx}:start:{sample_idx}:{source_index}"
            ):
                start_tree = dataset_split.sample_random_tree(pair["target_tree"])
            start_tree = str(start_tree)
            start_trees.append(start_tree)
            sample_kwargs["target_trees"] = [posterior_trees[sample_idx]]
            t0 = time.perf_counter()
            sampled, *_ = module.sample([start_tree], **sample_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sample_times.append(time.perf_counter() - t0)
            sampled_tree, _changed = align_numeric_leaf_labels_to_reference(
                sampled[0],
                start_tree,
                target_tree=posterior_trees[sample_idx],
            )
            sampled_trees.append(str(sampled_tree))

    model_split_kl = kl_divergence_topological_distributions(
        posterior_trees,
        sampled_trees,
        num_leaves=int(pair["n_leaves"]),
    )
    model_topology_kl = kl_divergence_tree_topology_distributions(
        posterior_trees,
        sampled_trees,
    )
    random_split_kl = kl_divergence_topological_distributions(
        posterior_trees,
        start_trees,
        num_leaves=int(pair["n_leaves"]),
    )
    random_topology_kl = kl_divergence_tree_topology_distributions(
        posterior_trees,
        start_trees,
    )

    row: Dict[str, Any] = {
        "case_idx": int(case_idx),
        "source_bank_index": int(source_index),
        "dataset_id": str(pair.get("dataset_id")),
        "n_leaves": int(pair["n_leaves"]),
        "num_samples": int(num_samples),
        "model_kl_divergence_topological": float(
            model_split_kl["kl_divergence_topological"]
        ),
        "model_kl_divergence_tree_topology": float(
            model_topology_kl["kl_divergence_tree_topology"]
        ),
        "model_n_shared_topologies": float(model_topology_kl["n_shared_topologies"]),
        "model_posterior_topology_support_recall": float(
            model_topology_kl["posterior_topology_support_recall"]
        ),
        "model_n_unique_sampled_topologies": float(
            model_topology_kl["n_unique_sampled_topologies"]
        ),
        "posterior_n_unique_topologies": float(
            model_topology_kl["n_unique_posterior_topologies"]
        ),
        "random_start_kl_divergence_topological": float(
            random_split_kl["kl_divergence_topological"]
        ),
        "random_start_kl_divergence_tree_topology": float(
            random_topology_kl["kl_divergence_tree_topology"]
        ),
        "random_start_n_shared_topologies": float(
            random_topology_kl["n_shared_topologies"]
        ),
        "random_start_posterior_topology_support_recall": float(
            random_topology_kl["posterior_topology_support_recall"]
        ),
        "rf_norm_model_vs_posterior_paired_mean": _paired_rf_mean(
            sampled_trees,
            posterior_trees,
        ),
        "rf_norm_start_vs_posterior_paired_mean": _paired_rf_mean(
            start_trees,
            posterior_trees,
        ),
        "sample_sec_mean": float(np.asarray(sample_times, dtype=np.float64).mean()),
        "sample_sec_total": float(np.asarray(sample_times, dtype=np.float64).sum()),
    }
    if dump_trees_dir is not None:
        dump_trees_dir.mkdir(parents=True, exist_ok=True)
        dump_payload = {
            "case_idx": int(case_idx),
            "source_bank_index": int(source_index),
            "dataset_id": str(pair.get("dataset_id")),
            "posterior_trees": posterior_trees,
            "start_trees": start_trees,
            "sampled_trees": sampled_trees,
        }
        dump_path = dump_trees_dir / f"case_{case_idx:04d}_{pair.get('dataset_id')}.json"
        with dump_path.open("w", encoding="utf-8") as handle:
            json.dump(dump_payload, handle)
            handle.write("\n")
        row["tree_dump_path"] = str(dump_path.resolve())
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint to evaluate, or 'random-init' to leave model weights random.",
    )
    parser.add_argument("--sample-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-datasets", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default=None)
    parser.add_argument("--dump-trees", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "per_dataset_rows.jsonl"
    summary_path = output_dir / "summary.json"

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    trainer_cfg = config.setdefault("trainer", {})
    data_cfg = config.setdefault("data", {})
    data_cfg["sample_metrics_config_path"] = str(args.sample_config)
    trainer_cfg["record"] = False
    trainer_cfg["sample_metrics_tree_dump_enabled"] = False
    trainer_cfg["sample_metrics_checkpoint_enabled"] = False
    trainer_cfg["sample_metrics_trace_path"] = None
    if args.label:
        trainer_cfg["wandb_name"] = str(args.label)

    effective_config_path = output_dir / "effective_config.yaml"
    with effective_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    _configure_torch_runtime()
    _set_global_seed(trainer_cfg.get("seed"))
    train_ids, test_ids = _split_ids(config)

    setup_start = time.perf_counter()
    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    phyla_flow = return_model(config)
    random_init = str(args.checkpoint).lower() in {"random-init", "random_init", "none"}
    if not random_init:
        phyla_flow = _load_model_init_checkpoint(phyla_flow, args.checkpoint, device)
    phyla_flow.to(device)
    module = TrainingModule(**_training_module_kwargs(config, phyla_flow, dataset))
    module.legacy_first_hit_gather_only = bool(
        trainer_cfg.get("legacy_first_hit_gather_only", False)
    )
    if random_init:
        module_state_info = {
            "loaded_full_module_state": 0,
            "random_init": 1,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    else:
        module_state_info = _load_full_module_state_if_available(
            module,
            args.checkpoint,
            device,
        )
    module.to(device)
    module.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    setup_sec = time.perf_counter() - setup_start

    dataset_split = module._sample_metrics_dataset_split(train=True)
    split_len = len(dataset_split)
    if int(args.num_datasets) > split_len:
        raise ValueError(
            f"Requested {args.num_datasets} datasets but split only has {split_len}"
        )
    rng = random.Random(int(args.seed))
    selected_indices = rng.sample(range(split_len), k=int(args.num_datasets))

    rows: List[Dict[str, Any]] = []
    eval_start = time.perf_counter()
    dump_trees_dir = output_dir / "tree_dumps" if args.dump_trees else None
    with rows_path.open("w", encoding="utf-8") as rows_handle:
        for case_idx, source_index in enumerate(selected_indices):
            case_start = time.perf_counter()
            row = _evaluate_case(
                module,
                dataset_split,
                int(source_index),
                int(case_idx),
                num_samples=int(args.num_samples),
                seed=int(args.seed),
                dump_trees_dir=dump_trees_dir,
            )
            row["case_elapsed_sec"] = float(time.perf_counter() - case_start)
            rows.append(row)
            rows_handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows_handle.flush()
            print(
                json.dumps(
                    {
                        "case": case_idx + 1,
                        "num_cases": int(args.num_datasets),
                        "dataset_id": row["dataset_id"],
                        "model_kl": row["model_kl_divergence_topological"],
                        "random_start_kl": row[
                            "random_start_kl_divergence_topological"
                        ],
                        "case_elapsed_sec": row["case_elapsed_sec"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed_sec = time.perf_counter() - eval_start
    aggregate = _aggregate_scalar_rows(rows)
    summary = {
        "label": args.label,
        "checkpoint": (
            "random-init"
            if random_init
            else str(Path(args.checkpoint).resolve())
        ),
        "config": str(Path(args.config).resolve()),
        "sample_config": str(Path(args.sample_config).resolve()),
        "effective_config": str(effective_config_path.resolve()),
        "rows_path": str(rows_path.resolve()),
        "num_datasets": int(args.num_datasets),
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "selected_indices": selected_indices,
        "setup_sec": float(setup_sec),
        "elapsed_sec": float(elapsed_sec),
        "module_state_info": module_state_info,
        "aggregate": aggregate,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
