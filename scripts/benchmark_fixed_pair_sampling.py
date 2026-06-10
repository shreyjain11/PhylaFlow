#!/usr/bin/env python
"""Benchmark fixed-pair PhylaFlow sampling for decoder variants."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

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
from utils.metric_utils import align_numeric_leaf_labels_to_reference, calculate_norm_rf


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
    kwargs["record"] = False
    return kwargs


def _load_module(config, checkpoint_path, device):
    train_ids, test_ids = _split_ids(config)
    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    model = return_model(config)
    model = _load_model_init_checkpoint(model, checkpoint_path, device)
    model.to(device)
    module = TrainingModule(**_training_module_kwargs(config, model, dataset))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    module.to(device)
    module.eval()
    return module, {
        "missing_keys": list(missing)[:20],
        "missing_key_count": len(missing),
        "unexpected_keys": list(unexpected)[:20],
        "unexpected_key_count": len(unexpected),
    }


def _fixed_pair(module, train=True):
    fixed_pair = module._get_fixed_pair_sampling_details(train=train)
    if fixed_pair is None:
        raise RuntimeError("No fixed pair available in the configured dataset.")
    start_tree = fixed_pair.get("random_tree", fixed_pair.get("start_tree"))
    target_tree = fixed_pair.get("effective_target_tree", fixed_pair.get("target_tree"))
    dataset_split = (
        getattr(module.dataset, "dataset_train", None)
        if train
        else getattr(module.dataset, "dataset_val", None)
    )
    name_mapping = fixed_pair.get("name_mapping")
    if name_mapping is None and hasattr(dataset_split, "return_nexus_number_to_name"):
        name_mapping = dataset_split.return_nexus_number_to_name(0)
    return {
        "start_tree": start_tree,
        "target_tree": target_tree,
        "bank_group_key": fixed_pair.get("bank_group_key") or fixed_pair.get("group_key"),
        "dataset_id": fixed_pair.get("dataset_id"),
        "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
        "max_events": int(
            len(fixed_pair.get("final_labels", []) or [])
            or fixed_pair.get("fixed_pair_num_events", 1024)
        ),
        "name_mapping": name_mapping,
    }


def _case_index_from_group_key(group_key):
    if group_key is None:
        return None
    match = re.search(r"case(\d+)$", str(group_key))
    if match is None:
        return None
    return int(match.group(1))


def _load_json_case_pairs(config, limit):
    data_cfg = config.get("data") or {}
    start_path = data_cfg.get("overfit_fixed_pair_start_tree_json_path")
    target_path = data_cfg.get("overfit_fixed_pair_target_tree_json_path")
    if not start_path or not target_path:
        return []

    start_path = Path(start_path)
    target_path = Path(target_path)
    match = re.match(r"^(?P<prefix>.*case)(?P<case>\d+)_start\.json$", start_path.name)
    if match is None:
        return []

    pairs = []
    width = len(match.group("case"))
    for start_file in sorted(start_path.parent.glob(f"{match.group('prefix')}*_start.json")):
        case_match = re.match(
            rf"^{re.escape(match.group('prefix'))}(?P<case>\d+)_start\.json$",
            start_file.name,
        )
        if case_match is None:
            continue
        case_id = case_match.group("case")
        target_file = target_path.parent / f"{match.group('prefix')}{case_id}_target.json"
        if not target_file.exists():
            continue
        start_payload = json.loads(start_file.read_text(encoding="utf-8"))
        target_payload = json.loads(target_file.read_text(encoding="utf-8"))
        start_tree = str(start_payload["tree"])
        target_tree = str(target_payload["tree"])
        group_key = (
            target_payload.get("group_key")
            or start_payload.get("group_key")
            or f"{match.group('prefix')}{int(case_id):0{width}d}"
        )
        pairs.append(
            {
                "start_tree": start_tree,
                "target_tree": target_tree,
                "bank_group_key": group_key,
                "dataset_id": str(data_cfg.get("short_run_dataset_id", "DS2")).upper(),
                "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                "max_events": 1024,
                "name_mapping": None,
            }
        )
        if len(pairs) >= int(limit):
            break
    return pairs


def _sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _trace_metrics(module, trace):
    metrics = module._trace_topology_decoder_metrics(trace, prefix="")
    metrics["velocity_events"] = float(len((trace or {}).get("velocity", []) or []))
    metrics["stopped_for_no_valid_merge"] = float(
        1.0 if (trace or {}).get("stopped_for_no_valid_merge", False) else 0.0
    )
    metrics["stopped_for_repeated_topology"] = float(
        1.0 if (trace or {}).get("stopped_for_repeated_topology", False) else 0.0
    )
    return metrics


def _expand_fixed_pair_sample_kwargs(module, pair, batch_size, pairs=None):
    pairs = list(pairs or [pair])
    if not pairs:
        pairs = [pair]
    sample_kwargs = module._build_harness_sample_kwargs(pairs[0], train=True)
    batch_size = max(1, int(batch_size))
    pairs = pairs[:batch_size]
    if batch_size == 1:
        return [pairs[0]["start_tree"]], sample_kwargs

    sample_kwargs["target_trees"] = [item["target_tree"] for item in pairs]
    case_indices = sample_kwargs.get("case_indices")
    pair_case_indices = [
        _case_index_from_group_key(item.get("bank_group_key") or item.get("group_key"))
        for item in pairs
    ]
    if all(value is not None for value in pair_case_indices):
        sample_kwargs["case_indices"] = [int(value) for value in pair_case_indices]
    elif case_indices is not None:
        case_values = torch.as_tensor(case_indices).reshape(-1).tolist()
        sample_kwargs["case_indices"] = (case_values * batch_size)[:batch_size]

    for key, value in list(sample_kwargs.items()):
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == 1:
            sample_kwargs[key] = value.expand(batch_size, *value.shape[1:]).contiguous()
        elif isinstance(value, list) and len(value) == 1 and key != "target_trees":
            sample_kwargs[key] = value * batch_size
    return [item["start_tree"] for item in pairs], sample_kwargs


def _run_once(module, pair, decoder, fallback, device, batch_size=1, pairs=None, serial=False):
    old_decoder = module.topology_decoder
    old_fallback = module.birthset_fallback
    module.topology_decoder = decoder
    module.birthset_fallback = fallback
    _sync(device)
    start = time.perf_counter()
    pairs_for_run = list(pairs or [pair])[: max(1, int(batch_size))]
    if serial:
        sampled_trees = []
        traces = []
        with torch.inference_mode():
            for single_pair in pairs_for_run:
                starts, sample_kwargs = _expand_fixed_pair_sample_kwargs(
                    module,
                    single_pair,
                    1,
                    pairs=[single_pair],
                )
                sample_kwargs["return_trace"] = True
                sample_kwargs["trace_state_rf"] = False
                single_sampled, *_rest, single_trace = module.sample(
                    starts,
                    **sample_kwargs,
                )
                sampled_trees.extend(single_sampled)
                traces.append(single_trace)
        trace = {
            "velocity": [
                item
                for single_trace in traces
                for item in (single_trace or {}).get("velocity", [])
            ],
            "autoregressive": [
                item
                for single_trace in traces
                for item in (single_trace or {}).get("autoregressive", [])
            ],
            "stopped_for_no_valid_merge": any(
                bool((single_trace or {}).get("stopped_for_no_valid_merge", False))
                for single_trace in traces
            ),
            "stopped_for_repeated_topology": any(
                bool((single_trace or {}).get("stopped_for_repeated_topology", False))
                for single_trace in traces
            ),
            "birthset_incomplete_without_fallback": any(
                bool((single_trace or {}).get("birthset_incomplete_without_fallback", False))
                for single_trace in traces
            ),
        }
    else:
        starts, sample_kwargs = _expand_fixed_pair_sample_kwargs(
            module,
            pair,
            batch_size,
            pairs=pairs_for_run,
        )
        sample_kwargs["return_trace"] = True
        sample_kwargs["trace_state_rf"] = False
        with torch.inference_mode():
            sampled_trees, *_rest, trace = module.sample(starts, **sample_kwargs)
    _sync(device)
    elapsed = time.perf_counter() - start
    module.topology_decoder = old_decoder
    module.birthset_fallback = old_fallback
    rf_values = []
    remapped_values = []
    for sampled_tree_raw, target_pair in zip(sampled_trees, pairs_for_run):
        sampled_tree, remapped = align_numeric_leaf_labels_to_reference(
            sampled_tree_raw,
            target_pair["start_tree"],
            target_tree=target_pair["target_tree"],
        )
        rf_values.append(float(calculate_norm_rf(sampled_tree, target_pair["target_tree"])))
        remapped_values.append(bool(remapped))
    metrics = _trace_metrics(module, trace)
    metrics.update(
        {
            "elapsed_sec": float(elapsed),
            "elapsed_sec_per_tree": float(elapsed) / float(max(1, int(batch_size))),
            "rf_norm": float(statistics.mean(rf_values)),
            "rf_norm_max_batch": float(max(rf_values)),
            "sampled_tree_label_remapped": bool(any(remapped_values)),
            "batch_size": int(batch_size),
            "serial": bool(serial),
        }
    )
    return metrics


def _summarize(rows):
    elapsed = [float(row["elapsed_sec"]) for row in rows]
    rf = [float(row["rf_norm"]) for row in rows]
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), (int, float))})
    summary = {
        "runs": len(rows),
        "elapsed_sec_mean": statistics.mean(elapsed),
        "elapsed_sec_median": statistics.median(elapsed),
        "elapsed_sec_min": min(elapsed),
        "elapsed_sec_max": max(elapsed),
        "rf_norm_mean": statistics.mean(rf),
        "rf_norm_min": min(rf),
        "rf_norm_max": max(rf),
    }
    for key in keys:
        if key in summary or key in {"elapsed_sec", "rf_norm"}:
            continue
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = statistics.mean(values)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--distinct-starts",
        action="store_true",
        help="Use distinct case JSON start/target pairs instead of duplicating one pair.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Sample each item in the batch sequentially and report the combined time.",
    )
    parser.add_argument(
        "--decoder",
        action="append",
        choices=("ar", "birthset", "birthset_with_ar_fallback"),
        default=None,
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _configure_torch_runtime()
    _set_global_seed((config.get("trainer") or {}).get("seed"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    module, load_info = _load_module(config, args.checkpoint, device)
    pair = _fixed_pair(module, train=True)
    pairs = None
    if args.distinct_starts:
        pairs = _load_json_case_pairs(config, max(1, int(args.batch_size)))
        if len(pairs) < int(args.batch_size):
            raise RuntimeError(
                f"Requested {args.batch_size} distinct starts but found {len(pairs)}."
            )

    decoders = args.decoder or ["birthset", "ar"]
    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "load_info": load_info,
        "pair": {
            "n_leaves": pair["n_leaves"],
            "max_events": pair["max_events"],
            "start_rf_norm": float(calculate_norm_rf(pair["start_tree"], pair["target_tree"])),
            "distinct_starts": bool(args.distinct_starts),
            "serial": bool(args.serial),
            "batch_size": int(args.batch_size),
        },
        "decoders": {},
    }
    for decoder in decoders:
        fallback = "none"
        if decoder == "birthset_with_ar_fallback":
            fallback = "ar"
        rows = []
        for _ in range(max(0, int(args.warmup))):
            _run_once(
                module,
                pair,
                decoder,
                fallback,
                device,
                batch_size=args.batch_size,
                pairs=pairs,
                serial=args.serial,
            )
        for _ in range(max(1, int(args.repeats))):
            rows.append(
                _run_once(
                    module,
                    pair,
                    decoder,
                    fallback,
                    device,
                    batch_size=args.batch_size,
                    pairs=pairs,
                    serial=args.serial,
                )
            )
        results["decoders"][decoder] = {
            "fallback": fallback,
            "summary": _summarize(rows),
            "runs": rows,
        }

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
