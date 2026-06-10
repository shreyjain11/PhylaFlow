#!/usr/bin/env python
import argparse
import json
import os
import sys

import torch
import yaml

sys.path.append(os.getcwd())

from model.model import return_model
from run.TrainingModule import (
    TrainingModule,
    _birthset_full_mask,
    _birthset_map_split_to_local_subset,
    _build_case_index_tensor_from_group_keys,
    _move_tokenized_batch_to_device,
    _tokenize_trees_with_structural_cache,
)
from run.run import _birthset_trainer_kwargs
from utils.bhv_movie import build_tree_from_splits
from utils.bhv_utils import (
    BHVEncoder,
    get_structural_polytomy_groups_from_newick,
    return_tree_boundary_merge_paths,
)
from utils.metric_utils import calculate_norm_rf
from utils.random_tree import Tree


def _load_tree_json(path):
    payload = json.load(open(path, "r", encoding="utf-8"))
    tree = (
        payload.get("target_tree")
        or payload.get("final_tree")
        or payload.get("start_tree")
        or payload.get("tree")
    )
    if not tree:
        raise ValueError(f"No tree field found in {path}")
    return str(tree), payload


def _latest_sample_checkpoint(config):
    root = config.get("trainer", {}).get("checkpoint_dir")
    if not root:
        return None
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.startswith("sample-metrics-") or not filename.endswith(".ckpt"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                matches.append((os.path.getmtime(path), path))
            except OSError:
                pass
    if not matches:
        return None
    return sorted(matches)[-1][1]


def _make_module(config, checkpoint_path, device):
    phyla_flow = return_model(config)
    module = TrainingModule(
        model=phyla_flow,
        dataset=None,
        lr=float(config.get("trainer", {}).get("lr", 1e-4)),
        optimizer_name=config.get("trainer", {}).get("optimizer_name", "adamw"),
        record=False,
        epochs=int(config.get("trainer", {}).get("epochs", 1)),
        autoregressive_use_time=bool(
            config.get("trainer", {}).get("autoregressive_use_time", False)
        ),
        **_birthset_trainer_kwargs(config),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    module.to(device)
    module.eval()
    return module, {
        "checkpoint_path": checkpoint_path,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_examples": list(missing)[:10],
        "unexpected_key_examples": list(unexpected)[:10],
    }


def _tokenized_existing_lengths(module, newick, tokenized):
    tree = Tree(newick)
    masks, lengths = BHVEncoder().return_BHV_encoding(tree)
    bhv_lengths = {
        int(mask): float(length)
        for mask, length in zip(masks, lengths)
        if length is not None
    }
    edge_masks = [int(mask) for mask in tokenized[-1][0] if int(mask) != 0]
    full_mask = _birthset_full_mask(tree.n_leaves)
    result = {}
    for mask in edge_masks:
        length = bhv_lengths.get(mask)
        if length is None and full_mask:
            length = bhv_lengths.get(full_mask ^ mask)
        if length is not None and float(length) > 1e-8:
            result[int(mask)] = float(length)
    return result, tree.n_leaves, tree.id_to_name, full_mask


def _canonical_unrooted(mask, full_mask):
    mask = int(mask) & int(full_mask)
    comp = int(full_mask) ^ mask
    return min(mask, comp)


def _split_set_overlap(selected, gold, full_mask):
    selected_set = {_canonical_unrooted(split, full_mask) for split in selected}
    gold_set = {_canonical_unrooted(split, full_mask) for split in gold}
    hits = len(selected_set & gold_set)
    precision = hits / len(selected_set) if selected_set else 0.0
    recall = hits / len(gold_set) if gold_set else 0.0
    return {
        "selected_gold_hits": hits,
        "selected_precision_vs_gold_births": precision,
        "selected_recall_vs_gold_births": recall,
        "selected_count": len(selected_set),
        "gold_count": len(gold_set),
    }


def _plan_boundary_with_optional_gold_candidates(
    module,
    outputs,
    existing_splits,
    num_leaves,
    gold_births,
    *,
    oracle_gold_candidates=False,
):
    if not oracle_gold_candidates:
        return module._plan_birthset_boundary_splits(
            outputs,
            existing_splits,
            num_leaves,
        )

    full_mask = _birthset_full_mask(num_leaves)
    max_bit_length = int(full_mask).bit_length()
    for output in outputs or []:
        for split in output.get("splits_represented", []) or []:
            max_bit_length = max(max_bit_length, int(split).bit_length())
    for split in existing_splits or []:
        max_bit_length = max(max_bit_length, int(split).bit_length())
    for split in gold_births or []:
        max_bit_length = max(max_bit_length, int(split).bit_length())
    if max_bit_length > int(full_mask).bit_length():
        full_mask = (1 << int(max_bit_length)) - 1

    planned_existing = {int(split) for split in existing_splits}
    selected_all = []
    metrics = {
        "num_polytomies": 0.0,
        "num_candidate_splits": 0.0,
        "num_selected_birth_splits": 0.0,
        "num_required_birth_splits": 0.0,
        "fraction_resolved_without_fallback": 0.0,
        "num_ar_fallback_calls": 0.0,
        "num_transformer_forwards": 1.0,
    }
    candidate_infos = []
    sorted_outputs = sorted(
        outputs,
        key=lambda output: float(output["polytomy_pred"].detach().cpu().item()),
        reverse=True,
    )
    resolved_count = 0
    for group in sorted_outputs:
        component_masks = [int(split) for split in group["splits_represented"]]
        group_gold = [
            int(split)
            for split in gold_births
            if _birthset_map_split_to_local_subset(split, component_masks)
            is not None
        ]
        candidate_info = module._birthset_build_candidates(
            component_masks,
            num_leaves,
            gold_splits=group_gold,
            component_embeddings=group["group_embeddings"],
            context=group.get("graph_context"),
            train=True,
        )
        candidate_info = dict(candidate_info)
        candidate_info["component_masks"] = component_masks
        candidate_info["group_gold"] = group_gold
        candidate_infos.append(candidate_info)

        required = module._birthset_num_required_splits(
            len(component_masks),
            component_masks=component_masks,
            full_mask=candidate_info.get("full_mask", full_mask),
        )
        if required <= 0:
            continue
        metrics["num_polytomies"] += 1.0
        metrics["num_required_birth_splits"] += float(required)
        candidates = candidate_info["candidates"]
        metrics["num_candidate_splits"] += float(len(candidates))
        if not candidates:
            metrics["num_ar_fallback_calls"] += 1.0
            continue
        local_subsets = [int(item["local_subset"]) for item in candidates]
        with torch.inference_mode():
            logits = module.birthset_topology_head(
                group["group_embeddings"],
                local_subsets,
                context=group.get("graph_context"),
            )
        selected = module._birthset_select_compatible_top_k(
            candidates,
            logits,
            required,
            planned_existing,
            candidate_info.get("full_mask", full_mask),
        )
        if len(selected) < int(required):
            metrics["num_ar_fallback_calls"] += 1.0
        else:
            resolved_count += 1
        for item in selected:
            split = int(item["split_mask"])
            if split in planned_existing:
                continue
            selected_all.append(item)
            planned_existing.add(split)

    metrics["num_selected_birth_splits"] = float(len(selected_all))
    if metrics["num_polytomies"] > 0.0:
        metrics["fraction_resolved_without_fallback"] = (
            float(resolved_count) / metrics["num_polytomies"]
        )
    return {
        "selected": selected_all,
        "metrics": metrics,
        "candidate_infos": candidate_infos,
        "oracle_gold_candidates": True,
    }


def evaluate(config, checkpoint_path, device, *, oracle_gold_candidates=False):
    data_cfg = config.get("data", {})
    start_path = data_cfg.get("overfit_fixed_pair_start_tree_json_path") or (
        data_cfg.get("overfit_fixed_pair_start_tree_json_paths") or [None]
    )[0]
    target_path = data_cfg.get("overfit_fixed_pair_target_tree_json_path") or (
        data_cfg.get("overfit_fixed_pair_target_tree_json_paths") or [None]
    )[0]
    if not start_path or not target_path:
        raise ValueError("Config must define fixed-pair start/target JSON paths.")

    start_tree, start_payload = _load_tree_json(start_path)
    target_tree, target_payload = _load_tree_json(target_path)
    group_key = start_payload.get("group_key") or target_payload.get("group_key")
    module, load_info = _make_module(config, checkpoint_path, device)

    paths = return_tree_boundary_merge_paths(
        start_tree,
        target_tree,
        legacy_training_semantics=False,
    )
    case_indices = None
    if group_key is not None:
        case_indices = _build_case_index_tensor_from_group_keys(
            [str(group_key)],
            device=device,
        )

    rows = []
    for path in paths:
        boundary_newick = str(path["start_newick"])
        tokenized = _move_tokenized_batch_to_device(
            _tokenize_trees_with_structural_cache(module, [boundary_newick]),
            device,
        )
        existing, n_leaves, mapping, full_mask = _tokenized_existing_lengths(
            module,
            boundary_newick,
            tokenized,
        )
        groups = [
            tuple(int(component) for component in group)
            for group in get_structural_polytomy_groups_from_newick(boundary_newick)
        ]
        t = torch.tensor([float(path["boundary_index"])], dtype=torch.float32, device=device)
        with torch.inference_mode():
            outputs = module.forward(
                tokenized,
                t,
                None,
                autoregressive=True,
                autoregressive_component_groups=[groups],
                autoregressive_case_indices=case_indices,
            )
            gold_births = [int(split) for split in path.get("births", [])]
            plan = _plan_boundary_with_optional_gold_candidates(
                module,
                outputs,
                existing.keys(),
                n_leaves,
                gold_births,
                oracle_gold_candidates=oracle_gold_candidates,
            )

        selected = [int(item["split_mask"]) for item in plan.get("selected", [])]
        candidate_gold_hits = 0
        candidate_gold_count = 0
        candidate_gold_by_group = []
        candidate_infos = plan.get("candidate_infos")
        if candidate_infos is None:
            candidate_infos = []
            for group in outputs:
                component_masks = [
                    int(split) for split in group.get("splits_represented", [])
                ]
                group_gold = [
                    int(split)
                    for split in gold_births
                    if _birthset_map_split_to_local_subset(split, component_masks)
                    is not None
                ]
                candidate_info = module._birthset_build_candidates(
                    component_masks,
                    n_leaves,
                    gold_splits=group_gold,
                    component_embeddings=group.get("group_embeddings"),
                    context=group.get("graph_context"),
                    train=False,
                )
                candidate_info = dict(candidate_info)
                candidate_info["component_masks"] = component_masks
                candidate_info["group_gold"] = group_gold
                candidate_infos.append(candidate_info)
        for candidate_info in candidate_infos:
            component_masks = candidate_info.get("component_masks", [])
            group_gold = candidate_info.get("group_gold", [])
            if not group_gold:
                continue
            candidate_splits = {
                _canonical_unrooted(item["split_mask"], full_mask)
                for item in candidate_info.get("candidates", [])
            }
            group_gold_set = {
                _canonical_unrooted(split, full_mask) for split in group_gold
            }
            hits = len(candidate_splits & group_gold_set)
            candidate_gold_hits += int(hits)
            candidate_gold_count += int(len(group_gold_set))
            candidate_gold_by_group.append(
                {
                    "components": len(component_masks),
                    "gold": int(len(group_gold_set)),
                    "hits": int(hits),
                    "candidates": int(len(candidate_splits)),
                }
            )
        decoded_lengths = dict(existing)
        for split in selected:
            decoded_lengths[int(split)] = float(module.birthset_birth_length)
        _tree_obj, decoded_newick = build_tree_from_splits(
            list(decoded_lengths.keys()),
            decoded_lengths,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )
        overlap = _split_set_overlap(selected, gold_births, full_mask)
        metrics = dict(plan.get("metrics", {}) or {})
        row = {
            "boundary_index": int(path["boundary_index"]),
            "num_polytomy_groups": len(groups),
            "rf_to_gold_next_orthant": float(
                calculate_norm_rf(decoded_newick, path["end_newick"])
            ),
            "rf_boundary_start_to_gold_next_orthant": float(
                calculate_norm_rf(boundary_newick, path["end_newick"])
            ),
            "rf_decoded_to_target": float(calculate_norm_rf(decoded_newick, target_tree)),
            "required_birth_splits": float(metrics.get("num_required_birth_splits", 0.0)),
            "selected_birth_splits": len(selected),
            "candidate_splits": float(metrics.get("num_candidate_splits", 0.0)),
            "candidate_gold_hits": int(candidate_gold_hits),
            "candidate_gold_count": int(candidate_gold_count),
            "candidate_recall_vs_gold_births": (
                float(candidate_gold_hits) / float(candidate_gold_count)
                if candidate_gold_count
                else 0.0
            ),
            "candidate_gold_by_group": candidate_gold_by_group,
            "ar_fallback_calls_metric": float(metrics.get("num_ar_fallback_calls", 0.0)),
            "fraction_resolved_without_fallback": float(
                metrics.get("fraction_resolved_without_fallback", 0.0)
            ),
            **overlap,
        }
        rows.append(row)

    return {
        **load_info,
        "device": str(device),
        "oracle_gold_candidates": bool(oracle_gold_candidates),
        "start_rf_to_target": float(calculate_norm_rf(start_tree, target_tree)),
        "num_boundaries": len(paths),
        "mean_rf_to_gold_next_orthant": (
            sum(row["rf_to_gold_next_orthant"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "max_rf_to_gold_next_orthant": (
            max(row["rf_to_gold_next_orthant"] for row in rows) if rows else None
        ),
        "all_boundaries_exact": all(row["rf_to_gold_next_orthant"] == 0.0 for row in rows),
        "mean_selected_recall_vs_gold_births": (
            sum(row["selected_recall_vs_gold_births"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "mean_selected_precision_vs_gold_births": (
            sum(row["selected_precision_vs_gold_births"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "mean_candidate_recall_vs_gold_births": (
            sum(row["candidate_recall_vs_gold_births"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "total_selected_birth_splits": sum(row["selected_birth_splits"] for row in rows),
        "total_required_birth_splits": sum(row["required_birth_splits"] for row in rows),
        "total_gold_birth_splits": sum(row["gold_count"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--oracle-gold-candidates", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint_path = args.checkpoint or _latest_sample_checkpoint(config)
    if not checkpoint_path:
        raise SystemExit("No checkpoint provided and no sample checkpoint found.")
    device = torch.device(args.device)
    payload = evaluate(
        config,
        checkpoint_path,
        device,
        oracle_gold_candidates=bool(args.oracle_gold_candidates),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
