"""Dataset and DataModule skeletons for PhylaFlow.

This version targets a common layout for Nexus alignments and MrBayes outputs:

data_root/
    nexus/                     # directory of source Nexus files (one per ID)
        <id>.nex | <id>.nexus
    runs/                      # directory containing MrBayes outputs per ID
        <id>/
            <id>_DNA.run1.t       # tree samples (we'll index .t files)
            <id>_DNA.run2.t
            ... other MrBayes files ...

"""

from __future__ import annotations

import os
import re
import json
import math
import itertools
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from utils.bhv_utils import (
    BHVEncoder,
    _split_multi_label_training_events,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_orthant_velocity,
    return_sampled_tree_boundary_decisions,
    return_tree_boundary_merge_paths,
)
import random
from model.treeTokenizer import TreeFeatureTokenizer, _worker_newick_parser
from utils.random_tree import Tree
from ete3 import Tree as EteTree
from utils.utils import get_possible_ids, remove_bit
from utils.bhv_movie import build_tree_from_splits


def _birthset_local_subset_for_components(
    split_mask: int,
    component_masks: List[int],
) -> Optional[int]:
    split_mask = int(split_mask)
    parent_mask = 0
    for component in component_masks:
        parent_mask |= int(component)
    if parent_mask == 0 or (split_mask & ~parent_mask):
        return None

    local_subset = 0
    for idx, component in enumerate(component_masks):
        component = int(component)
        overlap = component & split_mask
        if overlap and overlap != component:
            return None
        if overlap:
            local_subset |= 1 << int(idx)

    G = len(component_masks)
    subset_size = int(local_subset).bit_count()
    if 2 <= subset_size <= max(G - 1, 0):
        return int(local_subset)
    return None


def _birthset_boundary_training_event(boundary_newick: str, birth_splits) -> Optional[Dict[str, Any]]:
    try:
        groups = [
            tuple(int(component) for component in group)
            for group in get_structural_polytomy_groups_from_newick(boundary_newick)
        ]
    except Exception:
        return None
    if not groups:
        return None

    labels: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        if len(group) <= 2:
            continue
        for birth_split in birth_splits or []:
            local_subset = _birthset_local_subset_for_components(
                int(birth_split),
                list(group),
            )
            if local_subset is None:
                continue
            merge_indices = [
                idx
                for idx in range(len(group))
                if (int(local_subset) >> int(idx)) & 1
            ]
            if len(merge_indices) < 2:
                continue
            result_split = 0
            for idx in merge_indices:
                result_split |= int(group[idx])
            key = (int(result_split), tuple(group), tuple(merge_indices))
            if key in seen:
                continue
            seen.add(key)
            labels.append(
                {
                    "result_split": int(result_split),
                    "parent_split": int(np.bitwise_or.reduce(group)),
                    "components": [int(component) for component in group],
                    "merge_indices": [int(idx) for idx in merge_indices],
                }
            )

    if not labels:
        return None
    return {
        "newick": str(boundary_newick),
        "labels": labels,
        "stop_after_merge": True,
    }


def _birthset_full_mask_for_num_leaves(num_leaves: int) -> int:
    biological_bits = max(int(num_leaves) - 1, 0)
    return (1 << biological_bits) - 1 if biological_bits > 0 else 0


def _birthset_local_subset_size(local_subset: int) -> int:
    return int(local_subset).bit_count()


def _birthset_local_subset_to_split(
    local_subset: int,
    component_masks: List[int],
) -> int:
    split = 0
    for idx, component in enumerate(component_masks):
        if (int(local_subset) >> int(idx)) & 1:
            split |= int(component)
    return int(split)


def _birthset_valid_local_subset(local_subset: int, num_components: int) -> bool:
    size = _birthset_local_subset_size(local_subset)
    return 2 <= size <= max(int(num_components) - 1, 0)


def _birthset_valid_rooted_split(split_mask: int, full_mask: int) -> bool:
    split_mask = int(split_mask) & int(full_mask)
    return 1 < split_mask.bit_count() < int(full_mask).bit_count()


def _birthset_canonical_unrooted_split(split_mask: int, full_mask: int) -> int:
    full_mask = int(full_mask)
    split_mask = int(split_mask) & full_mask
    complement = full_mask ^ split_mask
    return min(int(split_mask), int(complement))


def _birthset_candidate_record(
    local_subset: int,
    component_masks: List[int],
    source: str,
) -> Dict[str, Any]:
    local_subset = int(local_subset)
    return {
        "local_subset": local_subset,
        "split_mask": _birthset_local_subset_to_split(local_subset, component_masks),
        "source": str(source),
        "size": _birthset_local_subset_size(local_subset),
    }


def _birthset_add_candidate_record(
    candidates_by_subset: Dict[int, Dict[str, Any]],
    local_subset: int,
    component_masks: List[int],
    full_mask: int,
    source: str,
    *,
    max_candidates: int,
    force: bool = False,
) -> bool:
    local_subset = int(local_subset)
    if not _birthset_valid_local_subset(local_subset, len(component_masks)):
        return False
    split_mask = _birthset_local_subset_to_split(local_subset, component_masks)
    if not _birthset_valid_rooted_split(split_mask, full_mask):
        return False
    if not force and len(candidates_by_subset) >= int(max_candidates):
        return False
    existing = candidates_by_subset.get(local_subset)
    if existing is not None and existing.get("source") == "gold":
        return True
    candidates_by_subset[local_subset] = _birthset_candidate_record(
        local_subset,
        component_masks,
        source,
    )
    return True


def _birthset_num_required_splits_for_group(
    num_components: int,
    component_masks: List[int],
    full_mask: int,
) -> int:
    parent_mask = 0
    for component in component_masks:
        parent_mask |= int(component)
    is_root_polytomy = bool(
        int(full_mask) != 0 and (int(parent_mask) & int(full_mask)) == int(full_mask)
    )
    offset = 3 if is_root_polytomy else 2
    return max(int(num_components) - offset, 0)


def _birthset_subset_inside_any_gold(
    local_subset: int,
    gold_local_subsets: List[int],
) -> bool:
    local_subset = int(local_subset)
    for gold_subset in gold_local_subsets or []:
        gold_subset = int(gold_subset)
        if gold_subset and (local_subset & ~gold_subset) == 0:
            return True
    return False


def _birthset_constructive_pair_targets(
    gold_local_subsets: List[int],
    num_components: int,
) -> set[int]:
    gold = [
        int(mask)
        for mask in sorted({int(mask) for mask in gold_local_subsets or []})
        if _birthset_valid_local_subset(int(mask), int(num_components))
    ]
    if not gold:
        return set()

    direct_pair_targets = {
        int(mask)
        for mask in gold
        if _birthset_local_subset_size(int(mask)) == 2
    }
    if direct_pair_targets:
        return direct_pair_targets

    min_size = min(_birthset_local_subset_size(mask) for mask in gold)
    targets = set()
    for gold_subset in gold:
        if _birthset_local_subset_size(gold_subset) != int(min_size):
            continue
        members = [
            idx
            for idx in range(int(num_components))
            if (int(gold_subset) >> int(idx)) & 1
        ]
        for left_pos, left_idx in enumerate(members):
            for right_idx in members[left_pos + 1 :]:
                pair = (1 << int(left_idx)) | (1 << int(right_idx))
                if _birthset_valid_local_subset(pair, int(num_components)):
                    targets.add(int(pair))
    return targets


def _birthset_num_leaves_for_precompute(
    sample: Dict[str, Any],
    component_masks: List[int],
    gold_splits: List[int],
) -> int:
    if sample.get("num_leaves") is not None:
        try:
            return int(sample["num_leaves"])
        except Exception:
            pass
    max_bit = 0
    for mask in list(component_masks or []) + list(gold_splits or []):
        max_bit = max(max_bit, int(mask).bit_length())
    if max_bit > 0:
        return int(max_bit + 1)
    newick = sample.get("newick") or sample.get("newick_tree")
    if newick:
        try:
            return int(Tree(str(newick)).n_leaves)
        except Exception:
            pass
    return 0


def _birthset_precompute_group(
    sample: Dict[str, Any],
    component_masks: List[int],
    group_targets: List[Tuple[int, int]],
    *,
    use_small_polytomy_enumeration: bool,
    use_static_pair_triple_candidates: bool,
    max_enum_components: int,
    max_candidates_per_polytomy: int,
    proposal_pair_target_mode: str,
    proposal_max_expansion_examples: int,
    proposal_max_order_seed_pairs: int,
    proposal_train_topk: bool,
) -> Dict[str, Any]:
    component_masks = [int(mask) for mask in component_masks]
    G = len(component_masks)
    gold_splits = sorted({int(split) for split, _ in group_targets})
    gold_local_subsets = sorted(
        {
            int(mask)
            for _, mask in group_targets
            if _birthset_valid_local_subset(int(mask), G)
        }
    )
    num_leaves = _birthset_num_leaves_for_precompute(
        sample,
        component_masks,
        gold_splits,
    )
    full_mask = _birthset_full_mask_for_num_leaves(num_leaves)
    max_bit_length = max(
        [int(full_mask).bit_length()]
        + [int(mask).bit_length() for mask in component_masks]
        + [int(mask).bit_length() for mask in gold_splits]
    )
    if max_bit_length > int(full_mask).bit_length():
        full_mask = (1 << int(max_bit_length)) - 1

    candidates_by_subset: Dict[int, Dict[str, Any]] = {}
    if use_small_polytomy_enumeration and G <= int(max_enum_components):
        for size in range(2, G):
            for combo in itertools.combinations(range(G), size):
                local_subset = 0
                for idx in combo:
                    local_subset |= 1 << int(idx)
                _birthset_add_candidate_record(
                    candidates_by_subset,
                    local_subset,
                    component_masks,
                    full_mask,
                    "enum",
                    max_candidates=max_candidates_per_polytomy,
                )
                if len(candidates_by_subset) >= int(max_candidates_per_polytomy):
                    break
            if len(candidates_by_subset) >= int(max_candidates_per_polytomy):
                break

    if use_static_pair_triple_candidates and G > int(max_enum_components):
        for left_idx in range(G):
            for right_idx in range(left_idx + 1, G):
                subset = (1 << int(left_idx)) | (1 << int(right_idx))
                _birthset_add_candidate_record(
                    candidates_by_subset,
                    subset,
                    component_masks,
                    full_mask,
                    "pair_static",
                    max_candidates=max_candidates_per_polytomy,
                )
                if len(candidates_by_subset) >= int(max_candidates_per_polytomy):
                    break
            if len(candidates_by_subset) >= int(max_candidates_per_polytomy):
                break
        if len(candidates_by_subset) < int(max_candidates_per_polytomy):
            for combo in itertools.combinations(range(G), 3):
                subset = 0
                for idx in combo:
                    subset |= 1 << int(idx)
                _birthset_add_candidate_record(
                    candidates_by_subset,
                    subset,
                    component_masks,
                    full_mask,
                    "triple_static",
                    max_candidates=max_candidates_per_polytomy,
                )
                if len(candidates_by_subset) >= int(max_candidates_per_polytomy):
                    break

    pre_gold_candidate_splits = {
        _birthset_canonical_unrooted_split(item["split_mask"], full_mask)
        for item in candidates_by_subset.values()
    }
    pre_gold_candidate_local_subsets = {
        int(item["local_subset"]) for item in candidates_by_subset.values()
    }
    pre_gold_target_count = len(
        {("split", int(split)) for split in gold_splits}
        | {("local", int(local_subset)) for local_subset in gold_local_subsets}
    )
    pre_gold_target_hits = len(
        {
            ("split", int(split))
            for split in gold_splits
            if _birthset_canonical_unrooted_split(split, full_mask)
            in pre_gold_candidate_splits
        }
        | {
            ("local", int(local_subset))
            for local_subset in gold_local_subsets
            if int(local_subset) in pre_gold_candidate_local_subsets
        }
    )

    gold_mismatches = 0
    for split in gold_splits:
        local_subset = _birthset_local_subset_for_components(
            split,
            component_masks,
        )
        if local_subset is None:
            gold_mismatches += 1
            continue
        _birthset_add_candidate_record(
            candidates_by_subset,
            local_subset,
            component_masks,
            full_mask,
            "gold",
            max_candidates=max_candidates_per_polytomy,
            force=True,
        )
    for local_subset in gold_local_subsets:
        added = _birthset_add_candidate_record(
            candidates_by_subset,
            local_subset,
            component_masks,
            full_mask,
            "gold",
            max_candidates=max_candidates_per_polytomy,
            force=True,
        )
        if not added:
            gold_mismatches += 1

    candidates = list(candidates_by_subset.values())
    source_rank = {"gold": 0, "pair_static": 1, "triple_static": 2, "bank": 3, "enum": 4}
    candidates.sort(
        key=lambda item: (
            source_rank.get(item["source"], 9),
            int(item["size"]),
            int(item["split_mask"]),
        )
    )
    gold_split_keys = {
        _birthset_canonical_unrooted_split(split, full_mask)
        for split in gold_splits
    }
    candidate_labels = [
        1.0
        if (
            _birthset_canonical_unrooted_split(item["split_mask"], full_mask)
            in gold_split_keys
            or int(item["local_subset"]) in gold_local_subsets
        )
        else 0.0
        for item in candidates
    ]

    pair_subsets = []
    for left_idx in range(G):
        for right_idx in range(left_idx + 1, G):
            subset = (1 << int(left_idx)) | (1 << int(right_idx))
            if _birthset_valid_local_subset(subset, G):
                pair_subsets.append(int(subset))
    constructive_pair_targets = _birthset_constructive_pair_targets(
        gold_local_subsets,
        G,
    )
    strict_pair_targets = (
        constructive_pair_targets
        if str(proposal_pair_target_mode) == "strict_minimal"
        else None
    )
    pair_labels = [
        1.0
        if (
            int(subset) in strict_pair_targets
            if strict_pair_targets is not None
            else _birthset_subset_inside_any_gold(subset, gold_local_subsets)
        )
        else 0.0
        for subset in pair_subsets
    ]

    expansion_subsets = None
    expansion_labels = None
    if not bool(proposal_train_topk):
        expansion_subsets = []
        for combo in itertools.combinations(range(G), 3):
            subset = 0
            for idx in combo:
                subset |= 1 << int(idx)
            expansion_subsets.append(int(subset))
        if expansion_subsets:
            positives = [
                subset
                for subset in expansion_subsets
                if _birthset_subset_inside_any_gold(subset, gold_local_subsets)
            ]
            negatives = [
                subset
                for subset in expansion_subsets
                if not _birthset_subset_inside_any_gold(subset, gold_local_subsets)
            ]
            cap = int(proposal_max_expansion_examples)
            if len(positives) + len(negatives) > cap:
                expansion_subsets = positives + negatives[: max(0, cap - len(positives))]
            else:
                expansion_subsets = positives + negatives
            expansion_labels = [
                1.0
                if _birthset_subset_inside_any_gold(subset, gold_local_subsets)
                else 0.0
                for subset in expansion_subsets
            ]

    if strict_pair_targets is not None:
        positive_pair_subsets = [
            subset for subset in pair_subsets if int(subset) in strict_pair_targets
        ]
    else:
        positive_pair_subsets = [
            subset
            for subset in pair_subsets
            if _birthset_subset_inside_any_gold(subset, gold_local_subsets)
        ]
    positive_pair_subsets.sort(
        key=lambda subset: (
            min(
                (
                    _birthset_local_subset_size(gold_subset)
                    for gold_subset in gold_local_subsets
                    if (int(subset) & ~int(gold_subset)) == 0
                ),
                default=G + 1,
            ),
            int(subset),
        )
    )
    positive_pair_subsets = positive_pair_subsets[
        : int(proposal_max_order_seed_pairs)
    ]
    order_candidate_subsets = []
    order_slices = []
    for pair_subset in positive_pair_subsets:
        pair_subset = int(pair_subset)
        target_ranks = []
        start = len(order_candidate_subsets)
        for idx in range(G):
            if (pair_subset >> int(idx)) & 1:
                continue
            candidate_subset = int(pair_subset | (1 << int(idx)))
            if not _birthset_valid_local_subset(candidate_subset, G):
                continue
            containing_sizes = [
                _birthset_local_subset_size(gold_subset)
                for gold_subset in gold_local_subsets
                if (pair_subset & ~int(gold_subset)) == 0
                and (candidate_subset & ~int(gold_subset)) == 0
            ]
            rank = min(containing_sizes) if containing_sizes else G + 1
            order_candidate_subsets.append(candidate_subset)
            target_ranks.append(float(rank))
        end = len(order_candidate_subsets)
        if end - start >= 2:
            order_slices.append((start, end, target_ranks))
        else:
            del order_candidate_subsets[start:end]

    return {
        "components": tuple(component_masks),
        "gold_splits": gold_splits,
        "gold_local_subsets": gold_local_subsets,
        "candidate_info": {
            "candidates": candidates,
            "candidate_labels": candidate_labels,
            "full_mask": int(full_mask),
            "gold_mismatches": int(gold_mismatches),
            "pre_gold_target_count": int(pre_gold_target_count),
            "pre_gold_target_hits": int(pre_gold_target_hits),
            "static_pair_triple_candidates": bool(use_static_pair_triple_candidates),
        },
        "proposal": {
            "pair_subsets": pair_subsets,
            "pair_labels": pair_labels,
            "strict_pair_targets": sorted(int(x) for x in strict_pair_targets or []),
            "gold_local_subsets": gold_local_subsets,
            "expansion_subsets": expansion_subsets,
            "expansion_labels": expansion_labels,
            "order_candidate_subsets": order_candidate_subsets,
            "order_slices": order_slices,
            "positive_pair_count": int(len(positive_pair_subsets)),
            "train_topk_dynamic": bool(proposal_train_topk),
        },
        "required_splits": int(
            _birthset_num_required_splits_for_group(G, component_masks, full_mask)
        ),
    }


def _birthset_precompute_sample(
    sample: Dict[str, Any],
    *,
    use_small_polytomy_enumeration: bool,
    use_static_pair_triple_candidates: bool,
    max_enum_components: int,
    max_candidates_per_polytomy: int,
    proposal_pair_target_mode: str,
    proposal_max_expansion_examples: int,
    proposal_max_order_seed_pairs: int,
    proposal_train_topk: bool,
) -> Optional[Dict[str, Any]]:
    labels = sample.get("labels") or sample.get("autoregressive_labels")
    if not labels:
        return None
    group_targets: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    for label in labels:
        if not isinstance(label, dict):
            continue
        components = tuple(int(component) for component in label.get("components", []))
        if not components:
            continue
        merge_indices = [int(idx) for idx in label.get("merge_indices", [])]
        local_subset = 0
        for idx in merge_indices:
            local_subset |= 1 << int(idx)
        result_split = int(label.get("result_split", 0))
        group_targets.setdefault(components, []).append((result_split, local_subset))
    if not group_targets:
        return None
    groups_by_components = {}
    for components, targets in group_targets.items():
        groups_by_components[components] = _birthset_precompute_group(
            sample,
            list(components),
            targets,
            use_small_polytomy_enumeration=use_small_polytomy_enumeration,
            use_static_pair_triple_candidates=use_static_pair_triple_candidates,
            max_enum_components=max_enum_components,
            max_candidates_per_polytomy=max_candidates_per_polytomy,
            proposal_pair_target_mode=proposal_pair_target_mode,
            proposal_max_expansion_examples=proposal_max_expansion_examples,
            proposal_max_order_seed_pairs=proposal_max_order_seed_pairs,
            proposal_train_topk=proposal_train_topk,
        )
    return {
        "groups_by_components": groups_by_components,
        "num_groups": int(len(groups_by_components)),
    }


def _detach_tensors(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tensors(item) for item in value)
    if isinstance(value, list):
        return [_detach_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tensors(item) for key, item in value.items()}
    return value


def _load_full_path_control_extra_velocity_samples(json_path: Optional[str]) -> List[Dict[str, Any]]:
    if not json_path:
        return []
    payload = json.loads(Path(json_path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("samples", [])
    if not isinstance(payload, list):
        raise ValueError(
            "overfit_full_path_control_extra_velocity_samples_json_path must point "
            "to a JSON list or an object with a 'samples' list."
        )

    samples: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not item.get("newick_tree"):
            continue
        velocity = item.get("velocity") or {}
        samples.append(
            {
                "path_index": int(item.get("path_index", item.get("phase_idx", 0))),
                "phase_idx": int(item.get("phase_idx", item.get("path_index", 0))),
                "newick_tree": str(item["newick_tree"]),
                "target_tree": str(item.get("target_tree", "")),
                "velocity": {
                    int(k): float(v) for k, v in dict(velocity).items()
                },
                "velocity_next_boundary_tree": (
                    None
                    if item.get("velocity_next_boundary_tree") in {None, ""}
                    else str(item.get("velocity_next_boundary_tree"))
                ),
                "timepoint": float(item.get("timepoint", item.get("phase_idx", 0.0))),
                "num_leaves": int(
                    item.get("num_leaves", int(Tree(str(item["newick_tree"])).n_leaves))
                ),
                "anchor_family": (
                    None
                    if item.get("anchor_family") in {None, ""}
                    else str(item.get("anchor_family"))
                ),
                "source_checkpoint": (
                    None
                    if item.get("source_checkpoint") in {None, ""}
                    else str(item.get("source_checkpoint"))
                ),
                "bank_group_key": (
                    None
                    if item.get("bank_group_key") in {None, ""}
                    else str(item.get("bank_group_key"))
                ),
            }
        )
    return samples


def _coerce_dataset_path_map(raw_value: Optional[Any]) -> Dict[str, str]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return {
            str(key).upper(): str(value)
            for key, value in raw_value.items()
            if value not in {None, ""}
        }
    if isinstance(raw_value, list):
        result: Dict[str, str] = {}
        for item in raw_value:
            if not isinstance(item, dict):
                continue
            dataset_id = item.get("dataset_id") or item.get("id")
            path = item.get("path") or item.get("json_path")
            if dataset_id and path:
                result[str(dataset_id).upper()] = str(path)
        return result
    raise ValueError(
        "overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id "
        "must be a mapping or a list of {dataset_id, path} entries."
    )


def _index_velocity_samples_by_group_key(
    samples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    indexed: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        group_key = sample.get("bank_group_key")
        if group_key is None:
            continue
        indexed.setdefault(str(group_key), []).append(sample)
    return indexed


def _load_json_or_jsonl_records(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    records: List[Dict[str, Any]] = []
    bank_path = Path(path)
    if not bank_path.is_file():
        raise ValueError(f"Bank index path is not a file: {bank_path}")
    if bank_path.suffix.lower() == ".jsonl":
        with bank_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    records.append(row)
    else:
        payload = json.loads(bank_path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("cases", payload.get("records", []))
        if not isinstance(payload, list):
            raise ValueError(
                f"Bank index {bank_path} must contain a JSON list or JSONL rows."
            )
        records = [dict(row) for row in payload if isinstance(row, dict)]
    return records


def _tree_from_json_payload(payload: Dict[str, Any], *, role: str) -> str:
    keys = (
        ("start_tree", "final_tree", "tree")
        if role == "start"
        else ("target_tree", "final_tree", "tree", "start_tree")
    )
    tree = None
    for key in keys:
        if payload.get(key):
            tree = payload.get(key)
            break
    if tree is None:
        raise ValueError(f"Topology-stream {role} JSON does not contain a tree.")
    tree = str(tree).strip()
    return tree if tree.endswith(";") else f"{tree};"


def _derive_anchor_path_from_start_path(start_path: str) -> Optional[str]:
    start = str(start_path)
    candidates = []
    if start.endswith("_start.json"):
        candidates.append(start[: -len("_start.json")] + "_velocity_anchors.json")
    if start.endswith("start.json"):
        candidates.append(start[: -len("start.json")] + "velocity_anchors.json")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class SizeDetector:
    def __init__(self, max_aa=None):
        self.max_aa = max_aa

    def update_max_aa(self, new_max_aa):
        self.max_aa = new_max_aa


def resolve_training_target_tree_for_prefix(
    start_tree_newick: str,
    target_tree_newick: str,
    prefix_k: int,
) -> str:
    if int(prefix_k) < 0:
        return target_tree_newick

    boundary_paths = return_tree_boundary_merge_paths(
        start_tree_newick,
        target_tree_newick,
        legacy_training_semantics=False,
    )
    if not boundary_paths:
        return target_tree_newick

    prefix_idx = min(int(prefix_k), len(boundary_paths) - 1)
    return boundary_paths[prefix_idx]["end_newick"]


def resolve_training_target_tree_for_event_prefix(
    start_tree_newick: str,
    target_tree_newick: str,
    event_prefix_count: int,
) -> str:
    event_prefix_count = int(event_prefix_count)
    if event_prefix_count < 0:
        return target_tree_newick
    if event_prefix_count == 0:
        return start_tree_newick

    boundary_paths = return_tree_boundary_merge_paths(
        start_tree_newick,
        target_tree_newick,
        legacy_training_semantics=False,
    )
    if not boundary_paths:
        return target_tree_newick

    remaining_events = event_prefix_count
    current_tree = start_tree_newick
    for path in boundary_paths:
        events = path["events"]
        if remaining_events == 0:
            return current_tree
        if remaining_events < len(events):
            return path["events"][remaining_events]["newick"]
        if remaining_events == len(events):
            return path["end_newick"]
        remaining_events -= len(events)
        current_tree = path["end_newick"]

    return target_tree_newick


def _remap_tree_leaf_names_to_match_reference(
    tree_newick: str,
    reference_tree_newick: str,
) -> str:
    tree = EteTree(tree_newick, format=1)
    reference_tree = EteTree(reference_tree_newick, format=1)

    tree_leaves = sorted((leaf.name for leaf in tree.get_leaves()), key=lambda x: int(x))
    reference_leaves = sorted(
        (leaf.name for leaf in reference_tree.get_leaves()),
        key=lambda x: int(x),
    )

    if tree_leaves == reference_leaves:
        return tree_newick

    if len(tree_leaves) != len(reference_leaves):
        raise ValueError(
            "Cannot remap tree leaf names: leaf counts differ between tree and reference."
        )

    remap = {src: dst for src, dst in zip(tree_leaves, reference_leaves)}
    for leaf in tree.get_leaves():
        if leaf.name not in remap:
            raise ValueError(f"Leaf {leaf.name} missing from remap dictionary.")
        leaf.name = remap[leaf.name]

    return tree.write(format=1)


def _leaf_count_from_newick(tree_newick: str) -> int:
    return len(EteTree(tree_newick, format=1).get_leaves())


def _numeric_name_sort_key(name: Any) -> Tuple[int, Any]:
    name = str(name)
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _canonicalize_ete_child_order(tree: EteTree) -> None:
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.add_feature("sort_val", _numeric_name_sort_key(node.name))
        else:
            child_values = [
                getattr(child, "sort_val", (1, ""))
                for child in node.children
            ]
            node.add_feature(
                "sort_val",
                min(child_values) if child_values else (1, ""),
            )
    tree.sort_descendants(attr="sort_val")


def _normalize_topology_stream_tree_pair(
    start_obj: EteTree,
    target_obj: EteTree,
) -> Tuple[str, str, List[str], Dict[str, str]]:
    """Put topology-stream start/target trees in the dataset leaf space.

    The BHV wrapper adds a dummy root at the serialized Newick root, so two
    equivalent unrooted trees can become different supervised problems if their
    Newick roots differ after pruning.  Normalize both trees to the same sorted
    leaf order, rename leaves to 0..N-1, and root them on leaf 0 before label
    generation.
    """
    original_leaf_order = [
        str(leaf.name)
        for leaf in sorted(
            target_obj.get_leaves(),
            key=lambda leaf: _numeric_name_sort_key(leaf.name),
        )
    ]
    start_leaf_names = {str(leaf.name) for leaf in start_obj.get_leaves()}
    target_leaf_names = {str(leaf.name) for leaf in target_obj.get_leaves()}
    expected_leaf_names = set(original_leaf_order)
    if start_leaf_names != expected_leaf_names or target_leaf_names != expected_leaf_names:
        raise ValueError(
            "Topology-stream start/target leaf sets differ after pruning. "
            f"start_only={sorted(start_leaf_names - expected_leaf_names)[:5]}, "
            f"target_only={sorted(target_leaf_names - start_leaf_names)[:5]}"
        )

    leaf_to_index = {
        original_name: str(idx)
        for idx, original_name in enumerate(original_leaf_order)
    }

    def _rewrite(tree: EteTree) -> str:
        for leaf in tree.get_leaves():
            leaf.name = leaf_to_index[str(leaf.name)]
        if len(original_leaf_order) > 2:
            leaves_by_name = {str(leaf.name): leaf for leaf in tree.get_leaves()}
            outgroup = leaves_by_name.get("0")
            if outgroup is not None:
                tree.set_outgroup(outgroup)
        _canonicalize_ete_child_order(tree)
        return tree.write(format=1)

    return (
        _rewrite(start_obj),
        _rewrite(target_obj),
        original_leaf_order,
        leaf_to_index,
    )


def _coerce_id_list(raw_ids: Optional[Any]) -> List[str]:
    if raw_ids is None:
        return []
    if isinstance(raw_ids, str):
        stripped = raw_ids.strip()
        if not stripped:
            return []
        range_match = re.fullmatch(r"DS(\d+)\s*-\s*(?:DS)?(\d+)", stripped, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            return [f"DS{i}" for i in range(start, end + step, step)]
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(raw_ids, (list, tuple, set)):
        return [str(item).strip() for item in raw_ids if str(item).strip()]
    return [str(raw_ids).strip()]


def _normalize_bank_group_key(
    explicit_group_key: Optional[str],
    selected_original_labels: Optional[List[str]],
    tree_newick: str,
) -> str:
    if explicit_group_key is not None and str(explicit_group_key).strip():
        return str(explicit_group_key).strip()
    if selected_original_labels:
        return "labels:" + ",".join(str(label) for label in selected_original_labels)
    return f"n{_leaf_count_from_newick(tree_newick)}"


class TreeDataset(Dataset):
    """Dataset mapping IDs to Nexus sequences and MrBayes tree files.

    Layout assumptions (configurable):
    - nexus_root contains files: <id>.nex or <id>.nexus
    - mrbayes_root contains subdirs per <id> with one or more .t files
      (e.g., <id>_DNA.run1.t, <id>_DNA.run2.t)

    No parsing is performed here by default; this class only builds an index
    and returns paths with placeholders. Fill in parse methods as needed.

    Args:
        nexus_root: Directory with Nexus source files.
        mrbayes_root: Directory with MrBayes output folders.
        prefer_run: Which run's .t to prefer ("run1", "run2", "any").
        transform: Optional callable applied to each sample dict.
        cache: If True, eagerly parse and cache sequences/trees (TODO).
    """

    def __init__(
        self,
        nexus_root: str,
        mrbayes_root: str,
        filter_ids: Optional[List[str]] = None,
        validation=False,
        sanity_check: bool = False,
        random_sanity_check: bool = False,
        overfit_velocity_zero: bool = False,
        overfit_velocity_event_states: bool = False,
        overfit_velocity_orthant_start_states: bool = False,
        overfit_velocity_explicit_boundary_end_states: bool = False,
        overfit_velocity_fixed_timepoints: Optional[List[float]] = None,
        overfit_velocity_explicit_boundary_label_scale_mode: str = "local",
        overfit_boundary_prefix_k: int = -1,
        overfit_start_boundary_prefix_k: int = -1,
        overfit_event_prefix_count: int = -1,
        overfit_event_horizon: int = 1,
        overfit_fixed_pair: bool = False,
        overfit_fixed_pair_start_tree_newick: Optional[str] = None,
        overfit_fixed_pair_start_tree_json_path: Optional[str] = None,
        overfit_fixed_pair_start_tree_json_paths: Optional[List[str]] = None,
        overfit_fixed_pair_start_tree_json_dir: Optional[str] = None,
        overfit_fixed_pair_target_tree_newick: Optional[str] = None,
        overfit_fixed_pair_target_tree_json_path: Optional[str] = None,
        overfit_fixed_pair_target_tree_json_paths: Optional[List[str]] = None,
        overfit_fixed_pair_target_tree_json_dir: Optional[str] = None,
        overfit_fixed_pair_joint_bank_jsonl_path: Optional[str] = None,
        overfit_split_multi_subset_events: bool = False,
        overfit_full_path_control_mode: bool = False,
        overfit_full_path_control_birthset_boundary_labels: bool = False,
        overfit_full_path_control_seed: int = 42,
        overfit_full_path_control_use_discrete_phase_time: bool = False,
        overfit_full_path_control_terminal_label_mode: str = "phase_start",
        overfit_full_path_control_terminal_include_ar_states: bool = False,
        overfit_full_path_control_terminal_include_target_one_split_off: bool = False,
        overfit_full_path_control_extra_velocity_samples_json_path: Optional[str] = None,
        overfit_full_path_control_extra_velocity_samples_json_paths: Optional[List[str]] = None,
        overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id: Optional[Any] = None,
        full_path_preparse_structural_trees: bool = False,
        overfit_oracle_prefix_start_prob: float = 0.0,
        overfit_oracle_prefix_max_fraction: float = 0.5,
        overfit_fixed_pair_group_by_json_metadata: bool = False,
        overfit_fixed_pair_reference_tree_from_target_bank: bool = False,
        overfit_virtual_epoch_size: Optional[int] = None,
        overfit_fixed_pair_cache_virtual_index_selection: bool = False,
        topology_stream_index_jsonl_path: Optional[str] = None,
        topology_stream_index_max_cases: int = 0,
        topology_stream_index_min_num_leaves: int = 0,
        topology_stream_index_max_num_leaves: int = 0,
        topology_stream_subset_num_leaves: int = 0,
        posterior_subset_num_leaves: int = 0,
        posterior_subset_max_input_tokens: int = 0,
        sample_metrics_iterate_dataset_indices: bool = False,
        topology_stream_use_real_sequences: bool = False,
        topology_stream_max_input_tokens: int = 0,
        posterior_trprobs_root: Optional[str] = None,
        posterior_dataset_id: Optional[Any] = None,
        posterior_dataset_ids: Optional[Any] = None,
        use_random_sequence_distribution: bool = False,
        random_distribution_sequence_length: int = 256,
        random_distribution_sequence_seed: int = 0,
        random_distribution_alphabet: str = "ACGT",
        trprobs_sample_count_per_file: int = 1000,
    ) -> None:
        self.nexus_root = nexus_root
        self.mrbayes_root = mrbayes_root
        self.filter_ids = filter_ids
        self.validation = validation
        self.posterior_trprobs_root = (
            str(posterior_trprobs_root) if posterior_trprobs_root else None
        )
        self.topology_stream_index_jsonl_path = (
            str(topology_stream_index_jsonl_path)
            if topology_stream_index_jsonl_path
            else None
        )
        self.topology_stream_index_max_cases = max(
            0,
            int(topology_stream_index_max_cases or 0),
        )
        self.topology_stream_index_min_num_leaves = max(
            0,
            int(topology_stream_index_min_num_leaves or 0),
        )
        self.topology_stream_index_max_num_leaves = max(
            0,
            int(topology_stream_index_max_num_leaves or 0),
        )
        self.topology_stream_subset_num_leaves = max(
            0,
            int(topology_stream_subset_num_leaves or 0),
        )
        self.posterior_subset_num_leaves = max(
            0,
            int(posterior_subset_num_leaves or 0),
        )
        self.posterior_subset_max_input_tokens = max(
            0,
            int(posterior_subset_max_input_tokens or 0),
        )
        self.sample_metrics_iterate_dataset_indices = bool(
            sample_metrics_iterate_dataset_indices
        )
        self.topology_stream_use_real_sequences = bool(
            topology_stream_use_real_sequences
        )
        self.topology_stream_max_input_tokens = max(
            0,
            int(topology_stream_max_input_tokens or 0),
        )
        posterior_ids = _coerce_id_list(posterior_dataset_ids)
        if not posterior_ids:
            posterior_ids = _coerce_id_list(posterior_dataset_id)
        self.posterior_dataset_ids = posterior_ids
        self.use_random_sequence_distribution = bool(
            use_random_sequence_distribution
            or self.posterior_trprobs_root
            or (
                self.topology_stream_index_jsonl_path
                and not self.topology_stream_use_real_sequences
            )
        )
        self.random_distribution_sequence_length = max(
            1, int(random_distribution_sequence_length)
        )
        self.random_distribution_sequence_seed = int(random_distribution_sequence_seed)
        alphabet = str(random_distribution_alphabet or "ACGT").strip()
        self.random_distribution_alphabet = alphabet or "ACGT"
        self.trprobs_sample_count_per_file = max(0, int(trprobs_sample_count_per_file))
        self.overfit_velocity_zero = overfit_velocity_zero
        self.overfit_velocity_event_states = bool(overfit_velocity_event_states)
        self.overfit_velocity_orthant_start_states = bool(
            overfit_velocity_orthant_start_states
        )
        self.overfit_velocity_explicit_boundary_end_states = bool(
            overfit_velocity_explicit_boundary_end_states
        )
        self.overfit_velocity_fixed_timepoints = (
            [float(t) for t in overfit_velocity_fixed_timepoints]
            if overfit_velocity_fixed_timepoints
            else None
        )
        label_scale_mode = str(overfit_velocity_explicit_boundary_label_scale_mode)
        if label_scale_mode not in {"local", "remaining"}:
            raise ValueError(
                "overfit_velocity_explicit_boundary_label_scale_mode must be "
                f"'local' or 'remaining', got {label_scale_mode!r}."
            )
        self.overfit_velocity_explicit_boundary_label_scale_mode = label_scale_mode
        self.overfit_boundary_prefix_k = int(overfit_boundary_prefix_k)
        self.overfit_start_boundary_prefix_k = int(overfit_start_boundary_prefix_k)
        self.overfit_event_prefix_count = int(overfit_event_prefix_count)
        self.overfit_event_horizon = max(1, int(overfit_event_horizon))
        self.overfit_fixed_pair = bool(overfit_fixed_pair)
        self.overfit_full_path_control_mode = bool(overfit_full_path_control_mode)
        self.overfit_full_path_control_seed = int(overfit_full_path_control_seed)
        self.overfit_full_path_control_use_discrete_phase_time = bool(
            overfit_full_path_control_use_discrete_phase_time
        )
        terminal_label_mode = str(overfit_full_path_control_terminal_label_mode).lower()
        if terminal_label_mode not in {"phase_start", "target"}:
            raise ValueError(
                "overfit_full_path_control_terminal_label_mode must be "
                f"'phase_start' or 'target', got {terminal_label_mode!r}."
            )
        self.overfit_full_path_control_terminal_label_mode = terminal_label_mode
        self.overfit_full_path_control_terminal_include_ar_states = bool(
            overfit_full_path_control_terminal_include_ar_states
        )
        self.overfit_full_path_control_terminal_include_target_one_split_off = bool(
            overfit_full_path_control_terminal_include_target_one_split_off
        )
        self.full_path_preparse_structural_trees = bool(
            full_path_preparse_structural_trees
        )
        self.overfit_full_path_control_extra_velocity_samples = (
            _load_full_path_control_extra_velocity_samples(
                overfit_full_path_control_extra_velocity_samples_json_path
            )
        )
        for extra_path in overfit_full_path_control_extra_velocity_samples_json_paths or []:
            self.overfit_full_path_control_extra_velocity_samples.extend(
                _load_full_path_control_extra_velocity_samples(str(extra_path))
            )
        self._overfit_full_path_control_extra_velocity_samples_by_group_key = (
            _index_velocity_samples_by_group_key(
                self.overfit_full_path_control_extra_velocity_samples
            )
        )
        self._overfit_full_path_control_extra_velocity_sample_paths_by_dataset_id = (
            _coerce_dataset_path_map(
                overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id
            )
        )
        self._overfit_full_path_control_loaded_extra_dataset_ids = set()
        self.overfit_oracle_prefix_start_prob = float(
            overfit_oracle_prefix_start_prob
        )
        self.overfit_oracle_prefix_max_fraction = float(
            overfit_oracle_prefix_max_fraction
        )
        self.overfit_fixed_pair_group_by_json_metadata = bool(
            overfit_fixed_pair_group_by_json_metadata
        )
        self.overfit_fixed_pair_reference_tree_from_target_bank = bool(
            overfit_fixed_pair_reference_tree_from_target_bank
        )
        self.overfit_fixed_pair_cache_virtual_index_selection = bool(
            overfit_fixed_pair_cache_virtual_index_selection
        )
        self.overfit_virtual_epoch_size = (
            int(overfit_virtual_epoch_size)
            if overfit_virtual_epoch_size is not None
            and int(overfit_virtual_epoch_size) > 0
            else None
        )
        override_start_tree = None
        override_start_tree_loaded_from_json = False
        override_start_tree_bank: List[Any] = []
        if overfit_fixed_pair_start_tree_newick:
            override_start_tree = str(overfit_fixed_pair_start_tree_newick)
        elif overfit_fixed_pair_start_tree_json_path:
            override_payload = json.loads(
                Path(overfit_fixed_pair_start_tree_json_path).read_text()
            )
            override_start_tree = str(
                override_payload.get("final_tree")
                or override_payload.get("start_tree")
                or override_payload.get("tree")
            )
            override_start_tree_loaded_from_json = True
            override_start_tree_bank.append(dict(override_payload))
        if overfit_fixed_pair_start_tree_json_paths:
            for raw_path in overfit_fixed_pair_start_tree_json_paths:
                override_payload = json.loads(Path(raw_path).read_text())
                override_tree = (
                    override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                    or override_payload.get("tree")
                )
                if override_tree:
                    override_start_tree_bank.append(dict(override_payload))
        if overfit_fixed_pair_start_tree_json_dir:
            for raw_path in sorted(Path(overfit_fixed_pair_start_tree_json_dir).glob("*.json")):
                override_payload = json.loads(raw_path.read_text())
                override_tree = (
                    override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                    or override_payload.get("tree")
                )
                if override_tree:
                    override_start_tree_bank.append(dict(override_payload))
        if (
            override_start_tree is not None
            and not override_start_tree_loaded_from_json
        ):
            override_start_tree_bank.append(str(override_start_tree))
        override_target_tree = None
        override_target_tree_loaded_from_json = False
        override_target_tree_bank: List[Any] = []
        if overfit_fixed_pair_target_tree_newick:
            override_target_tree = str(overfit_fixed_pair_target_tree_newick)
        elif overfit_fixed_pair_target_tree_json_path:
            override_payload = json.loads(
                Path(overfit_fixed_pair_target_tree_json_path).read_text()
            )
            override_target_tree = str(
                override_payload.get("target_tree")
                or override_payload.get("final_tree")
                or override_payload.get("start_tree")
                or override_payload.get("tree")
            )
            override_target_tree_loaded_from_json = True
            override_target_tree_bank.append(dict(override_payload))
        if overfit_fixed_pair_target_tree_json_paths:
            for raw_path in overfit_fixed_pair_target_tree_json_paths:
                override_payload = json.loads(Path(raw_path).read_text())
                override_tree = (
                    override_payload.get("target_tree")
                    or override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                    or override_payload.get("tree")
                )
                if override_tree:
                    override_target_tree_bank.append(dict(override_payload))
        if overfit_fixed_pair_target_tree_json_dir:
            for raw_path in sorted(Path(overfit_fixed_pair_target_tree_json_dir).glob("*.json")):
                override_payload = json.loads(raw_path.read_text())
                override_tree = (
                    override_payload.get("target_tree")
                    or override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                    or override_payload.get("tree")
                )
                if override_tree:
                    override_target_tree_bank.append(dict(override_payload))
        joint_bank_records = _load_json_or_jsonl_records(
            overfit_fixed_pair_joint_bank_jsonl_path
        )
        if joint_bank_records:
            override_start_tree_bank.extend(dict(row) for row in joint_bank_records)
            override_target_tree_bank.extend(dict(row) for row in joint_bank_records)
        if (
            override_target_tree is not None
            and not override_target_tree_loaded_from_json
        ):
            override_target_tree_bank.append(str(override_target_tree))
        self.overfit_fixed_pair_start_tree_newick = override_start_tree
        self.overfit_fixed_pair_start_tree_newick_bank: List[str] = []
        self.overfit_fixed_pair_start_tree_bank_items: List[Dict[str, Any]] = []
        self._overfit_fixed_pair_start_tree_groups: Dict[str, List[Dict[str, Any]]] = {}
        self.overfit_fixed_pair_target_tree_newick = override_target_tree
        self.overfit_fixed_pair_target_tree_newick_bank: List[str] = []
        self.overfit_fixed_pair_target_tree_bank_items: List[Dict[str, Any]] = []
        self._overfit_fixed_pair_target_tree_groups: Dict[str, List[Dict[str, Any]]] = {}
        self.overfit_split_multi_subset_events = bool(
            overfit_split_multi_subset_events
        )
        self.overfit_full_path_control_birthset_boundary_labels = bool(
            overfit_full_path_control_birthset_boundary_labels
        )
        self.size_detector = SizeDetector()
        # State tracker for adaptive batching (index, subtree_size, num_subtrees)
        # Default initialization
        self.chosen_tree = (0, 100, 1)
        self.name_to_seq = {}

        # Internal containers
        self._ids: List[str] = []  # populated by build_index()
        self._index: List[Dict[str, Any]] = []  # list of sample metadata dicts
        self._id_to_idx: Dict[str, int] = {}
        self._cached_overfit_pairs: Dict[int, Dict[str, Any]] = {}
        self._cached_overfit_pair_banks: Dict[int, List[Dict[str, Any]]] = {}
        self._cached_overfit_bank_pairs_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._cached_overfit_bank_selection_by_virtual_index: Dict[int, Dict[str, Any]] = {}
        self._cached_full_path_control_samples_by_key: Dict[
            Tuple[Any, ...],
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]],
        ] = {}
        self._frozen_full_path_control_selections: List[Dict[str, Any]] = []
        self._cached_posterior_trees_by_key: Dict[Tuple[Any, ...], List[str]] = {}
        self._cached_sequences_by_nexus_path: Dict[
            str,
            Tuple[Dict[str, str], List[str]],
        ] = {}
        self._cached_topology_stream_sequences_by_dataset_id: Dict[
            str,
            Tuple[Dict[str, str], List[str]],
        ] = {}
        self._cached_topology_stream_pair_by_key: Dict[
            Tuple[str, str, str],
            Dict[str, Any],
        ] = {}
        self.same_dataset_batch_size = 0
        self.same_dataset_batch_seed = 0
        self.random_tree = None
        self.sanity_check = sanity_check
        self.random_sanity_check = random_sanity_check

        if self.sanity_check and self.random_sanity_check:
            raise Exception("Cannot have both sanity_check and random_sanity_check enabled!")

        # Build index immediately; optionally preload
        self.build_index()
        self._filter_index_by_available_tree_paths()
        self._filter_index_by_posterior_subset_token_limit()
        self.set_overfit_fixed_pair_start_tree_bank(override_start_tree_bank)
        self.set_overfit_fixed_pair_target_tree_bank(override_target_tree_bank)
        if (
            self.overfit_full_path_control_mode
            and self.overfit_fixed_pair
            and self.overfit_fixed_pair_cache_virtual_index_selection
            and self.overfit_virtual_epoch_size is not None
            and len(self.overfit_fixed_pair_target_tree_newick_bank) > 1
        ):
            self.freeze_full_path_control_pair_bank(
                int(self.overfit_virtual_epoch_size),
                int(self.overfit_full_path_control_seed),
            )

    def _coerce_bank_item(
        self,
        raw_item: Any,
        *,
        tree_role: str,
    ) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]] = None
        if isinstance(raw_item, dict):
            payload = dict(raw_item)
            tree_newick = payload.get(tree_role)
            if tree_newick is None:
                tree_newick = payload.get("tree")
            if tree_newick is None and tree_role != "final_tree":
                tree_newick = payload.get("final_tree")
            if tree_newick is None:
                alt_keys = (
                    ("start_tree", "target_tree")
                    if tree_role == "start_tree"
                    else ("target_tree", "start_tree")
                )
                for alt_key in alt_keys:
                    tree_newick = payload.get(alt_key)
                    if tree_newick is not None:
                        break
        else:
            tree_newick = raw_item

        if tree_newick is None:
            return None

        tree_str = str(tree_newick).strip()
        if not tree_str:
            return None
        if not tree_str.endswith(";"):
            tree_str += ";"

        selected_original_labels = None
        if payload and payload.get("selected_original_labels") is not None:
            selected_original_labels = [
                str(label) for label in payload.get("selected_original_labels", [])
            ]

        explicit_group_key = None
        if payload:
            explicit_group_key = (
                payload.get("bank_group_key")
                or payload.get("subset_key")
                or payload.get("group_key")
            )
        group_key = _normalize_bank_group_key(
            explicit_group_key,
            selected_original_labels,
            tree_str,
        )
        subset_size = (
            int(payload.get("subset_size"))
            if payload and payload.get("subset_size") is not None
            else _leaf_count_from_newick(tree_str)
        )

        item = {
            "tree": tree_str,
            "group_key": str(group_key),
            "subset_size": int(subset_size),
            "selected_original_labels": selected_original_labels,
        }
        if payload:
            item["payload"] = payload
            if payload.get("dataset_id") is not None:
                item["dataset_id"] = str(payload.get("dataset_id")).upper()
        return item

    def _normalize_bank_items(
        self,
        raw_bank: List[Any],
        *,
        tree_role: str,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for raw_item in raw_bank:
            item = self._coerce_bank_item(raw_item, tree_role=tree_role)
            if item is None:
                continue
            dedupe_key = (item["group_key"], item["tree"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(item)
        return normalized

    def _rebuild_overfit_fixed_pair_bank_groups(self) -> None:
        start_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.overfit_fixed_pair_start_tree_bank_items:
            start_groups.setdefault(str(item["group_key"]), []).append(item)
        self._overfit_fixed_pair_start_tree_groups = start_groups

        target_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.overfit_fixed_pair_target_tree_bank_items:
            target_groups.setdefault(str(item["group_key"]), []).append(item)
        self._overfit_fixed_pair_target_tree_groups = target_groups

    def _sample_matching_overfit_fixed_pair_bank_items(
        self,
        dataset_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        start_items = list(self.overfit_fixed_pair_start_tree_bank_items or [])
        target_items = list(self.overfit_fixed_pair_target_tree_bank_items or [])
        if dataset_id is not None:
            dataset_id = str(dataset_id).upper()

            def _matches_dataset(item: Dict[str, Any]) -> bool:
                item_dataset_id = item.get("dataset_id")
                if item_dataset_id is None:
                    payload = item.get("payload") or {}
                    item_dataset_id = payload.get("dataset_id")
                return item_dataset_id is None or str(item_dataset_id).upper() == dataset_id

            start_items = [item for item in start_items if _matches_dataset(item)]
            target_items = [item for item in target_items if _matches_dataset(item)]
        if not start_items or not target_items:
            return None, None

        if not self.overfit_fixed_pair_group_by_json_metadata:
            chosen_start = random.choice(start_items)
            chosen_target = (
                random.choice(target_items) if len(target_items) > 1 else target_items[0]
            )
            return chosen_start, chosen_target

        start_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in start_items:
            start_groups.setdefault(str(item["group_key"]), []).append(item)
        target_groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in target_items:
            target_groups.setdefault(str(item["group_key"]), []).append(item)
        compatible_group_keys = sorted(set(start_groups.keys()) & set(target_groups.keys()))
        if not compatible_group_keys:
            return None, None

        chosen_group_key = random.choice(compatible_group_keys)
        chosen_start = random.choice(start_groups[chosen_group_key])
        chosen_target = random.choice(target_groups[chosen_group_key])
        return chosen_start, chosen_target

    def _normalize_start_tree_bank(self, start_tree_bank: List[Any]) -> List[Dict[str, Any]]:
        return self._normalize_bank_items(start_tree_bank, tree_role="start_tree")

    def set_overfit_fixed_pair_start_tree_bank(
        self,
        start_tree_bank: List[Any],
    ) -> List[str]:
        normalized = self._normalize_start_tree_bank(start_tree_bank)
        self.overfit_fixed_pair_start_tree_bank_items = list(normalized)
        self.overfit_fixed_pair_start_tree_newick_bank = [
            str(item["tree"]) for item in normalized
        ]
        self.overfit_fixed_pair_start_tree_newick = (
            self.overfit_fixed_pair_start_tree_newick_bank[0]
            if self.overfit_fixed_pair_start_tree_newick_bank
            else None
        )
        self._rebuild_overfit_fixed_pair_bank_groups()
        self._cached_overfit_pairs.clear()
        self._cached_overfit_pair_banks.clear()
        self._cached_overfit_bank_pairs_by_key.clear()
        self._cached_overfit_bank_selection_by_virtual_index.clear()
        self._cached_full_path_control_samples_by_key.clear()
        return list(self.overfit_fixed_pair_start_tree_newick_bank)

    def _normalize_target_tree_bank(self, target_tree_bank: List[Any]) -> List[Dict[str, Any]]:
        return self._normalize_bank_items(target_tree_bank, tree_role="target_tree")

    def set_overfit_fixed_pair_target_tree_bank(
        self,
        target_tree_bank: List[Any],
    ) -> List[str]:
        normalized = self._normalize_target_tree_bank(target_tree_bank)
        self.overfit_fixed_pair_target_tree_bank_items = list(normalized)
        self.overfit_fixed_pair_target_tree_newick_bank = [
            str(item["tree"]) for item in normalized
        ]
        self.overfit_fixed_pair_target_tree_newick = (
            self.overfit_fixed_pair_target_tree_newick_bank[0]
            if self.overfit_fixed_pair_target_tree_newick_bank
            else None
        )
        self._rebuild_overfit_fixed_pair_bank_groups()
        self._cached_overfit_pairs.clear()
        self._cached_overfit_pair_banks.clear()
        self._cached_overfit_bank_pairs_by_key.clear()
        self._cached_overfit_bank_selection_by_virtual_index.clear()
        self._cached_full_path_control_samples_by_key.clear()
        return list(self.overfit_fixed_pair_target_tree_newick_bank)

    def _oracle_prefix_candidates(
        self,
        start_tree_newick: str,
        target_tree_newick: str,
    ) -> List[str]:
        boundary_paths = return_tree_boundary_merge_paths(
            start_tree_newick,
            target_tree_newick,
            legacy_training_semantics=False,
        )
        candidates = [str(path["end_newick"]) for path in boundary_paths[:-1]]
        if not candidates:
            return []
        max_fraction = float(self.overfit_oracle_prefix_max_fraction)
        if 0.0 < max_fraction < 1.0:
            keep = max(1, int(math.ceil(len(candidates) * max_fraction)))
            candidates = candidates[:keep]
        return candidates

    def _sample_overfit_fixed_pair_bank_selection(
        self,
        *,
        allow_oracle_prefix: bool,
        dataset_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        chosen_start_item, chosen_target_item = (
            self._sample_matching_overfit_fixed_pair_bank_items(dataset_id=dataset_id)
        )
        if chosen_start_item is None or chosen_target_item is None:
            return None

        chosen_start_tree = str(chosen_start_item["tree"])
        chosen_target_tree = str(chosen_target_item["tree"])
        start_payload = chosen_start_item.get("payload") or {}
        target_payload = chosen_target_item.get("payload") or {}
        selection = {
            "start_item": chosen_start_item,
            "target_item": chosen_target_item,
            "forced_start_tree_newick": chosen_start_tree,
            "forced_target_tree_newick": chosen_target_tree,
            "bank_group_key": str(
                chosen_target_item.get(
                    "group_key",
                    chosen_start_item.get("group_key", f"n{_leaf_count_from_newick(chosen_target_tree)}"),
                )
            ),
            "selected_original_labels": (
                chosen_target_item.get("selected_original_labels")
                or chosen_start_item.get("selected_original_labels")
            ),
        }
        source_group_key = (
            target_payload.get("source_group_key")
            or start_payload.get("source_group_key")
            or target_payload.get("original_group_key")
            or start_payload.get("original_group_key")
        )
        if source_group_key is not None:
            selection["source_group_key"] = str(source_group_key)
        item_dataset_id = (
            chosen_target_item.get("dataset_id")
            or chosen_start_item.get("dataset_id")
            or target_payload.get("dataset_id")
            or start_payload.get("dataset_id")
        )
        if item_dataset_id is not None:
            selection["dataset_id"] = str(item_dataset_id).upper()

        if allow_oracle_prefix:
            oracle_prefix_candidates = self._oracle_prefix_candidates(
                chosen_start_tree,
                chosen_target_tree,
            )
            if oracle_prefix_candidates:
                oracle_start_tree = random.choice(oracle_prefix_candidates)
                selection["forced_start_tree_newick"] = str(oracle_start_tree)
                selection["oracle_prefix_start_tree"] = str(oracle_start_tree)
                selection["oracle_prefix_base_start_tree"] = str(chosen_start_tree)
                selection["oracle_prefix_target_tree"] = str(chosen_target_tree)

        return selection

    def freeze_full_path_control_pair_bank(
        self,
        sample_count: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        sample_count = max(0, int(sample_count))
        if sample_count == 0:
            self._frozen_full_path_control_selections = []
            return []
        if (
            not self.overfit_fixed_pair
            or len(self.overfit_fixed_pair_target_tree_newick_bank) <= 1
        ):
            self._frozen_full_path_control_selections = []
            return []

        random_state = random.getstate()
        selections: List[Dict[str, Any]] = []
        seen_starts = set()
        try:
            random.seed(int(seed))
            for _attempt in range(max(100, int(sample_count) * 50)):
                selection = self._sample_overfit_fixed_pair_bank_selection(
                    allow_oracle_prefix=False,
                )
                if selection is None:
                    break
                start_tree = str(selection.get("forced_start_tree_newick"))
                if start_tree in seen_starts:
                    continue
                seen_starts.add(start_tree)
                selections.append(dict(selection))
                if len(selections) >= int(sample_count):
                    break
        finally:
            random.setstate(random_state)

        if len(selections) < int(sample_count):
            raise RuntimeError(
                f"Could only freeze {len(selections)} unique full-path control pairs; requested {sample_count}."
            )

        self._cached_overfit_pairs.clear()
        self._cached_overfit_pair_banks.clear()
        self._cached_overfit_bank_pairs_by_key.clear()
        self._cached_overfit_bank_selection_by_virtual_index.clear()
        self._cached_full_path_control_samples_by_key.clear()
        for index, selection in enumerate(selections):
            self._cached_overfit_bank_selection_by_virtual_index[int(index)] = dict(
                selection
            )
        self._frozen_full_path_control_selections = [dict(x) for x in selections]
        return [dict(x) for x in selections]

    def set_overfit_fixed_pair_best_start_tree(
        self,
        start_tree_newick: str,
        *,
        max_bank_size: int = 2,
        keep_first: bool = True,
    ) -> List[str]:
        candidate = str(start_tree_newick).strip()
        if not candidate:
            return list(self.overfit_fixed_pair_start_tree_newick_bank)

        current_bank = list(self.overfit_fixed_pair_start_tree_newick_bank)
        if not current_bank:
            current_bank = [candidate]
        elif keep_first:
            anchor = current_bank[0]
            new_bank = [anchor]
            if candidate != anchor:
                new_bank.append(candidate)
            if int(max_bank_size) > 0:
                new_bank = new_bank[: max(1, int(max_bank_size))]
            return self.set_overfit_fixed_pair_start_tree_bank(new_bank)
        else:
            current_bank.append(candidate)

        if int(max_bank_size) > 0 and len(current_bank) > int(max_bank_size):
            current_bank = current_bank[-int(max_bank_size) :]
        return self.set_overfit_fixed_pair_start_tree_bank(current_bank)

    def resolve_training_target_tree(
        self,
        start_tree_newick: str,
        target_tree_newick: str,
        base_start_tree_newick: Optional[str] = None,
    ) -> str:
        resolved_target_tree = target_tree_newick
        if (
            self.overfit_start_boundary_prefix_k >= 0
            and self.overfit_boundary_prefix_k >= 0
            and (self.sanity_check or self.random_sanity_check)
        ):
            if base_start_tree_newick is None:
                original_prefix = self.overfit_start_boundary_prefix_k
                self.overfit_start_boundary_prefix_k = -1
                try:
                    base_start_tree = self.sample_random_tree(target_tree_newick)
                finally:
                    self.overfit_start_boundary_prefix_k = original_prefix
            else:
                base_start_tree = base_start_tree_newick
            resolved_target_tree = resolve_training_target_tree_for_prefix(
                base_start_tree,
                target_tree_newick,
                self.overfit_boundary_prefix_k,
            )
        else:
            resolved_target_tree = resolve_training_target_tree_for_prefix(
                start_tree_newick,
                target_tree_newick,
                self.overfit_boundary_prefix_k,
            )

        resolved_target_tree = resolve_training_target_tree_for_event_prefix(
            start_tree_newick,
            resolved_target_tree,
            self.overfit_event_prefix_count,
        )
        return _remap_tree_leaf_names_to_match_reference(
            resolved_target_tree,
            start_tree_newick,
        )

    def _build_target_one_split_off_terminal_samples(
        self,
        target_tree: str,
        *,
        path_index: int,
        timepoint: float,
    ) -> List[Dict[str, Any]]:
        target_obj = Tree(target_tree)
        n_leaves = int(target_obj.n_leaves)
        if n_leaves <= 3:
            return []

        masks, lengths = BHVEncoder().return_BHV_encoding(target_obj)
        full_mask = (1 << n_leaves) - 1
        split_lengths = {
            int(mask): float(length)
            for mask, length in zip(masks, lengths)
            if int(mask) != 0 and length is not None and float(length) > 1e-8
        }
        if not split_lengths:
            return []

        removable_splits = []
        for mask in split_lengths:
            side_a = int(mask).bit_count()
            side_b = int(full_mask ^ int(mask)).bit_count()
            if min(side_a, side_b) > 1:
                removable_splits.append(int(mask))

        samples: List[Dict[str, Any]] = []
        seen_newicks = set()
        for removed_split in sorted(set(removable_splits)):
            remaining_splits = [
                split for split in split_lengths if int(split) != int(removed_split)
            ]
            try:
                _, near_target_newick = build_tree_from_splits(
                    remaining_splits,
                    split_lengths,
                    n_leaves,
                    root_leaf=n_leaves - 1,
                    mapping=target_obj.id_to_name,
                )
            except Exception:
                continue
            if near_target_newick in seen_newicks:
                continue
            seen_newicks.add(near_target_newick)
            samples.append(
                {
                    "path_index": int(path_index),
                    "newick_tree": str(near_target_newick),
                    "timepoint": float(timepoint),
                    "terminal_stop": False,
                    "target_tree": str(target_tree),
                    "terminal_hard_negative_kind": "target_minus_one_split",
                    "removed_split": int(removed_split),
                }
            )

        return samples

    @staticmethod
    def _clone_full_path_control_sample_groups(
        groups: Tuple[
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
        ],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        cloned_groups: List[List[Dict[str, Any]]] = []
        for samples in groups:
            cloned_samples = []
            for sample in samples:
                cloned = dict(sample)
                velocity = cloned.get("velocity")
                if isinstance(velocity, dict):
                    cloned["velocity"] = dict(velocity)
                labels = cloned.get("labels")
                if isinstance(labels, list):
                    cloned["labels"] = [
                        dict(label) if isinstance(label, dict) else label
                        for label in labels
                    ]
                cloned_samples.append(cloned)
            cloned_groups.append(cloned_samples)
        return cloned_groups[0], cloned_groups[1], cloned_groups[2]

    @staticmethod
    def _attach_structural_newick(sample: Dict[str, Any], newick_key: str, out_key: str) -> None:
        if sample.get(out_key) is not None:
            return
        newick = sample.get(newick_key)
        if newick in {None, ""}:
            return
        sample[out_key] = _worker_newick_parser(str(newick))

    @classmethod
    def _attach_structural_fields_to_full_path_samples(
        cls,
        groups: Tuple[
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
        ],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        velocity_samples, autoregressive_samples, terminal_samples = groups
        for sample in velocity_samples:
            cls._attach_structural_newick(
                sample,
                "newick_tree",
                "newick_tree_structural",
            )
        for sample in terminal_samples:
            cls._attach_structural_newick(
                sample,
                "newick_tree",
                "newick_tree_structural",
            )
        for sample in autoregressive_samples:
            cls._attach_structural_newick(
                sample,
                "newick",
                "newick_structural",
            )
        return groups

    @staticmethod
    def _attach_batch_metadata_to_full_path_samples(
        groups: Tuple[
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
        ],
        *,
        dataset_id: str,
        sample_id: str,
        num_to_name: Dict[str, str],
        seq_ordering_map: Dict[str, Any],
        selected_sequences: Optional[List[str]] = None,
        selected_sequence_names: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        for samples in groups:
            for sample in samples:
                sample.setdefault("dataset_id", str(dataset_id).upper())
                sample.setdefault("id", sample_id)
                sample.setdefault("num_to_name", num_to_name)
                sample.setdefault("seq_ordering_map", seq_ordering_map)
                if selected_sequences is not None:
                    sample.setdefault("selected_sequences", list(selected_sequences))
                if selected_sequence_names is not None:
                    sample.setdefault(
                        "selected_sequence_names",
                        list(selected_sequence_names),
                    )
        return groups

    def _full_path_control_samples_cache_key(
        self,
        pair: Dict[str, Any],
        start_tree: str,
        target_tree: str,
        boundary_paths: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        return (
            start_tree,
            target_tree,
            pair.get("bank_group_key"),
            len(boundary_paths),
            self.overfit_velocity_explicit_boundary_label_scale_mode,
            bool(self.overfit_full_path_control_use_discrete_phase_time),
            self.overfit_full_path_control_terminal_label_mode,
            bool(self.overfit_full_path_control_terminal_include_ar_states),
            bool(self.overfit_full_path_control_terminal_include_target_one_split_off),
            len(self.overfit_full_path_control_extra_velocity_samples),
            len(pair.get("extra_velocity_samples") or []),
        )

    def _build_full_path_control_samples(
        self,
        pair: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        label_scale_mode = self.overfit_velocity_explicit_boundary_label_scale_mode
        start_tree = str(pair["random_tree"])
        target_tree = str(pair["effective_target_tree"])
        boundary_paths = list(pair["boundary_paths"])
        pair_group_key = pair.get("bank_group_key")
        cache_key = self._full_path_control_samples_cache_key(
            pair,
            start_tree,
            target_tree,
            boundary_paths,
        )
        cached = self._cached_full_path_control_samples_by_key.get(cache_key)
        if cached is not None:
            return self._clone_full_path_control_sample_groups(cached)

        def _attach_pair_group(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample.setdefault("start_tree", start_tree)
            if pair_group_key is not None:
                sample["bank_group_key"] = str(pair_group_key)
            if pair.get("dataset_id") is not None:
                sample["dataset_id"] = str(pair.get("dataset_id")).upper()
            return sample

        velocity_samples: List[Dict[str, Any]] = []
        autoregressive_samples: List[Dict[str, Any]] = []
        terminal_samples: List[Dict[str, Any]] = []
        prev_time = 0.0
        for path_index, path in enumerate(boundary_paths):
            source_tree = (
                start_tree
                if int(path_index) == 0
                else str(boundary_paths[int(path_index) - 1]["end_newick"])
            )
            path_start_time = (
                float(path_index)
                if self.overfit_full_path_control_use_discrete_phase_time
                else float(prev_time)
            )
            velocity_newick, velocity = return_sampled_tree_orthant_velocity(
                source_tree,
                target_tree,
                0.0,
                legacy_training_semantics=False,
            )
            scale = 1.0
            if label_scale_mode == "remaining":
                scale = 1.0 / max(1.0 - float(prev_time), 1e-6)
            elif label_scale_mode != "local":
                raise ValueError(
                    f"Unknown overfit_velocity_explicit_boundary_label_scale_mode={label_scale_mode!r}."
                )
            velocity_samples.append(
                _attach_pair_group({
                    "path_index": int(path_index),
                    "newick_tree": str(velocity_newick),
                    "target_tree": target_tree,
                    "velocity": {
                        int(k): float(v) * float(scale)
                        for k, v in velocity.items()
                    },
                    "velocity_next_boundary_tree": str(path["start_newick"]),
                    "timepoint": path_start_time,
                    "num_leaves": int(Tree(source_tree).n_leaves),
                })
            )
            boundary_time = (
                float(path_index)
                if self.overfit_full_path_control_use_discrete_phase_time
                else float(path["global_time"])
            )
            if self.overfit_full_path_control_birthset_boundary_labels:
                birthset_event = _birthset_boundary_training_event(
                    str(path["start_newick"]),
                    path.get("births", []),
                )
                boundary_events = [] if birthset_event is None else [birthset_event]
            else:
                boundary_events = []
                for event in path.get("events", []):
                    if not event.get("labels"):
                        continue
                    boundary_events.append(
                        {
                            "newick": str(event["newick"]),
                            "labels": list(event["labels"]),
                        }
                    )
                boundary_events = _split_multi_label_training_events(boundary_events)
            for event in boundary_events:
                autoregressive_samples.append(
                    _attach_pair_group({
                        "path_index": int(path_index),
                        "newick": str(event["newick"]),
                        "target_tree": target_tree,
                        "labels": list(event["labels"]),
                        "stop_after_merge": bool(
                            event.get("stop_after_merge", False)
                        ),
                        "time": boundary_time,
                    })
                )
                if self.overfit_full_path_control_terminal_include_ar_states:
                    terminal_samples.append(
                        _attach_pair_group({
                            "path_index": int(path_index),
                            "newick_tree": str(event["newick"]),
                            "timepoint": boundary_time,
                            "terminal_stop": False,
                            "target_tree": target_tree,
                        })
                    )
            if self.overfit_full_path_control_terminal_label_mode == "phase_start":
                terminal_samples.append(
                    _attach_pair_group({
                        "path_index": int(path_index),
                        "newick_tree": str(source_tree),
                        "timepoint": path_start_time,
                        "terminal_stop": bool(
                            int(path_index) == (len(boundary_paths) - 1)
                        ),
                        "target_tree": target_tree,
                    })
                )
            else:
                terminal_samples.append(
                    _attach_pair_group({
                        "path_index": int(path_index),
                        "newick_tree": str(source_tree),
                        "timepoint": path_start_time,
                        "terminal_stop": False,
                        "target_tree": target_tree,
                    })
                )
            prev_time = boundary_time

        if self.overfit_full_path_control_terminal_label_mode == "target":
            target_time = (
                float(len(boundary_paths))
                if self.overfit_full_path_control_use_discrete_phase_time
                else 1.0
            )
            terminal_samples.append(
                _attach_pair_group({
                    "path_index": int(len(boundary_paths)),
                    "newick_tree": str(target_tree),
                    "timepoint": target_time,
                    "terminal_stop": True,
                    "target_tree": target_tree,
                })
            )
            if self.overfit_full_path_control_terminal_include_target_one_split_off:
                for hard_negative in self._build_target_one_split_off_terminal_samples(
                    target_tree,
                    path_index=int(len(boundary_paths)),
                    timepoint=target_time,
                ):
                    terminal_samples.append(
                        _attach_pair_group(hard_negative)
                    )

        extra_velocity_samples = self._extra_velocity_samples_for_pair(pair)
        extra_velocity_samples.extend(pair.get("extra_velocity_samples") or [])
        acceptable_group_keys = {
            str(key)
            for key in (pair.get("bank_group_key"), pair.get("source_group_key"))
            if key is not None
        }
        for extra_sample in extra_velocity_samples:
            sample_group_key = extra_sample.get("bank_group_key")
            pair_group_key = pair.get("bank_group_key")
            if (
                sample_group_key is not None
                and acceptable_group_keys
                and str(sample_group_key) not in acceptable_group_keys
            ):
                continue
            relabeled_sample = dict(extra_sample)
            relabeled_sample["start_tree"] = start_tree
            relabeled_sample["target_tree"] = target_tree
            relabeled_sample["path_index"] = int(
                extra_sample.get("path_index", extra_sample.get("phase_idx", 0))
            )
            if self.overfit_full_path_control_use_discrete_phase_time:
                relabeled_sample["timepoint"] = float(
                    extra_sample.get(
                        "path_index",
                        extra_sample.get("phase_idx", extra_sample.get("timepoint", 0.0)),
                    )
                )
            else:
                relabeled_sample["timepoint"] = float(
                    extra_sample.get("timepoint", 0.0)
                )
            if pair_group_key is not None:
                relabeled_sample["bank_group_key"] = str(pair_group_key)
            velocity_samples.append(relabeled_sample)

        result = (velocity_samples, autoregressive_samples, terminal_samples)
        if self.full_path_preparse_structural_trees:
            result = self._attach_structural_fields_to_full_path_samples(result)
        self._cached_full_path_control_samples_by_key[cache_key] = (
            self._clone_full_path_control_sample_groups(result)
        )
        return self._clone_full_path_control_sample_groups(result)

    def _ensure_dataset_extra_velocity_samples_loaded(
        self,
        dataset_id: Optional[str],
    ) -> None:
        if dataset_id is None:
            return
        dataset_id = str(dataset_id).upper()
        if dataset_id in self._overfit_full_path_control_loaded_extra_dataset_ids:
            return
        path = self._overfit_full_path_control_extra_velocity_sample_paths_by_dataset_id.get(
            dataset_id
        )
        if not path:
            self._overfit_full_path_control_loaded_extra_dataset_ids.add(dataset_id)
            return
        samples = _load_full_path_control_extra_velocity_samples(path)
        for group_key, grouped_samples in _index_velocity_samples_by_group_key(samples).items():
            self._overfit_full_path_control_extra_velocity_samples_by_group_key.setdefault(
                str(group_key),
                [],
            ).extend(grouped_samples)
        self._overfit_full_path_control_loaded_extra_dataset_ids.add(dataset_id)

    def _extra_velocity_samples_for_pair(
        self,
        pair: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        pair_group_key = pair.get("bank_group_key")
        source_group_key = pair.get("source_group_key")
        dataset_id = pair.get("dataset_id")
        self._ensure_dataset_extra_velocity_samples_loaded(dataset_id)

        indexed = self._overfit_full_path_control_extra_velocity_samples_by_group_key
        keys = [
            str(key)
            for key in (pair_group_key, source_group_key)
            if key is not None
        ]
        samples: List[Dict[str, Any]] = []
        for key in keys:
            samples.extend(indexed.get(key, []))
        if samples:
            return list(samples)
        return list(self.overfit_full_path_control_extra_velocity_samples)

    def _random_distribution_sequences(
        self,
        dataset_id: str,
        taxa_order: List[str],
    ) -> Dict[str, str]:
        seqs: Dict[str, str] = {}
        alphabet = self.random_distribution_alphabet
        for taxon_name in taxa_order:
            rng = random.Random(
                f"{self.random_distribution_sequence_seed}:{dataset_id}:{taxon_name}"
            )
            seqs[str(taxon_name)] = "".join(
                rng.choice(alphabet)
                for _ in range(self.random_distribution_sequence_length)
            )
        return seqs

    def _sequences_for_meta(self, meta: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
        if meta.get("random_distribution"):
            taxa_order = [str(name) for name in meta.get("taxa_order", [])]
            if not taxa_order and meta.get("num_leaves") is not None:
                taxa_order = [str(i) for i in range(1, int(meta["num_leaves"]) + 1)]
            return (
                self._random_distribution_sequences(str(meta["id"]), taxa_order),
                taxa_order,
            )
        nexus_path = str(meta["nexus_path"])
        cached = self._cached_sequences_by_nexus_path.get(nexus_path)
        if cached is None:
            seqs, taxa_order = self.parse_nexus(nexus_path)
            cached = (dict(seqs), list(taxa_order))
            self._cached_sequences_by_nexus_path[nexus_path] = cached
        return dict(cached[0]), list(cached[1])

    def _filter_index_by_posterior_subset_token_limit(self) -> None:
        subset_size = int(getattr(self, "posterior_subset_num_leaves", 0) or 0)
        max_tokens = int(getattr(self, "posterior_subset_max_input_tokens", 0) or 0)
        if subset_size <= 0 or max_tokens <= 0 or not self._index:
            return

        filtered_index: List[Dict[str, Any]] = []
        dropped_ids: List[str] = []
        for meta in self._index:
            if meta.get("topology_stream_pair") or meta.get("random_distribution"):
                filtered_index.append(meta)
                continue
            try:
                seqs, taxa_order = self._sequences_for_meta(meta)
            except Exception:
                filtered_index.append(meta)
                continue
            stripped_lengths = sorted(
                len(self._strip_live_phyla_input_sequence(seqs.get(str(name), "")))
                for name in taxa_order
                if str(name) in seqs
            )
            if len(stripped_lengths) < subset_size:
                filtered_index.append(meta)
                continue
            shortest_tokens = int(sum(stripped_lengths[:subset_size]) + subset_size)
            if shortest_tokens <= max_tokens:
                filtered_index.append(meta)
            else:
                dropped_ids.append(str(meta.get("id", "")))

        if not dropped_ids:
            return
        self._index = filtered_index
        self._ids = [str(meta["id"]) for meta in self._index]
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}
        print(
            "Filtered "
            f"{len(dropped_ids)} posterior-subset datasets whose shortest "
            f"{subset_size} raw sequences exceed {max_tokens} live-Phyla tokens "
            f"({len(self._index)} remain)."
        )

    def _filter_index_by_available_tree_paths(self) -> None:
        if (
            self.topology_stream_index_jsonl_path
            or self.posterior_trprobs_root
            or not self._index
        ):
            return
        filtered_index = [
            meta for meta in self._index if meta.get("tree_paths")
        ]
        dropped_count = len(self._index) - len(filtered_index)
        if dropped_count <= 0:
            return
        self._index = filtered_index
        self._ids = [str(meta["id"]) for meta in self._index]
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}
        print(
            f"Filtered {dropped_count} posterior datasets with no tree paths "
            f"({len(self._index)} remain)."
        )

    @staticmethod
    def _strip_live_phyla_input_sequence(sequence: str) -> str:
        return str(sequence or "").replace("-", "").replace(".", "")

    def _topology_stream_nexus_path(self, dataset_id: str) -> str:
        dataset_id = str(dataset_id).upper()
        root = Path(str(self.nexus_root))
        candidates = [
            root / f"{dataset_id}.nex",
            root / f"{dataset_id}.nexus",
            root / f"{dataset_id.lower()}.nex",
            root / f"{dataset_id.lower()}.nexus",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"No NEXUS alignment found for topology-stream dataset {dataset_id} "
            f"under {root}"
        )

    def _topology_stream_sequences_for_dataset(
        self,
        dataset_id: str,
        nexus_path: Optional[str] = None,
    ) -> Tuple[Dict[str, str], List[str]]:
        dataset_id = str(dataset_id).upper()
        cached = self._cached_topology_stream_sequences_by_dataset_id.get(dataset_id)
        if cached is not None:
            return cached
        path = nexus_path or self._topology_stream_nexus_path(dataset_id)
        parsed = self.parse_nexus(path)
        self._cached_topology_stream_sequences_by_dataset_id[dataset_id] = parsed
        return parsed

    def _resolve_sequence_for_leaf_label(
        self,
        leaf_label: str,
        seqs: Dict[str, str],
        source_taxa_order: List[str],
    ) -> Tuple[str, str]:
        label = str(leaf_label)
        if label in seqs:
            return label, seqs[label]
        try:
            idx = int(label)
        except ValueError:
            idx = None
        if idx is not None:
            if 1 <= idx <= len(source_taxa_order):
                name = str(source_taxa_order[idx - 1])
                return name, seqs.get(name, "")
            if 0 <= idx < len(source_taxa_order):
                name = str(source_taxa_order[idx])
                return name, seqs.get(name, "")
        return label, seqs.get(label, "")

    def _selected_sequences_from_leaf_labels(
        self,
        leaf_labels: List[str],
        seqs: Dict[str, str],
        source_taxa_order: List[str],
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        new_seqs: Dict[str, str] = {}
        selected_names: List[str] = []
        selected_sequences: List[str] = []
        for idx, leaf_label in enumerate(leaf_labels):
            sequence_name, sequence = self._resolve_sequence_for_leaf_label(
                str(leaf_label),
                seqs,
                source_taxa_order,
            )
            clean_sequence = self._strip_live_phyla_input_sequence(sequence)
            new_seqs[str(idx)] = clean_sequence
            selected_names.append(str(sequence_name))
            selected_sequences.append(clean_sequence)
        return new_seqs, selected_names, selected_sequences

    def _topology_stream_input_tokens_for_labels(
        self,
        leaf_labels: List[str],
        seqs: Dict[str, str],
        source_taxa_order: List[str],
    ) -> int:
        _new_seqs, _names, selected_sequences = self._selected_sequences_from_leaf_labels(
            leaf_labels,
            seqs,
            source_taxa_order,
        )
        return int(sum(len(sequence) for sequence in selected_sequences) + len(selected_sequences))

    def sample_random_tree_with_base(
        self,
        real_tree,
        subtree_size: Optional[int] = None,
    ) -> Tuple[str, str]:
        if self.overfit_start_boundary_prefix_k < 0:
            start_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
            return start_tree, start_tree

        original_prefix = self.overfit_start_boundary_prefix_k
        self.overfit_start_boundary_prefix_k = -1
        try:
            base_random_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
        finally:
            self.overfit_start_boundary_prefix_k = original_prefix

        target_tree_newick = real_tree if isinstance(real_tree, str) else real_tree.write(format=1)
        start_tree = resolve_training_target_tree_for_prefix(
            base_random_tree,
            target_tree_newick,
            original_prefix,
        )
        return base_random_tree, start_tree

    def __len__(self) -> int:  # Required for torch Dataset
        if self.overfit_virtual_epoch_size is not None and len(self._ids) > 0:
            return int(self.overfit_virtual_epoch_size)
        return len(self._ids)

    def return_number_leaves(self, index: int) -> int:
        """Return number of leaves in the alignment for the given index."""
        if type(index) == str:
            index = self._id_to_idx[index]
        meta = self._index[index]
        if meta.get("random_distribution"):
            if meta.get("num_leaves") is not None:
                return int(meta.get("num_leaves"))
            return len(meta.get("taxa_order", []))
        seqs, _ = self.parse_nexus(meta["nexus_path"])
        return len(seqs)

    def return_posterior_trees(self, index: int) -> List[str]:
        """Return list of posterior Newick trees for the given index.

        Applies burn-in and thinning as per load_posterior_trees_from_tfiles.
        """
        if type(index) == str:
            index = self._id_to_idx[index] 
        meta = self._index[index]
        tree_paths = meta["tree_paths"]
        trees = self.load_posterior_trees_from_tfiles(tree_paths)
        if self.sanity_check or self.random_sanity_check:
            return ['((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);']
        return trees

    def return_nexus_filepath(self, index: int) -> str:
        """Return the Nexus file path for the given index."""
        if type(index) == str:
            index = self._id_to_idx[index]
        meta = self._index[index]
        return meta["nexus_path"]

    def return_nexus_number_to_name(self, index: int) -> Dict[int, str]:
        """Return mapping from taxon number to name for the given index."""
        if type(index) == str:
            index = self._id_to_idx[index]
        meta = self._index[index]
        if meta.get("random_distribution"):
            taxa_order = list(meta.get("taxa_order") or [])
            if not taxa_order and meta.get("num_leaves") is not None:
                taxa_order = [str(i) for i in range(1, int(meta["num_leaves"]) + 1)]
            return {i: name for i, name in enumerate(taxa_order)}
        _, taxa_order = self.parse_nexus(meta["nexus_path"])
        num_to_name = {i: name for i, name in enumerate(taxa_order)}
        return num_to_name

    def __getitem__(
        self, index: int, preset_subtree_size: Optional[int] = None
    ) -> Dict[str, Any]:  # Required for torch Dataset
        requested_index = int(index)
        same_dataset_batch_size = int(getattr(self, "same_dataset_batch_size", 0) or 0)
        if same_dataset_batch_size > 1 and len(self._index) > 0:
            group_code = requested_index // same_dataset_batch_size
            index = group_code % len(self._index)
            group_id = group_code // len(self._index)
            shared_subset_seed = (
                f"{getattr(self, 'same_dataset_batch_seed', 0)}:"
                f"{group_id}:{index}"
            )
        elif self.overfit_virtual_epoch_size is not None and len(self._index) > 0:
            index = requested_index % len(self._index)
            shared_subset_seed = None
        else:
            shared_subset_seed = None
        meta = self._index[index]
        if (
            preset_subtree_size is None
            and self.posterior_subset_num_leaves > 0
            and not meta.get("topology_stream_pair")
        ):
            preset_subtree_size = int(self.posterior_subset_num_leaves)
        if self.validation:
            return {
                "id": meta["id"],
                "posterior_trees": self.return_posterior_trees(index),
                "nexus_path": meta["nexus_path"],
                "tree_paths": meta["tree_paths"],
                "num_to_name": self.return_nexus_number_to_name(index),
            }

        seqs, taxa_order = self._sequences_for_meta(meta)
        source_taxa_order = list(taxa_order)
        selected_sequence_names = None
        selected_sequences = None

        # Update name_to_seq cache (dumb update for now)
        self.name_to_seq = seqs

        topology_stream_selection = None
        topology_stream_extra_velocity_samples: List[Dict[str, Any]] = []
        if meta.get("topology_stream_pair"):
            stream_pair = self._load_topology_stream_pair_for_meta(meta)
            start_tree_newick = str(stream_pair["start_tree"])
            target_tree_newick = str(stream_pair["target_tree"])
            topology_stream_pruned = False
            start_obj = EteTree(start_tree_newick, format=1)
            target_obj = EteTree(target_tree_newick, format=1)
            leaves = sorted(
                target_obj.get_leaves(),
                key=lambda leaf: _numeric_name_sort_key(leaf.name),
            )
            effective_subtree_size = preset_subtree_size
            if (
                effective_subtree_size is None
                and self.topology_stream_subset_num_leaves > 0
            ):
                effective_subtree_size = int(self.topology_stream_subset_num_leaves)
            if (
                effective_subtree_size is not None
                and len(leaves) > effective_subtree_size
            ):
                keep_count = max(2, int(effective_subtree_size))
                candidate_leaf_names = [str(leaf.name) for leaf in leaves]
                keep_names = set(random.sample(candidate_leaf_names, keep_count))
                if (
                    self.topology_stream_use_real_sequences
                    and self.topology_stream_max_input_tokens > 0
                ):
                    best_keep_names = keep_names
                    best_tokens = None
                    for _attempt in range(32):
                        candidate = set(
                            random.sample(candidate_leaf_names, keep_count)
                        )
                        candidate_tokens = self._topology_stream_input_tokens_for_labels(
                            sorted(candidate, key=_numeric_name_sort_key),
                            seqs,
                            source_taxa_order,
                        )
                        if best_tokens is None or candidate_tokens < best_tokens:
                            best_keep_names = candidate
                            best_tokens = candidate_tokens
                        if candidate_tokens <= self.topology_stream_max_input_tokens:
                            keep_names = candidate
                            break
                    else:
                        lengths = []
                        for leaf_name in candidate_leaf_names:
                            _sequence_name, sequence = self._resolve_sequence_for_leaf_label(
                                leaf_name,
                                seqs,
                                source_taxa_order,
                            )
                            lengths.append(
                                (
                                    len(self._strip_live_phyla_input_sequence(sequence)),
                                    leaf_name,
                                )
                            )
                        shortest = [
                            name
                            for _length, name in sorted(lengths)[:keep_count]
                        ]
                        shortest_tokens = self._topology_stream_input_tokens_for_labels(
                            sorted(shortest, key=_numeric_name_sort_key),
                            seqs,
                            source_taxa_order,
                        )
                        keep_names = (
                            set(shortest)
                            if shortest_tokens <= self.topology_stream_max_input_tokens
                            else best_keep_names
                        )
                start_leaf_names = {str(leaf.name) for leaf in start_obj.get_leaves()}
                missing_start_leaves = sorted(keep_names - start_leaf_names)
                if missing_start_leaves:
                    raise ValueError(
                        "Cannot prune topology-stream start/target pair because "
                        f"start tree is missing leaves: {missing_start_leaves[:5]}"
                    )
                ordered_keep_names = sorted(keep_names, key=_numeric_name_sort_key)
                start_obj.prune(ordered_keep_names, preserve_branch_length=True)
                target_obj.prune(ordered_keep_names, preserve_branch_length=True)
                topology_stream_pruned = True

            (
                start_tree_newick,
                target_tree_newick,
                original_topology_leaf_order,
                topology_leaf_to_index,
            ) = _normalize_topology_stream_tree_pair(start_obj, target_obj)
            leaves = [
                str(idx)
                for idx in range(len(original_topology_leaf_order))
            ]
            taxa_order = list(leaves)
            if self.topology_stream_use_real_sequences:
                (
                    new_seqs,
                    selected_sequence_names,
                    selected_sequences,
                ) = self._selected_sequences_from_leaf_labels(
                    original_topology_leaf_order,
                    seqs,
                    source_taxa_order,
                )
                input_tokens = int(
                    sum(len(sequence) for sequence in selected_sequences)
                    + len(selected_sequences)
                )
                if (
                    self.topology_stream_max_input_tokens > 0
                    and input_tokens > self.topology_stream_max_input_tokens
                ):
                    raise ValueError(
                        f"Topology-stream item {meta['id']} has {input_tokens} live "
                        "Phyla input tokens after pruning, exceeding "
                        f"topology_stream_max_input_tokens="
                        f"{self.topology_stream_max_input_tokens}."
                    )
            else:
                new_seqs = self._random_distribution_sequences(str(meta["id"]), taxa_order)
                selected_sequence_names = list(taxa_order)
                selected_sequences = [
                    new_seqs[str(idx)] for idx in range(len(taxa_order))
                ]
            original_names_map = {
                str(idx): str(name)
                for idx, name in enumerate(original_topology_leaf_order)
            }
            seq_ordering_map = {
                str(name): str(idx)
                for name, idx in topology_leaf_to_index.items()
            }
            current_size = len(taxa_order)
            self.chosen_tree = (index, current_size, 1)
            real_tree_newick = target_tree_newick
            real_tree_original_label_newick = target_tree_newick
            sample_source_tree = target_tree_newick

            def _remap_random_tree_to_dataset_indexing(random_tree_newick: str) -> str:
                tree_str = str(random_tree_newick).strip()
                return tree_str if tree_str.endswith(";") else f"{tree_str};"

            topology_stream_selection = {
                "forced_start_tree_newick": start_tree_newick,
                "forced_target_tree_newick": target_tree_newick,
                "bank_group_key": str(meta.get("bank_group_key")),
                "dataset_id": str(meta.get("dataset_id", meta["id"])).upper(),
            }
            if not topology_stream_pruned:
                # Extra velocity samples were serialized in the source leaf/root
                # convention.  After canonicalizing labels and root placement,
                # reusing their old split masks would mix coordinate systems.
                topology_stream_extra_velocity_samples = []
            forced_bank_selection = None
        else:
            # Attempt to parse translation block from the first tree file
            translate_map = {}
            if meta["tree_paths"]:
                translate_map = self.parse_translate_block(meta["tree_paths"][0])

            trees = self.load_posterior_trees_from_tfiles(meta["tree_paths"])
            if not trees:
                # Fallback: try to reload or skip. For now, raise informative error or return another item
                print(
                    f"Dataset Warning: No trees found in {meta['tree_paths']}. Skipping/Replacing with index 0."
                )
                return self.__getitem__(0, preset_subtree_size)
            forced_bank_selection = None
            if (
                self.overfit_fixed_pair
                and self.overfit_fixed_pair_reference_tree_from_target_bank
                and self.overfit_fixed_pair_target_tree_bank_items
            ):
                forced_bank_selection = self._sample_overfit_fixed_pair_bank_selection(
                    allow_oracle_prefix=(
                        not self.validation
                        and self.overfit_oracle_prefix_start_prob > 0.0
                        and random.random() < self.overfit_oracle_prefix_start_prob
                    ),
                    dataset_id=meta.get("id"),
                )

            using_selected_fixed_bank_subset = False
            if (
                forced_bank_selection is not None
                and forced_bank_selection.get("selected_original_labels")
            ):
                using_selected_fixed_bank_subset = True
                real_tree_newick = str(
                    forced_bank_selection["forced_target_tree_newick"]
                ).strip()
                if not real_tree_newick.endswith(";"):
                    real_tree_newick += ";"
                chosen_original_labels = [
                    str(label)
                    for label in forced_bank_selection["selected_original_labels"]
                ]
                current_size = len(chosen_original_labels)
                self.chosen_tree = (index, current_size, 1)
                real_tree_original_label_newick = real_tree_newick
                new_seqs = {}
                original_names_map = {}
                seq_ordering_map = {}
                for i, original_node_name in enumerate(chosen_original_labels):
                    taxon_name = translate_map.get(original_node_name, original_node_name)
                    new_idx_str = str(i)
                    new_seqs[new_idx_str] = seqs.get(taxon_name, "")
                    original_names_map[new_idx_str] = taxon_name
                    seq_ordering_map[original_node_name] = new_idx_str

                def _remap_random_tree_to_dataset_indexing(random_tree_newick: str) -> str:
                    tree_str = str(random_tree_newick).strip()
                    return tree_str if tree_str.endswith(";") else f"{tree_str};"

                sample_source_tree = real_tree_newick
            else:
                #######VERY IMPORTANT HERE FOR DEBUG PURPOSES WE WILL ALWAYS SAMPLE THE FIRST TREE########
                real_tree_newick = random.sample(trees, 1)[0]
                # real_tree_newick = trees[0]
                #########################################################################################

                t = EteTree(real_tree_newick, format=1)
                leaves = t.get_leaves()

            # Pruning logic for adaptive batching

            if (
                not using_selected_fixed_bank_subset
                and preset_subtree_size is not None
                and len(leaves) > preset_subtree_size
            ):
                keep_count = max(2, int(preset_subtree_size))
                candidate_leaf_names = sorted(
                    [str(leaf.name) for leaf in leaves],
                    key=_numeric_name_sort_key,
                )
                leaves_by_name = {str(leaf.name): leaf for leaf in leaves}

                def _posterior_subset_input_tokens(leaf_names: List[str]) -> int:
                    total = len(leaf_names)
                    for leaf_name in leaf_names:
                        taxon_name = translate_map.get(str(leaf_name))
                        if taxon_name is not None:
                            sequence = seqs.get(taxon_name, "")
                        else:
                            _sequence_name, sequence = self._resolve_sequence_for_leaf_label(
                                str(leaf_name),
                                seqs,
                                source_taxa_order,
                            )
                        total += len(self._strip_live_phyla_input_sequence(sequence))
                    return int(total)

                def _sample_keep_names(rng) -> set[str]:
                    return set(rng.sample(candidate_leaf_names, keep_count))

                if shared_subset_seed is not None:
                    subset_rng = random.Random(
                        f"{shared_subset_seed}:posterior_subset"
                    )
                    keep_names = _sample_keep_names(subset_rng)
                else:
                    subset_rng = random
                    keep_names = _sample_keep_names(subset_rng)
                if self.posterior_subset_max_input_tokens > 0:
                    best_keep_names = set(keep_names)
                    best_tokens = _posterior_subset_input_tokens(
                        sorted(best_keep_names, key=_numeric_name_sort_key)
                    )
                    if best_tokens > self.posterior_subset_max_input_tokens:
                        for _attempt in range(64):
                            candidate = _sample_keep_names(subset_rng)
                            candidate_tokens = _posterior_subset_input_tokens(
                                sorted(candidate, key=_numeric_name_sort_key)
                            )
                            if candidate_tokens < best_tokens:
                                best_keep_names = set(candidate)
                                best_tokens = candidate_tokens
                            if candidate_tokens <= self.posterior_subset_max_input_tokens:
                                keep_names = set(candidate)
                                break
                        else:
                            lengths = []
                            for leaf_name in candidate_leaf_names:
                                taxon_name = translate_map.get(str(leaf_name))
                                if taxon_name is not None:
                                    sequence = seqs.get(taxon_name, "")
                                else:
                                    _sequence_name, sequence = self._resolve_sequence_for_leaf_label(
                                        str(leaf_name),
                                        seqs,
                                        source_taxa_order,
                                    )
                                lengths.append(
                                    (
                                        len(self._strip_live_phyla_input_sequence(sequence)),
                                        str(leaf_name),
                                    )
                                )
                            shortest = [
                                name
                                for _length, name in sorted(lengths)[:keep_count]
                            ]
                            shortest_tokens = _posterior_subset_input_tokens(shortest)
                            keep_names = (
                                set(shortest)
                                if shortest_tokens <= self.posterior_subset_max_input_tokens
                                else best_keep_names
                            )
                kept_leaves = [
                    leaves_by_name[name]
                    for name in sorted(keep_names, key=_numeric_name_sort_key)
                ]
                t.prune(kept_leaves, preserve_branch_length=True)
                # real_tree_newick = t.write(format=1) # Don't write yet, wait for re-indexing
                # Update leaves for size tracking
                leaves = t.get_leaves()

            if not using_selected_fixed_bank_subset:
                real_tree_original_label_newick = t.write(format=1)

                current_size = len(leaves)
                self.chosen_tree = (index, current_size, 1)  # (index, size, num_subtrees)

                # Normalize tree indices to 0..N-1 and subset sequences
                # Sort leaves for deterministic indexing
                leaves.sort(key=lambda x: _numeric_name_sort_key(x.name))

                new_seqs = {}
                original_names_map = {}
                seq_ordering_map = {}

                for i, leaf in enumerate(leaves):
                    original_node_name = leaf.name
                    # Resolve taxon name: check translate map, else use node name
                    taxon_name = translate_map.get(original_node_name, original_node_name)

                    # Map new index (0..N-1) to sequence
                    new_idx_str = str(i)
                    # Store sequences using the new index as key
                    new_seqs[new_idx_str] = seqs.get(taxon_name, "")

                    # Rename leaf in the tree
                    leaf.name = new_idx_str

                    # Record mapping if needed
                    original_names_map[new_idx_str] = taxon_name

                    seq_ordering_map[original_node_name] = new_idx_str

                # Serialize the normalized tree
                real_tree_newick = t.write(format=1)

                # Re-parse purely to ensure we are passing consistent objects
                # (Though prune modifies in-place, let's keep it safe)
                t_pruned = EteTree(real_tree_newick, format=1)

                def _remap_random_tree_to_dataset_indexing(random_tree_newick: str) -> str:
                    t_random = EteTree(random_tree_newick, format=1)
                    dataset_leaf_names = set(new_seqs.keys())

                    # Now remap the random tree to make the indices match up with the real tree.
                    if self.sanity_check:
                        for leaf in t_random.get_leaves():
                            name = leaf.name
                            if name in dataset_leaf_names:
                                continue
                            if name in seq_ordering_map:
                                leaf.name = seq_ordering_map[name]
                            else:
                                raise Exception(
                                    "Leaf name in random tree not found in original names map!"
                                )
                    else:
                        for leaf in t_random.get_leaves():
                            raw_name = leaf.name
                            if raw_name in dataset_leaf_names:
                                continue
                            try:
                                name = str(int(raw_name))
                            except ValueError:
                                name = raw_name
                            if name in seq_ordering_map:
                                leaf.name = seq_ordering_map[name]
                            else:
                                raise Exception(
                                    "Leaf name in random tree not found in original names map!"
                                )
                    return t_random.write(format=1)

                sample_source_tree = t_pruned
                if self.overfit_start_boundary_prefix_k >= 0 and (
                    self.sanity_check or self.random_sanity_check
                ):
                    # Keep start/target prefix resolution in the original leaf-name space,
                    # then remap both together into dataset indexing.
                    sample_source_tree = real_tree_original_label_newick

        def _build_pair(
            forced_start_tree_newick: Optional[str] = None,
            forced_target_tree_newick: Optional[str] = None,
        ) -> Dict[str, Any]:
            chosen_start_tree_newick = (
                str(forced_start_tree_newick)
                if forced_start_tree_newick is not None
                else None
            )
            if (
                chosen_start_tree_newick is None
                and self.overfit_fixed_pair
                and self.overfit_fixed_pair_start_tree_newick
            ):
                chosen_start_tree_newick = str(self.overfit_fixed_pair_start_tree_newick)

            if chosen_start_tree_newick is not None:
                random_tree = chosen_start_tree_newick
                base_random_tree = random_tree
            else:
                base_random_tree_raw, random_tree_raw = self.sample_random_tree_with_base(
                    sample_source_tree
                )
                base_random_tree = _remap_random_tree_to_dataset_indexing(
                    base_random_tree_raw
                )
                random_tree = _remap_random_tree_to_dataset_indexing(random_tree_raw)

            chosen_target_tree_newick = (
                str(forced_target_tree_newick)
                if forced_target_tree_newick is not None
                else None
            )
            if (
                chosen_target_tree_newick is None
                and self.overfit_fixed_pair
                and self.overfit_fixed_pair_target_tree_newick
            ):
                chosen_target_tree_newick = str(
                    self.overfit_fixed_pair_target_tree_newick
                )

            target_tree_newick = (
                chosen_target_tree_newick
                if chosen_target_tree_newick is not None
                else real_tree_newick
            )
            cache_key = None
            if chosen_start_tree_newick is not None:
                cache_key = (
                    random_tree,
                    target_tree_newick,
                    int(self.overfit_boundary_prefix_k),
                    int(self.overfit_start_boundary_prefix_k),
                    int(self.overfit_event_prefix_count),
                    bool(self.overfit_split_multi_subset_events),
                )
                cached_pair = self._cached_overfit_bank_pairs_by_key.get(cache_key)
                if cached_pair is not None:
                    return dict(cached_pair)

            # Both trees now use "0".."N-1" names, so bhv utils will work happily
            effective_target_tree = self.resolve_training_target_tree(
                random_tree,
                target_tree_newick,
                base_start_tree_newick=base_random_tree,
            )
            boundary_paths = return_tree_boundary_merge_paths(
                random_tree,
                effective_target_tree,
                legacy_training_semantics=False,
            )
            final_labels = return_sampled_tree_boundary_decisions(
                random_tree,
                effective_target_tree,
                split_multi_label_events=self.overfit_split_multi_subset_events,
                legacy_training_semantics=False,
            )
            allow_velocity_only_pair = bool(boundary_paths) and not final_labels

            # If final_labels is empty, resample random tree until we get valid labels
            while not final_labels and not allow_velocity_only_pair:
                if chosen_start_tree_newick is not None:
                    raise ValueError(
                        "Configured overfit_fixed_pair_start_tree_newick did not "
                        "yield any valid boundary decisions."
                    )
                base_random_tree_raw, random_tree_raw = self.sample_random_tree_with_base(
                    sample_source_tree
                )
                base_random_tree = _remap_random_tree_to_dataset_indexing(
                    base_random_tree_raw
                )
                random_tree = _remap_random_tree_to_dataset_indexing(random_tree_raw)
                target_tree_newick = (
                    str(forced_target_tree_newick)
                    if forced_target_tree_newick is not None
                    else real_tree_newick
                )
                effective_target_tree = self.resolve_training_target_tree(
                    random_tree,
                    target_tree_newick,
                    base_start_tree_newick=base_random_tree,
                )
                boundary_paths = return_tree_boundary_merge_paths(
                    random_tree,
                    effective_target_tree,
                    legacy_training_semantics=False,
                )
                final_labels = return_sampled_tree_boundary_decisions(
                    random_tree,
                    effective_target_tree,
                    split_multi_label_events=self.overfit_split_multi_subset_events,
                    legacy_training_semantics=False,
                )
                allow_velocity_only_pair = bool(boundary_paths) and not final_labels

            pair = {
                "base_random_tree": base_random_tree,
                "random_tree": random_tree,
                "effective_target_tree": effective_target_tree,
                "boundary_paths": boundary_paths,
                "final_labels": final_labels,
            }
            if cache_key is not None:
                self._cached_overfit_bank_pairs_by_key[cache_key] = dict(pair)
            return pair

        if self.overfit_fixed_pair:
            pair = None

            if topology_stream_selection is not None:
                pair = _build_pair(
                    forced_start_tree_newick=topology_stream_selection.get(
                        "forced_start_tree_newick"
                    ),
                    forced_target_tree_newick=topology_stream_selection.get(
                        "forced_target_tree_newick"
                    ),
                )
                pair["bank_group_key"] = topology_stream_selection.get("bank_group_key")
                pair["dataset_id"] = topology_stream_selection.get("dataset_id")
                if topology_stream_extra_velocity_samples:
                    pair["extra_velocity_samples"] = list(
                        topology_stream_extra_velocity_samples
                    )

            if forced_bank_selection is not None:
                pair = _build_pair(
                    forced_start_tree_newick=forced_bank_selection.get(
                        "forced_start_tree_newick"
                    ),
                    forced_target_tree_newick=forced_bank_selection.get(
                        "forced_target_tree_newick"
                    ),
                )
                pair["bank_group_key"] = forced_bank_selection.get("bank_group_key")
                if forced_bank_selection.get("source_group_key") is not None:
                    pair["source_group_key"] = forced_bank_selection.get(
                        "source_group_key"
                    )
                if forced_bank_selection.get("dataset_id") is not None:
                    pair["dataset_id"] = forced_bank_selection.get("dataset_id")
                if "oracle_prefix_start_tree" in forced_bank_selection:
                    pair["oracle_prefix_start_tree"] = forced_bank_selection[
                        "oracle_prefix_start_tree"
                    ]
                    pair["oracle_prefix_base_start_tree"] = forced_bank_selection[
                        "oracle_prefix_base_start_tree"
                    ]
                    pair["oracle_prefix_target_tree"] = forced_bank_selection[
                        "oracle_prefix_target_tree"
                    ]

            if pair is None and len(self.overfit_fixed_pair_target_tree_newick_bank) > 1:
                selection_cache_key = None
                selection = None
                if (
                    self.overfit_fixed_pair_cache_virtual_index_selection
                    and self.overfit_virtual_epoch_size is not None
                ):
                    selection_cache_key = int(requested_index)
                    cached_selection = (
                        self._cached_overfit_bank_selection_by_virtual_index.get(
                            selection_cache_key
                        )
                    )
                    if cached_selection is not None:
                        selection = dict(cached_selection)

                if selection is None:
                    selection = self._sample_overfit_fixed_pair_bank_selection(
                        allow_oracle_prefix=(
                            not self.validation
                            and self.overfit_oracle_prefix_start_prob > 0.0
                            and random.random() < self.overfit_oracle_prefix_start_prob
                        ),
                        dataset_id=meta.get("id"),
                    )
                    if selection is not None and selection_cache_key is not None:
                        self._cached_overfit_bank_selection_by_virtual_index[
                            selection_cache_key
                        ] = dict(selection)

                if selection is not None:
                    pair = _build_pair(
                        forced_start_tree_newick=selection.get(
                            "forced_start_tree_newick"
                        ),
                        forced_target_tree_newick=selection.get(
                            "forced_target_tree_newick"
                        ),
                    )
                    pair["bank_group_key"] = selection.get("bank_group_key")
                    if selection.get("source_group_key") is not None:
                        pair["source_group_key"] = selection.get("source_group_key")
                    if selection.get("dataset_id") is not None:
                        pair["dataset_id"] = selection.get("dataset_id")
                    if "oracle_prefix_start_tree" in selection:
                        pair["oracle_prefix_start_tree"] = selection[
                            "oracle_prefix_start_tree"
                        ]
                        pair["oracle_prefix_base_start_tree"] = selection[
                            "oracle_prefix_base_start_tree"
                        ]
                        pair["oracle_prefix_target_tree"] = selection[
                            "oracle_prefix_target_tree"
                        ]
            elif pair is None and len(self.overfit_fixed_pair_start_tree_newick_bank) > 1:
                pair_bank = self._cached_overfit_pair_banks.get(index)
                if pair_bank is None:
                    random_state = random.getstate()
                    try:
                        random.seed(13)
                        pair_bank = [
                            _build_pair(forced_start_tree_newick=start_tree_newick)
                            for start_tree_newick in self.overfit_fixed_pair_start_tree_newick_bank
                        ]
                    finally:
                        random.setstate(random_state)
                    self._cached_overfit_pair_banks[index] = pair_bank
                pair = random.choice(pair_bank)
            elif pair is None:
                pair = self._cached_overfit_pairs.get(index)
                if pair is None:
                    random_state = random.getstate()
                    try:
                        random.seed(13)
                        pair = _build_pair()
                    finally:
                        random.setstate(random_state)
                    self._cached_overfit_pairs[index] = pair
        else:
            pair = _build_pair()

        base_random_tree = pair["base_random_tree"]
        random_tree = pair["random_tree"]
        effective_target_tree = pair["effective_target_tree"]
        boundary_paths = pair["boundary_paths"]
        final_labels = pair["final_labels"]
        sample_dataset_id = str(pair.get("dataset_id", meta["id"])).upper()
        if selected_sequences is None or selected_sequence_names is None:
            selected_sequences = [
                new_seqs.get(str(idx), "") for idx in range(int(current_size))
            ]
            selected_sequence_names = [
                str(original_names_map.get(str(idx), idx))
                for idx in range(int(current_size))
            ]
        pair["name_mapping"] = original_names_map
        pair["selected_sequences"] = list(selected_sequences)
        pair["selected_sequence_names"] = list(selected_sequence_names)

        if len(final_labels) == 0:
            velocity_next_boundary_tree = (
                str(boundary_paths[0]["start_newick"])
                if boundary_paths
                else str(effective_target_tree)
            )
            newick, velocity = return_sampled_tree_orthant_velocity(
                random_tree,
                effective_target_tree,
                0.0,
                legacy_training_semantics=False,
            )
            sample = {
                "id": meta["id"],
                "nexus_path": meta["nexus_path"],
                "tree_paths": meta["tree_paths"],
                "sequences": new_seqs,
                "taxa_order": list(new_seqs.keys()),
                "start_tree": random_tree,
                "newick_tree": newick,
                "target_tree": effective_target_tree,
                "fixed_pair_num_events": 0,
                "velocity": velocity,
                "velocity_next_boundary_tree": velocity_next_boundary_tree,
                "timepoint": 0.0,
                "autoregressive_newick": velocity_next_boundary_tree,
                "autoregressive_labels": [],
                "autoregressive_stop_after_merge": True,
                "autoregressive_event_index": -1,
                "autoregressive_newick_time": 0.0,
                "num_to_name": original_names_map,
                "seq_ordering_map": seq_ordering_map,
                "selected_sequences": list(selected_sequences),
                "selected_sequence_names": list(selected_sequence_names),
                "dataset_id": sample_dataset_id,
            }
            if "bank_group_key" in pair:
                sample["bank_group_key"] = pair["bank_group_key"]
            if "oracle_prefix_start_tree" in pair:
                sample["oracle_prefix_start_tree"] = pair["oracle_prefix_start_tree"]
                sample["oracle_prefix_base_start_tree"] = pair[
                    "oracle_prefix_base_start_tree"
                ]
                sample["oracle_prefix_target_tree"] = pair["oracle_prefix_target_tree"]
            if self.overfit_full_path_control_mode:
                full_path_groups = self._attach_batch_metadata_to_full_path_samples(
                    self._build_full_path_control_samples(pair),
                    dataset_id=sample_dataset_id,
                    sample_id=meta["id"],
                    num_to_name=original_names_map,
                    seq_ordering_map=seq_ordering_map,
                    selected_sequences=selected_sequences,
                    selected_sequence_names=selected_sequence_names,
                )
                (
                    sample["full_path_velocity_samples"],
                    sample["full_path_autoregressive_samples"],
                    sample["full_path_terminal_samples"],
                ) = full_path_groups
                sample["_full_path_control_mode"] = True
            return sample

        horizon = min(self.overfit_event_horizon, len(final_labels))
        max_start_index = max(0, len(final_labels) - horizon)
        random_index = random.randint(0, max_start_index)

        def _build_step_sample(event_index: int) -> Dict[str, Any]:
            chosen_autoregressive_event = final_labels[event_index]
            autoregressive_time = (
                0.0
                if len(final_labels) <= 1
                else event_index / float(len(final_labels) - 1)
            )
            velocity_next_boundary_tree = None
            if self.overfit_velocity_explicit_boundary_end_states:
                explicit_velocity_trees = [random_tree]
                explicit_velocity_trees.extend(
                    path["end_newick"] for path in boundary_paths[:-1]
                )
                if self.overfit_velocity_fixed_timepoints:
                    explicit_velocity_timepoints = list(
                        self.overfit_velocity_fixed_timepoints
                    )
                    if len(explicit_velocity_timepoints) == 1:
                        explicit_velocity_timepoints = explicit_velocity_timepoints * len(
                            explicit_velocity_trees
                        )
                else:
                    explicit_velocity_timepoints = [0.0]
                    explicit_velocity_timepoints.extend(
                        float(path["global_time"]) for path in boundary_paths[:-1]
                    )
                if len(explicit_velocity_trees) != len(explicit_velocity_timepoints):
                    raise ValueError(
                        "Explicit boundary-end velocity supervision requires one "
                        "global timepoint per orthant-start state. "
                        f"Got {len(explicit_velocity_timepoints)} "
                        f"timepoints for {len(explicit_velocity_trees)} states."
                    )

                explicit_velocity_options = list(
                    zip(
                        explicit_velocity_trees,
                        [path["start_newick"] for path in boundary_paths],
                        explicit_velocity_timepoints,
                    )
                )
                (
                    velocity_source_tree,
                    velocity_next_boundary_tree,
                    model_timepoint,
                ) = random.choice(
                    explicit_velocity_options
                )
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    0.0,
                    legacy_training_semantics=False,
                )
                if (
                    self.overfit_velocity_explicit_boundary_label_scale_mode
                    == "remaining"
                ):
                    scale = 1.0 / max(1.0 - float(model_timepoint), 1e-6)
                    velocity = {int(k): float(v) * scale for k, v in velocity.items()}
                timepoint = float(model_timepoint)
            elif self.overfit_velocity_orthant_start_states:
                orthant_start_trees = [random_tree]
                orthant_start_trees.extend(
                    path["end_newick"] for path in boundary_paths[:-1]
                )
                next_boundary_trees = [
                    path.get("start_newick", path["end_newick"])
                    for path in boundary_paths
                ]
                if all("start_newick" in path for path in boundary_paths):
                    (
                        velocity_source_tree,
                        velocity_next_boundary_tree,
                    ) = random.choice(
                        list(zip(orthant_start_trees, next_boundary_trees))
                    )
                else:
                    velocity_source_tree = random.choice(orthant_start_trees)
                    source_index = orthant_start_trees.index(velocity_source_tree)
                    velocity_next_boundary_tree = next_boundary_trees[source_index]
                timepoint = 0.0
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    timepoint,
                    legacy_training_semantics=False,
                )
            elif self.overfit_velocity_event_states:
                velocity_source_tree = chosen_autoregressive_event["newick"]
                timepoint = 0.0
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    timepoint,
                    legacy_training_semantics=False,
                )
            else:
                velocity_source_tree = random_tree
                if self.overfit_velocity_fixed_timepoints is not None:
                    timepoint = float(random.choice(self.overfit_velocity_fixed_timepoints))
                elif self.overfit_velocity_zero:
                    timepoint = 0.0
                else:
                    timepoint = random.uniform(0, 1)
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    timepoint,
                    legacy_training_semantics=False,
                )

            step_sample = {
                "id": meta["id"],
                "nexus_path": meta["nexus_path"],
                "tree_paths": meta["tree_paths"],
                "sequences": new_seqs,
                "taxa_order": list(new_seqs.keys()),
                "start_tree": random_tree,
                "newick_tree": newick,
                "target_tree": effective_target_tree,
                "fixed_pair_num_events": int(len(final_labels)),
                "velocity": velocity,
                "velocity_next_boundary_tree": velocity_next_boundary_tree,
                "timepoint": timepoint,
                "autoregressive_newick": chosen_autoregressive_event["newick"],
                "autoregressive_labels": chosen_autoregressive_event["labels"],
                "autoregressive_stop_after_merge": bool(
                    chosen_autoregressive_event.get("stop_after_merge", False)
                ),
                "autoregressive_event_index": int(event_index),
                "autoregressive_newick_time": autoregressive_time,
                "num_to_name": original_names_map,
                "seq_ordering_map": seq_ordering_map,
                "selected_sequences": list(selected_sequences),
                "selected_sequence_names": list(selected_sequence_names),
                "dataset_id": sample_dataset_id,
            }
            if "bank_group_key" in pair:
                step_sample["bank_group_key"] = pair["bank_group_key"]
            if "source_group_key" in pair:
                step_sample["source_group_key"] = pair["source_group_key"]
            return step_sample

        step_samples = [
            _build_step_sample(event_index)
            for event_index in range(random_index, random_index + horizon)
        ]

        num_to_name = self.return_nexus_number_to_name(index)
        sample = dict(step_samples[0])
        sample["num_to_name"] = original_names_map
        sample["dataset_id"] = sample_dataset_id
        sample["selected_sequences"] = list(selected_sequences)
        sample["selected_sequence_names"] = list(selected_sequence_names)
        if "bank_group_key" in pair:
            sample["bank_group_key"] = pair["bank_group_key"]
        if "oracle_prefix_start_tree" in pair:
            sample["oracle_prefix_start_tree"] = pair["oracle_prefix_start_tree"]
            sample["oracle_prefix_base_start_tree"] = pair[
                "oracle_prefix_base_start_tree"
            ]
            sample["oracle_prefix_target_tree"] = pair["oracle_prefix_target_tree"]
        sample["seq_ordering_map"] = seq_ordering_map
        if len(step_samples) > 1:
            sample["multi_step_samples"] = step_samples
        if self.overfit_full_path_control_mode:
            full_path_groups = self._attach_batch_metadata_to_full_path_samples(
                self._build_full_path_control_samples(pair),
                dataset_id=sample_dataset_id,
                sample_id=meta["id"],
                num_to_name=original_names_map,
                seq_ordering_map=seq_ordering_map,
                selected_sequences=selected_sequences,
                selected_sequence_names=selected_sequence_names,
            )
            (
                sample["full_path_velocity_samples"],
                sample["full_path_autoregressive_samples"],
                sample["full_path_terminal_samples"],
            ) = full_path_groups
            sample["_full_path_control_mode"] = True

        return sample

    def get_overfit_fixed_pair(self, index: int) -> Optional[Dict[str, Any]]:
        if not self.overfit_fixed_pair:
            return None
        if len(self.overfit_fixed_pair_target_tree_newick_bank) > 1:
            return None
        if len(self.overfit_fixed_pair_start_tree_newick_bank) > 1:
            if index not in self._cached_overfit_pair_banks:
                _ = self[index]
            pair_bank = self._cached_overfit_pair_banks.get(index)
            if pair_bank:
                return pair_bank[0]
            return None
        if index not in self._cached_overfit_pairs:
            _ = self[index]
        return self._cached_overfit_pairs.get(index)

    def sample_overfit_fixed_pair_bank_pair(
        self,
        dataset_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.overfit_fixed_pair:
            return None
        chosen_start_item, chosen_target_item = (
            self._sample_matching_overfit_fixed_pair_bank_items(dataset_id=dataset_id)
        )
        if chosen_start_item is None or chosen_target_item is None:
            return None

        chosen_start_tree = str(chosen_start_item["tree"])
        chosen_target_tree = str(chosen_target_item["tree"])
        start_payload = chosen_start_item.get("payload") or {}
        target_payload = chosen_target_item.get("payload") or {}
        base_random_tree = str(chosen_start_tree)
        random_tree = str(chosen_start_tree)
        effective_target_tree = self.resolve_training_target_tree(
            random_tree,
            str(chosen_target_tree),
            base_start_tree_newick=base_random_tree,
        )
        boundary_paths = return_tree_boundary_merge_paths(
            random_tree,
            effective_target_tree,
            legacy_training_semantics=False,
        )
        final_labels = return_sampled_tree_boundary_decisions(
            random_tree,
            effective_target_tree,
            split_multi_label_events=self.overfit_split_multi_subset_events,
            legacy_training_semantics=False,
        )
        pair = {
            "base_random_tree": base_random_tree,
            "random_tree": random_tree,
            "effective_target_tree": effective_target_tree,
            "boundary_paths": boundary_paths,
            "final_labels": final_labels,
            "bank_group_key": str(
                chosen_target_item.get(
                    "group_key",
                    chosen_start_item.get("group_key", f"n{_leaf_count_from_newick(chosen_target_tree)}"),
                )
            ),
        }
        source_group_key = (
            target_payload.get("source_group_key")
            or start_payload.get("source_group_key")
            or target_payload.get("original_group_key")
            or start_payload.get("original_group_key")
        )
        if source_group_key is not None:
            pair["source_group_key"] = str(source_group_key)
        item_dataset_id = (
            chosen_target_item.get("dataset_id")
            or chosen_start_item.get("dataset_id")
            or target_payload.get("dataset_id")
            or start_payload.get("dataset_id")
        )
        if item_dataset_id is not None:
            pair["dataset_id"] = str(item_dataset_id).upper()
        return pair

    def parse_translate_block(self, path: str) -> Dict[str, str]:
        """Extract 'translate' block from a Nexus/MrBayes file to map IDs to Taxon names."""
        mapping = {}
        in_translate = False
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for start of translate block
                    if not in_translate:
                        if line.lower().startswith("translate"):
                            in_translate = True
                            # Remove 'translate' keyword to process rest of line
                            line = line[9:].strip()
                            if not line:
                                continue

                    if in_translate:
                        # Parsing entries like: 1 Marmota_marmota, 2 Jaculus, ...
                        # Ends with ;
                        term = False
                        if ";" in line:
                            term = True
                            line = line.replace(";", "")

                        # Split by comma
                        tokens = line.split(",")
                        for token in tokens:
                            token = token.strip()
                            if not token:
                                continue
                            parts = token.split()
                            if len(parts) >= 2:
                                # mapping ID -> Name
                                mapping[parts[0]] = parts[1]

                        if term:
                            break
        except Exception:
            # If parsing fails or file not found, return empty dict
            pass
        return mapping

    def return_max_length(self, name_to_seq):
        if not name_to_seq:
            return 0
        return max(len(s) for s in name_to_seq.values())

    def sample_random_tree(self, real_tree, subtree_size: Optional[int] = None):
        """
        real_tree: Newick string or an ETE Tree.
        Returns: Newick string for a random tree with the same leaf names.
        """
        if self.overfit_start_boundary_prefix_k >= 0:
            original_prefix = self.overfit_start_boundary_prefix_k
            self.overfit_start_boundary_prefix_k = -1
            try:
                base_random_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
            finally:
                self.overfit_start_boundary_prefix_k = original_prefix
            target_tree_newick = real_tree if isinstance(real_tree, str) else real_tree.write(format=1)
            return resolve_training_target_tree_for_prefix(
                base_random_tree,
                target_tree_newick,
                original_prefix,
            )

        ###DEBUG PURPOSES ONLY RETURN THE SAME TREE###
        # if self.random_tree is not None:
        #     return self.random_tree

        # Parse to ETE
        if isinstance(real_tree, str):
            t = EteTree(real_tree, format=1)
        else:
            t = real_tree

        # Collect leaf names; order however you like (here: sorted for determinism)
        leaves = t.get_leaves()
        leaves_sorted = sorted(leaves, key=lambda x: _numeric_name_sort_key(x.name))
        n_leaves = len(leaves_sorted)

        # Build a random unrooted binary tree on {1,...,n_leaves}
        # Random tree creates leaves 0..n_leaves-1
        rt = Tree(num_leaves=n_leaves, random=True)

        # Map 0..n_leaves-1 back to the sorted real leaf names
        for i, real_leaf in enumerate(leaves_sorted):
            rt.id_to_name[i] = real_leaf.name

        # Produce Newick with the same taxa names but random topology/lengths
        random_newick = str(rt)
        # if self.random_tree is None:
        #     self.random_tree = random_newick
        if self.sanity_check:
            return '((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,51:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,45:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);'
        elif self.random_sanity_check:
            # Return a fixed random tree for sanity checking
            #return '((((((((((((((((((((114:0.10647175508658419,138:0.16166919943341312):0.16066029529927675,((((116:0.5627405078277302,128:0.9121985789594937):0.24817340926137418,(31:0.7138600194795188,70:0.27733161619156094):0.4220283090253836):0.13590539030009635,45:0.9266067111065456):0.41469258200132686,97:0.9567996962072277):0.3973856401724488):0.12827166224771633,93:0.3554077644937247):0.4654676518823466,82:0.6121947293792284):0.15119162175581607,136:0.5780135217434872):0.681561776165486,((((121:0.8205332597755289,26:0.4976519606747656):0.7351007020839916,50:0.16247867919653797):0.26513923220851,(((((21:0.6708560025782748,61:0.985812264384931):0.1457190042962776,123:0.887037358663522):0.16191218265401736,148:0.41410795146806356):0.2177904539147108,((144:0.9361164985349627,69:0.718171432223074):0.2006864925304015,27:0.6048161782972717):0.6809507099538845):0.5115021153134068,57:0.622530760364837):0.5163652615608707):0.22616406738529704,(((32:0.4183107711531565,56:0.3350936767511892):0.9331874403126009,35:0.7745298500291854):0.3130635990064228,67:0.5679283137099269):0.7002400695719887):0.1545514066693123):0.7553922295970644,83:0.8503526523736518):0.7311503693259214,41:0.24327805868265814):0.15263496280656946,133:0.29593080857275744):0.7621634609101391,(77:0.3935646143716599,9:0.5600565971871284):0.7002036136951567):0.24542377864649756,25:0.9818181584459523):0.8756372817731165,139:0.16116477703818152):0.5985293701820318,113:0.118930834006353):0.6295961029693159,154:0.33801932933841494):0.8941447676266638,(((118:0.3186975181561521,38:0.2614245970763863):0.9173431415967147,20:0.5548681522881885):0.32600912686290473,23:0.13967808561914502):0.6185904847488098):0.3231150522994072,((((((((((((140:0.17378506494538792,142:0.7027533772928463):0.6210271517564896,(((103:0.20068095278880554,22:0.8464132804349973):0.27243106492748864,122:0.1983421290203236):0.58265717851931,84:0.3734874543415184):0.13576749606015798):0.5749064086764286,(109:0.9679266154067097,63:0.9095542596200796):0.645934525734366):0.3167844332025098,((24:0.21773769463153203,59:0.6369141294889702):0.4002008847445794,105:0.758716835360165):0.1708813434622295):0.6106212493200189,19:0.4085922590139328):0.6390501260142061,((((((((102:0.9301906725564473,53:0.7115663065242372):0.6930844745631968,(((12:0.4699920484447643,129:0.1019398405162838):0.778208953617708,126:0.18393649856545236):0.3127415414875745,((78:0.5989712299001143,79:0.27265608369113276):0.6984364907914944,94:0.6974684526744989):0.7274875727749591):0.41886780993059847):0.35062953756809234,((117:0.7510173092928738,66:0.4924517639094751):0.4586193807083052,49:0.6940298644737661):0.5853105382552791):0.5368730492354417,48:0.30063119248522807):0.9581497728881525,1:0.7477126019973318):0.24539413217783806,(((55:0.8893345704180042,73:0.6970487286389898):0.1248404816076723,15:0.3722654286810009):0.5888961660354282,145:0.10838391119582176):0.942160858700341):0.3822443287927564,134:0.15486181432886426):0.14512810353755823,33:0.6038951672294827):0.5397451372260531):0.926644144956132,11:0.9360221437019591):0.24234197891020276,(((((((((104:0.3155070793042529,81:0.25662460888894956):0.3535806443121049,17:0.7681750155828988):0.23990311879749143,18:0.28956668932624874):0.7574547856739193,((131:0.6956841807556707,72:0.6608398689517683):0.416031516174477,51:0.7959939736963249):0.8050253756358585):0.986789241049431,((((((((37:0.47106496931841757,43:0.5236210189773913):0.25934713328834313,150:0.576046623342043):0.985467979226334,((101:0.43652246733203315,137:0.5570456472940767):0.8662082908097093,62:0.20408120562639742):0.826389948938628):0.5332225761739355,127:0.46577265215722485):0.8781853546271939,151:0.7010768833413252):0.9274484177488677,135:0.5291096653980414):0.8575024133552204,(((((141:0.33468912604326956,6:0.41111840051182025):0.8077853603833137,85:0.28916441639437773):0.47054184989639836,4:0.8041478465328685):0.6087042314596806,((10:0.9746992889831029,92:0.7623653013461995):0.6929656704985796,95:0.657334724826138):0.3847328613790544):0.6916358480252702,(2:0.8676215338527076,96:0.1840484800680064):0.7187474159211975):0.27901999158825974):0.5568965937798708,(147:0.21921091332982057,68:0.8258597827300935):0.5552962246438888):0.8260973243174158):0.42668269046284024,((((((((((119:0.184893216842694,52:0.5584651079752336):0.33905306127821844,143:0.38276547651601844):0.18811785538144032,(125:0.11328207084608549,47:0.8841907322767074):0.6869304045986698):0.5865291442902746,76:0.5564102536317422):0.7633339241412703,7:0.946236808357798):0.46635396533734297,107:0.6937757404726307):0.5256756486501344,64:0.27107089705206644):0.14875406850895348,120:0.8774530735326429):0.6881099394392124,65:0.139947880341908):0.24861426307671491,108:0.9521638117344918):0.5794355525765641):0.2131844603683972,(86:0.6459954134317863,91:0.9638488033147325):0.8071829447298221):0.27090808054087345,132:0.5380042704888965):0.7045182636744548,((111:0.9096526761786509,80:0.773321104143001):0.819898326916207,36:0.24625159039039476):0.43607845227636444):0.10292078449155637):0.7498828551789197,((((14:0.6127078691243664,90:0.8143618591954018):0.9474365145061804,13:0.8066164211147094):0.4633979961864497,30:0.9749766385155344):0.4513795818290318,((((153:0.8903142684762329,44:0.2796155470231767):0.9385804354924697,(146:0.5838709636914646,98:0.3113817386740156):0.19322785172639034):0.6590433299437836,(((115:0.15298639018869106,58:0.9851955155947784):0.13443147761793645,89:0.7080645098710899):0.3980348813929664,54:0.8592950469448727):0.6066942810750514):0.18998312403212964,75:0.9100421515873822):0.9338662662034):0.7370568518936087):0.9801860443096008,106:0.8344209280780797):0.7857659791468582,(100:0.6627701742097758,112:0.5063375272520168):0.5569135071438928):0.1957697405179345,5:0.3507427549207035):0.15762330669371044):0.779738676972225,((((29:0.13764597235197207,88:0.49760741515415763):0.13485214689920783,42:0.4442875503372409):0.5972725870704979,((((149:0.4943661157069389,152:0.59914205424071):0.31985597485731826,110:0.4590930699976863):0.35648865209499103,74:0.212162784097931):0.4881676596624882,(71:0.9286937705962361,8:0.4412307494102876):0.9539870373741829):0.48680804159269464):0.4322943860609726,39:0.9320385657173911):0.7416540755372096):0.7208808688896593,((((((40:0.32335051700566064,99:0.37937751786542373):0.7615517849450885,130:0.7269323532401463):0.7029164163593771,87:0.39002430676288435):0.9397792094685887,3:0.42818636379453046):0.9505458451419752,((34:0.9764533203735601,46:0.8384827428744058):0.887241451186982,28:0.13242532425474757):0.7219176613802888):0.23414885693066756,60:0.4927544954835472):0.35672430605222827):0.16784639953686042,(124:0.8703702032215355,16:0.9007414363051383):0.8272200202585324):0.0,0:0.0);'
            #return '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
            return '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
        return random_newick

    def extract_newick_from_line(self, line: str) -> str:
        """
        Given a line from a .t/.trees file, extract the Newick string.
        Handles BEAST-style 'tree STATE_... = [&R] (..);' or raw '(..);'.
        Returns '' if no Newick found.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return ""

        # Find first '(' and last ')' or ';'
        start = line.find("(")
        if start == -1:
            return ""

        # Newick typically ends at ';', but sometimes there's stuff after.
        # We'll go to the last ';' if it exists, else end of line.
        end = line.rfind(";")
        if end == -1:
            end = len(line)
        else:
            end = end + 1  # include ';'

        newick = line[start:end].strip()
        return newick if newick else ""

    def extract_tree_weight_from_line(self, line: str) -> Optional[float]:
        """Extract posterior tree probability from a .trprobs tree line."""
        match = re.search(r"p\s*=\s*([0-9.eE+-]+)", line)
        if match:
            return float(match.group(1))
        match = re.search(r"&W\s*([0-9.eE+-]+)", line)
        if match:
            return float(match.group(1))
        return None

    def _expand_trprobs_trees(
        self,
        trees: List[str],
        weights: List[Optional[float]],
    ) -> List[str]:
        sample_count = int(self.trprobs_sample_count_per_file)
        if sample_count <= 0 or not trees:
            return list(trees)

        clean_weights = [
            max(0.0, float(weight)) if weight is not None else 0.0
            for weight in weights
        ]
        total_weight = sum(clean_weights)
        if total_weight <= 0.0:
            return list(trees)

        scaled_counts = [
            (weight / total_weight) * sample_count for weight in clean_weights
        ]
        counts = [int(math.floor(count)) for count in scaled_counts]
        remaining = sample_count - sum(counts)
        if remaining > 0:
            remainder_order = sorted(
                range(len(scaled_counts)),
                key=lambda idx: (scaled_counts[idx] - counts[idx], clean_weights[idx]),
                reverse=True,
            )
            for idx in remainder_order[:remaining]:
                counts[idx] += 1

        expanded: List[str] = []
        for tree, count in zip(trees, counts):
            if count > 0:
                expanded.extend([tree] * count)
        return expanded or list(trees)

    def load_posterior_trees_from_tfiles(
        self,
        tree_files: List[str],
        burn_in_fraction: float = 0.25,
    ) -> List[str]:
        """
        Given a list of .t/.trees files, extract posterior Newick trees
        applying a per-file burn-in and thinning.

        Args
        ----
        tree_files : list of paths to .t files
        burn_in_fraction : fraction of samples per file to discard as burn-in

        Returns
        -------
        trees : list of Newick strings (posterior samples)
        """
        cache_key = (
            tuple(str(path) for path in tree_files),
            float(burn_in_fraction),
            int(self.trprobs_sample_count_per_file),
            bool(self.sanity_check),
            bool(self.random_sanity_check),
        )
        cached = self._cached_posterior_trees_by_key.get(cache_key)
        if cached is not None:
            return list(cached)

        all_trees = []

        for path in tree_files:
            file_trees = []
            file_weights = []
            is_trprobs = str(path).lower().endswith(".trprobs")

            with open(path, "r") as f:
                for line in f:
                    newick = self.extract_newick_from_line(line)
                    if newick:
                        file_trees.append(newick)
                        if is_trprobs:
                            file_weights.append(self.extract_tree_weight_from_line(line))

            if not file_trees:
                continue

            if is_trprobs:
                kept = self._expand_trprobs_trees(file_trees, file_weights)
            else:
                # Apply burn-in per raw MCMC sample file. .trprobs files are already
                # summarized and sorted by posterior probability, so burn-in would
                # incorrectly discard the high-probability trees.
                burn = int(len(file_trees) * burn_in_fraction)
                kept = file_trees[burn:]

            all_trees.extend(kept)
        if self.sanity_check or self.random_sanity_check:
            fixed = ['((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);']
            self._cached_posterior_trees_by_key[cache_key] = list(fixed)
            return fixed
        self._cached_posterior_trees_by_key[cache_key] = list(all_trees)
        return all_trees

    def parse_nexus(self, path: str) -> tuple[Dict[str, str], List[str]]:
        """Parse sequences from a NEXUS alignment file.

        Returns a dict mapping taxon/sequence ID to its sequence string.
        This lightweight parser targets common cases:
        - MATRIX block under BEGIN DATA/CHARACTERS
        - Interleaved or non-interleaved; sequence chunks are concatenated
        - Comments in square brackets are stripped

        Note: For complex/edge-case NEXUS dialects, consider using Biopython.
        """
        taxa_order = []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Remove NEXUS comments [ ... ] (non-nested) across lines
        text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)

        lines = [ln.strip() for ln in text.splitlines()]
        seqs: Dict[str, str] = {}
        in_matrix = False

        for raw in lines:
            line = raw
            if not in_matrix:
                # Look for the 'MATRIX' keyword (case-insensitive)
                idx = line.lower().find("matrix")
                if idx == -1:
                    continue

                # Switch to matrix mode; process any remainder on the same line
                in_matrix = True
                remainder = line[idx + len("matrix") :].strip()
                if remainder:
                    # Process potential inline first entry after MATRIX
                    term = False
                    if ";" in remainder:
                        remainder, _sep, _after = remainder.partition(";")
                        term = True
                    tokens = remainder.split()
                    if len(tokens) >= 2:
                        name = tokens[0]
                        if name not in taxa_order:
                            taxa_order.append(name)
                        seq = "".join(tokens[1:])
                        seqs[name] = seqs.get(name, "") + seq
                    if term:
                        break
                continue

            # In MATRIX: accumulate lines until a ';'
            if not line:
                continue

            terminated = False
            if ";" in line:
                line, _sep, _after = line.partition(";")
                terminated = True

            line = line.strip()
            if not line:
                if terminated:
                    break
                continue

            tokens = line.split()
            if len(tokens) >= 2:
                name = tokens[0]
                if name not in taxa_order:
                    taxa_order.append(name)
                seq = "".join(tokens[1:])
                seqs[name] = seqs.get(name, "") + seq
            # Lines with fewer than 2 tokens are ignored

            if terminated:
                break

        unaligned_seqs = {}
        for i in seqs:
            unaligned_seqs[i] = seqs[i].replace("-", "")

        return unaligned_seqs, taxa_order

    def build_index(self) -> None:
        """Scan nexus_root and mrbayes_root to build ID->paths mapping.

        Strategy:
        - Accept .nex or .nexus as nexus files.
        - ID := basename without extension.
        - For each ID, look for mrbayes_root/ID directory and collect .t files.
        - Include all .t files.
        """
        if self.topology_stream_index_jsonl_path:
            self._build_topology_stream_index()
            return

        if self.posterior_trprobs_root:
            self._build_trprobs_posterior_index()
            return

        nexus_exts = {".nex", ".nexus"}
        if not os.path.isdir(self.nexus_root):
            raise Exception(f"Nexus root is not a directory: {self.nexus_root}")

        ids: List[str] = []
        id_to_nexus: Dict[str, str] = {}

        for name in os.listdir(self.nexus_root):
            base, ext = os.path.splitext(name)
            if self.filter_ids is not None and base not in self.filter_ids:
                continue
            if ext.lower() in nexus_exts:
                ids.append(base)
                id_to_nexus[base] = os.path.join(self.nexus_root, name)

        ids.sort()

        index: List[Dict[str, Any]] = []
        for id_ in ids:
            run_dir = os.path.join(self.mrbayes_root, id_)
            tree_paths: List[str] = []
            if os.path.isdir(run_dir):
                # collect .t files
                t_files = [f for f in os.listdir(run_dir) if f.endswith(".t")]
                tree_paths = [os.path.join(run_dir, f) for f in sorted(t_files)]

            meta = {
                "id": id_,
                "nexus_path": id_to_nexus[id_],
                "tree_paths": tree_paths,  # may be empty if runs missing
            }
            index.append(meta)

        self._ids = ids
        self._index = index
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

    def _build_topology_stream_index(self) -> None:
        index_path = Path(self.topology_stream_index_jsonl_path)
        if not index_path.is_file():
            raise Exception(f"Topology-stream index is not a file: {index_path}")

        index: List[Dict[str, Any]] = []
        ids: List[str] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if self.topology_stream_index_max_cases and len(index) >= self.topology_stream_index_max_cases:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                case_index = int(row.get("case_index", len(index)))
                dataset_id = str(row.get("dataset_id", "realstream")).upper()
                num_leaves = int(row.get("num_leaves", 0) or 0)
                if num_leaves <= 0:
                    raise ValueError(
                        f"Topology-stream index row {line_number} is missing num_leaves."
                    )
                if (
                    self.topology_stream_index_min_num_leaves
                    and num_leaves < self.topology_stream_index_min_num_leaves
                ):
                    continue
                if (
                    self.topology_stream_index_max_num_leaves
                    and num_leaves > self.topology_stream_index_max_num_leaves
                ):
                    continue
                start_path = str(row["start_path"])
                target_path = str(row["target_path"])
                anchors_path = row.get("anchors_path") or _derive_anchor_path_from_start_path(
                    start_path
                )
                item_id = f"realstream_case{case_index:06d}"
                if self.topology_stream_use_real_sequences:
                    nexus_path = self._topology_stream_nexus_path(dataset_id)
                    if self.topology_stream_max_input_tokens > 0:
                        seqs, source_taxa_order = (
                            self._topology_stream_sequences_for_dataset(
                                dataset_id,
                                nexus_path,
                            )
                        )
                        target_payload = json.loads(Path(target_path).read_text())
                        target_tree = EteTree(
                            _tree_from_json_payload(
                                target_payload,
                                role="target",
                            ),
                            format=1,
                        )
                        leaf_labels = [
                            str(leaf.name)
                            for leaf in sorted(
                                target_tree.get_leaves(),
                                key=lambda leaf: _numeric_name_sort_key(leaf.name),
                            )
                        ]
                        if (
                            self.topology_stream_subset_num_leaves > 0
                            and len(leaf_labels) > self.topology_stream_subset_num_leaves
                        ):
                            leaf_labels = leaf_labels[
                                : int(self.topology_stream_subset_num_leaves)
                            ]
                        input_tokens = self._topology_stream_input_tokens_for_labels(
                            leaf_labels,
                            seqs,
                            source_taxa_order,
                        )
                        if input_tokens > self.topology_stream_max_input_tokens:
                            continue
                    random_distribution = False
                else:
                    nexus_path = f"random_distribution://{dataset_id}/{item_id}"
                    random_distribution = True
                ids.append(item_id)
                index.append(
                    {
                        "id": item_id,
                        "nexus_path": nexus_path,
                        "tree_paths": [],
                        "random_distribution": random_distribution,
                        "num_leaves": int(num_leaves),
                        "topology_stream_pair": True,
                        "case_index": case_index,
                        "dataset_id": dataset_id,
                        "start_path": start_path,
                        "target_path": target_path,
                        "anchors_path": None if anchors_path is None else str(anchors_path),
                        "bank_group_key": f"realstream_case{case_index:06d}",
                    }
                )

        if not index:
            raise RuntimeError(f"No topology-stream rows found in {index_path}")
        self._ids = ids
        self._index = index
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

    def _load_topology_stream_pair_for_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = (
            str(meta.get("start_path", "")),
            str(meta.get("target_path", "")),
            str(meta.get("anchors_path", "")),
        )
        cached = self._cached_topology_stream_pair_by_key.get(cache_key)
        if cached is not None:
            return cached

        start_payload = json.loads(Path(meta["start_path"]).read_text())
        target_payload = json.loads(Path(meta["target_path"]).read_text())
        start_tree = _tree_from_json_payload(start_payload, role="start")
        target_tree = _tree_from_json_payload(target_payload, role="target")
        anchors_path = meta.get("anchors_path")
        extra_velocity_samples: List[Dict[str, Any]] = []
        if anchors_path and os.path.exists(str(anchors_path)):
            extra_velocity_samples = _load_full_path_control_extra_velocity_samples(
                str(anchors_path)
            )
        loaded = {
            "start_tree": start_tree,
            "target_tree": target_tree,
            "extra_velocity_samples": extra_velocity_samples,
        }
        self._cached_topology_stream_pair_by_key[cache_key] = loaded
        return loaded

    def _available_trprobs_dataset_ids(self) -> List[str]:
        root = Path(self.posterior_trprobs_root)
        if not root.is_dir():
            raise Exception(f"Posterior trprobs root is not a directory: {root}")
        ids = [
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("DS")
        ]
        return sorted(ids, key=_numeric_name_sort_key)

    def _collect_trprobs_paths(self, dataset_id: str) -> List[str]:
        dataset_dir = Path(self.posterior_trprobs_root) / str(dataset_id)
        if not dataset_dir.is_dir():
            raise Exception(f"Posterior dataset directory is missing: {dataset_dir}")
        return [
            str(path)
            for path in sorted(
                dataset_dir.rglob("*.trprobs"),
                key=lambda path: tuple(_numeric_name_sort_key(part) for part in path.parts),
            )
        ]

    def _taxa_order_from_tree_paths(self, tree_paths: List[str]) -> List[str]:
        if not tree_paths:
            return []

        translation = self.parse_translate_block(tree_paths[0])
        if translation:
            return [
                translation[key]
                for key in sorted(translation.keys(), key=_numeric_name_sort_key)
            ]

        first_tree = ""
        with open(tree_paths[0], "r") as handle:
            for line in handle:
                first_tree = self.extract_newick_from_line(line)
                if first_tree:
                    break
        if not first_tree:
            return []

        tree = EteTree(first_tree, format=1)
        return [
            str(leaf.name)
            for leaf in sorted(
                tree.get_leaves(), key=lambda leaf: _numeric_name_sort_key(leaf.name)
            )
        ]

    def _build_trprobs_posterior_index(self) -> None:
        ids = list(self.posterior_dataset_ids) or self._available_trprobs_dataset_ids()
        if self.filter_ids is not None:
            filter_set = {str(id_) for id_ in self.filter_ids}
            ids = [id_ for id_ in ids if id_ in filter_set]
        ids = sorted(ids, key=_numeric_name_sort_key)

        index: List[Dict[str, Any]] = []
        for id_ in ids:
            tree_paths = self._collect_trprobs_paths(id_)
            if not tree_paths:
                raise Exception(
                    f"No .trprobs files found for posterior dataset {id_} under "
                    f"{self.posterior_trprobs_root}"
                )
            taxa_order = self._taxa_order_from_tree_paths(tree_paths)
            if not taxa_order:
                raise Exception(
                    f"Could not infer taxa for posterior dataset {id_} from "
                    f"{tree_paths[0]}"
                )

            index.append(
                {
                    "id": str(id_),
                    "nexus_path": f"random_distribution://{id_}",
                    "tree_paths": tree_paths,
                    "random_distribution": True,
                    "taxa_order": taxa_order,
                }
            )

        self._ids = [str(id_) for id_ in ids]
        self._index = index
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}


class SameDatasetBatchSampler(Sampler[List[int]]):
    """Yield batches whose items share one alignment/subset context."""

    def __init__(
        self,
        dataset: TreeDataset,
        batch_size: int,
        *,
        num_batches: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.dataset_size = len(getattr(dataset, "_index", []))
        default_num_batches = len(dataset)
        self.num_batches = max(
            0,
            int(num_batches) if num_batches is not None else int(default_num_batches),
        )
        self.seed = seed
        self.shuffle = bool(shuffle)
        self._iteration = 0

    def __iter__(self):
        if self.dataset_size <= 0 or self.num_batches <= 0:
            return
        seed = 0 if self.seed is None else int(self.seed)
        rng = random.Random(f"{seed}:{self._iteration}")
        self._iteration += 1
        for group_id in range(self.num_batches):
            if self.shuffle:
                dataset_index = rng.randrange(self.dataset_size)
            else:
                dataset_index = group_id % self.dataset_size
            group_code = group_id * self.dataset_size + dataset_index
            yield [
                group_code * self.batch_size + path_slot
                for path_slot in range(self.batch_size)
            ]

    def __len__(self) -> int:
        return int(self.num_batches)


class PhylaDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for managing TreeDataset splits.

    Responsibilities:
    - prepare_data(): download / generate raw data (non-distributed)
    - setup(stage): create Train/Val/Test datasets (distributed safe)
    - train_dataloader()/val_dataloader()/test_dataloader()/predict_dataloader()
    """

    def __init__(
        self,
        config,
        train_ids: List[str],
        test_ids: List[str],
    ) -> None:
        super().__init__()
        self.nexus_dir = config["data"].get("nexus_root", "unused")
        self.mrbayes_dir = config["data"].get("mrbayes_root", "unused")
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.pin_memory = config["data"]["pin_memory"]
        self.loader_seed = config["data"].get(
            "loader_seed",
            config.get("trainer", {}).get("seed"),
        )
        self.same_dataset_batch = bool(
            config["data"].get("same_dataset_batch", False)
        )
        self.same_dataset_batches_per_epoch = config["data"].get(
            "same_dataset_batches_per_epoch"
        )
        self.full_path_precompute_tokenizer_raw_graphs = bool(
            config["data"].get(
                "full_path_precompute_tokenizer_raw_graphs",
                config["data"].get("full_path_precompute_tokenizer_graph_cache", False),
            )
        )
        self.full_path_precompute_birthset_targets = bool(
            config["data"].get("full_path_precompute_birthset_targets", False)
        )
        self.full_path_precompute_birthset_candidate_info = bool(
            config["data"].get("full_path_precompute_birthset_candidate_info", False)
        )
        self.full_path_birthset_static_pair_triple_candidates = bool(
            config["data"].get("full_path_birthset_static_pair_triple_candidates", True)
        )
        trainer_cfg = config.get("trainer", {})
        self.birthset_use_small_polytomy_enumeration = bool(
            trainer_cfg.get("birthset_use_small_polytomy_enumeration", True)
        )
        self.birthset_max_enum_components = int(
            trainer_cfg.get("birthset_max_enum_components", 12)
        )
        self.birthset_max_candidates_per_polytomy = int(
            trainer_cfg.get("birthset_max_candidates_per_polytomy", 2048)
        )
        self.birthset_proposal_pair_target_mode = str(
            trainer_cfg.get("birthset_proposal_pair_target_mode", "contained")
        )
        self.birthset_proposal_max_expansion_examples = int(
            trainer_cfg.get("birthset_proposal_max_expansion_examples", 4096)
        )
        self.birthset_proposal_max_order_seed_pairs = int(
            trainer_cfg.get("birthset_proposal_max_order_seed_pairs", 128)
        )
        self.birthset_proposal_train_topk = bool(
            trainer_cfg.get("birthset_proposal_train_topk", False)
        )
        self.full_path_use_terminal_samples = (
            float(config.get("trainer", {}).get("velocity_terminal_head_weight", 0.0))
            > 0.0
        )
        self.full_path_joint_tokenizer_batch = bool(
            config.get("trainer", {}).get(
                "training_step_joint_tokenize_velocity_ar",
                False,
            )
        )
        self.velocity_probe_direct_set_anchor_only = bool(
            config.get("trainer", {}).get(
                "velocity_probe_direct_set_anchor_only",
                False,
            )
        )

        self.train_ids = train_ids
        self.test_ids = test_ids
        posterior_dataset_id = config["data"].get(
            "posterior_dataset_id",
            config["data"].get("short_run_dataset_id"),
        )
        posterior_dataset_ids = config["data"].get(
            "posterior_dataset_ids",
            config["data"].get("short_run_dataset_ids"),
        )
        configured_posterior_ids = _coerce_id_list(posterior_dataset_ids)
        if not configured_posterior_ids:
            configured_posterior_ids = _coerce_id_list(posterior_dataset_id)
        if configured_posterior_ids:
            configured_set = set(configured_posterior_ids)
            filtered_train_ids = [
                str(id_) for id_ in self.train_ids if str(id_) in configured_set
            ]
            filtered_test_ids = [
                str(id_) for id_ in self.test_ids if str(id_) in configured_set
            ]
            if config["data"].get("train_all_configured_posterior_dataset_ids", False):
                self.train_ids = list(configured_posterior_ids)
                self.test_ids = filtered_test_ids or list(configured_posterior_ids)
            else:
                self.train_ids = filtered_train_ids or list(configured_posterior_ids)
                self.test_ids = filtered_test_ids or list(configured_posterior_ids)
        if config["data"].get("train_from_test_ids", False):
            self.train_ids = list(self.test_ids)
        posterior_trprobs_root = config["data"].get(
            "posterior_trprobs_root",
            config["data"].get("short_run_root"),
        )
        if (
            posterior_trprobs_root is None
            and (
                config["data"].get("short_run_dataset_id") is not None
                or config["data"].get("short_run_dataset_ids") is not None
            )
        ):
            posterior_trprobs_root = "/home/yektefai/30272299/short_run_data_DS1-8"
        trprobs_dataset_kwargs = {
            "posterior_trprobs_root": posterior_trprobs_root,
            "posterior_dataset_id": posterior_dataset_id,
            "posterior_dataset_ids": posterior_dataset_ids,
            "use_random_sequence_distribution": config["data"].get(
                "use_random_sequence_distribution",
                config["data"].get("random_distribution", False),
            ),
            "random_distribution_sequence_length": config["data"].get(
                "random_distribution_sequence_length", 256
            ),
            "random_distribution_sequence_seed": config["data"].get(
                "random_distribution_sequence_seed",
                config.get("trainer", {}).get("seed", 0),
            ),
            "random_distribution_alphabet": config["data"].get(
                "random_distribution_alphabet", "ACGT"
            ),
            "trprobs_sample_count_per_file": config["data"].get(
                "trprobs_sample_count_per_file", 1000
            ),
        }
        topology_stream_dataset_kwargs = {
            "topology_stream_index_jsonl_path": config["data"].get(
                "topology_stream_index_jsonl_path"
            ),
            "topology_stream_index_max_cases": config["data"].get(
                "topology_stream_index_max_cases", 0
            ),
            "topology_stream_index_min_num_leaves": config["data"].get(
                "topology_stream_index_min_num_leaves", 0
            ),
            "topology_stream_index_max_num_leaves": config["data"].get(
                "topology_stream_index_max_num_leaves", 0
            ),
            "topology_stream_subset_num_leaves": config["data"].get(
                "topology_stream_subset_num_leaves", 0
            ),
            "posterior_subset_num_leaves": config["data"].get(
                "posterior_subset_num_leaves", 0
            ),
            "posterior_subset_max_input_tokens": config["data"].get(
                "posterior_subset_max_input_tokens",
                config.get("trainer", {}).get("live_phyla_max_input_tokens", 0),
            ),
            "sample_metrics_iterate_dataset_indices": config["data"].get(
                "sample_metrics_iterate_dataset_indices", False
            ),
            "topology_stream_use_real_sequences": config["data"].get(
                "topology_stream_use_real_sequences", False
            ),
            "topology_stream_max_input_tokens": config["data"].get(
                "topology_stream_max_input_tokens", 0
            ),
        }
        topology_stream_val_dataset_kwargs = dict(topology_stream_dataset_kwargs)
        if topology_stream_val_dataset_kwargs.get("topology_stream_index_jsonl_path"):
            topology_stream_val_dataset_kwargs["topology_stream_index_max_cases"] = (
                config["data"].get("topology_stream_validation_max_cases", 1)
            )

        self.dataset_train = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.train_ids, sanity_check=config["data"].get("sanity_check", False), random_sanity_check=config["data"].get("random_sanity_check", False),
            overfit_velocity_zero=config["data"].get("overfit_velocity_zero", False),
            overfit_velocity_event_states=config["data"].get("overfit_velocity_event_states", False),
            overfit_velocity_orthant_start_states=config["data"].get(
                "overfit_velocity_orthant_start_states", False
            ),
            overfit_velocity_explicit_boundary_end_states=config["data"].get(
                "overfit_velocity_explicit_boundary_end_states", False
            ),
            overfit_velocity_fixed_timepoints=config["data"].get(
                "overfit_velocity_fixed_timepoints"
            ),
            overfit_velocity_explicit_boundary_label_scale_mode=config["data"].get(
                "overfit_velocity_explicit_boundary_label_scale_mode", "local"
            ),
            overfit_boundary_prefix_k=config["data"].get("overfit_boundary_prefix_k", -1),
            overfit_start_boundary_prefix_k=config["data"].get("overfit_start_boundary_prefix_k", -1),
            overfit_event_prefix_count=config["data"].get("overfit_event_prefix_count", -1),
            overfit_event_horizon=config["data"].get("overfit_event_horizon", 1),
            overfit_fixed_pair=config["data"].get("overfit_fixed_pair", False),
            overfit_fixed_pair_start_tree_newick=config["data"].get(
                "overfit_fixed_pair_start_tree_newick"
            ),
            overfit_fixed_pair_start_tree_json_path=config["data"].get(
                "overfit_fixed_pair_start_tree_json_path"
            ),
            overfit_fixed_pair_start_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_start_tree_json_paths"
            ),
            overfit_fixed_pair_start_tree_json_dir=config["data"].get(
                "overfit_fixed_pair_start_tree_json_dir"
            ),
            overfit_fixed_pair_target_tree_newick=config["data"].get(
                "overfit_fixed_pair_target_tree_newick"
            ),
            overfit_fixed_pair_target_tree_json_path=config["data"].get(
                "overfit_fixed_pair_target_tree_json_path"
            ),
            overfit_fixed_pair_target_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_target_tree_json_paths"
            ),
            overfit_fixed_pair_target_tree_json_dir=config["data"].get(
                "overfit_fixed_pair_target_tree_json_dir"
            ),
            overfit_fixed_pair_joint_bank_jsonl_path=config["data"].get(
                "overfit_fixed_pair_joint_bank_jsonl_path"
            ),
            overfit_split_multi_subset_events=config["data"].get(
                "overfit_split_multi_subset_events", False
            ),
            overfit_full_path_control_mode=config["data"].get(
                "overfit_full_path_control_mode", False
            ),
            overfit_full_path_control_birthset_boundary_labels=config["data"].get(
                "overfit_full_path_control_birthset_boundary_labels", False
            ),
            overfit_full_path_control_seed=config["data"].get(
                "overfit_full_path_control_seed", 42
            ),
            overfit_full_path_control_use_discrete_phase_time=config["data"].get(
                "overfit_full_path_control_use_discrete_phase_time", False
            ),
            overfit_full_path_control_terminal_label_mode=config["data"].get(
                "overfit_full_path_control_terminal_label_mode", "phase_start"
            ),
            overfit_full_path_control_terminal_include_ar_states=config["data"].get(
                "overfit_full_path_control_terminal_include_ar_states", False
            ),
            overfit_full_path_control_terminal_include_target_one_split_off=config["data"].get(
                "overfit_full_path_control_terminal_include_target_one_split_off",
                False,
            ),
            overfit_full_path_control_extra_velocity_samples_json_path=config["data"].get(
                "overfit_full_path_control_extra_velocity_samples_json_path"
            ),
            overfit_full_path_control_extra_velocity_samples_json_paths=config["data"].get(
                "overfit_full_path_control_extra_velocity_samples_json_paths"
            ),
            overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id=config["data"].get(
                "overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id"
            ),
            full_path_preparse_structural_trees=config["data"].get(
                "full_path_preparse_structural_trees", False
            ),
            overfit_oracle_prefix_start_prob=config["data"].get(
                "overfit_oracle_prefix_start_prob",
                config["data"].get(
                    "analysis_oracle_prefix_start_prob",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_start_prob", 0.0
                    ),
                ),
            ),
            overfit_oracle_prefix_max_fraction=config["data"].get(
                "overfit_oracle_prefix_max_fraction",
                config["data"].get(
                    "analysis_oracle_prefix_max_fraction",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_max_fraction", 0.5
                    ),
                ),
            ),
            overfit_fixed_pair_group_by_json_metadata=config["data"].get(
                "overfit_fixed_pair_group_by_json_metadata", False
            ),
            overfit_fixed_pair_reference_tree_from_target_bank=config["data"].get(
                "overfit_fixed_pair_reference_tree_from_target_bank", False
            ),
            overfit_virtual_epoch_size=config["data"].get(
                "overfit_virtual_epoch_size"
            ),
            overfit_fixed_pair_cache_virtual_index_selection=config["data"].get(
                "overfit_fixed_pair_cache_virtual_index_selection", False
            ),
            **topology_stream_dataset_kwargs,
            **trprobs_dataset_kwargs,
        )
        if self.same_dataset_batch:
            self.dataset_train.same_dataset_batch_size = int(self.batch_size)
            self.dataset_train.same_dataset_batch_seed = int(self.loader_seed or 0)
        self.dataset_val = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.test_ids, validation=True, sanity_check=config["data"].get("sanity_check", False), random_sanity_check=config["data"].get("random_sanity_check", False),
            overfit_velocity_zero=config["data"].get("overfit_velocity_zero", False),
            overfit_velocity_event_states=config["data"].get("overfit_velocity_event_states", False),
            overfit_velocity_orthant_start_states=config["data"].get(
                "overfit_velocity_orthant_start_states", False
            ),
            overfit_velocity_explicit_boundary_end_states=config["data"].get(
                "overfit_velocity_explicit_boundary_end_states", False
            ),
            overfit_velocity_fixed_timepoints=config["data"].get(
                "overfit_velocity_fixed_timepoints"
            ),
            overfit_velocity_explicit_boundary_label_scale_mode=config["data"].get(
                "overfit_velocity_explicit_boundary_label_scale_mode", "local"
            ),
            overfit_boundary_prefix_k=config["data"].get("overfit_boundary_prefix_k", -1),
            overfit_start_boundary_prefix_k=config["data"].get("overfit_start_boundary_prefix_k", -1),
            overfit_event_prefix_count=config["data"].get("overfit_event_prefix_count", -1),
            overfit_event_horizon=config["data"].get("overfit_event_horizon", 1),
            overfit_fixed_pair=config["data"].get("overfit_fixed_pair", False),
            overfit_fixed_pair_start_tree_newick=config["data"].get(
                "overfit_fixed_pair_start_tree_newick"
            ),
            overfit_fixed_pair_start_tree_json_path=config["data"].get(
                "overfit_fixed_pair_start_tree_json_path"
            ),
            overfit_fixed_pair_start_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_start_tree_json_paths"
            ),
            overfit_fixed_pair_start_tree_json_dir=config["data"].get(
                "overfit_fixed_pair_start_tree_json_dir"
            ),
            overfit_fixed_pair_target_tree_newick=config["data"].get(
                "overfit_fixed_pair_target_tree_newick"
            ),
            overfit_fixed_pair_target_tree_json_path=config["data"].get(
                "overfit_fixed_pair_target_tree_json_path"
            ),
            overfit_fixed_pair_target_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_target_tree_json_paths"
            ),
            overfit_fixed_pair_target_tree_json_dir=config["data"].get(
                "overfit_fixed_pair_target_tree_json_dir"
            ),
            overfit_fixed_pair_joint_bank_jsonl_path=config["data"].get(
                "overfit_fixed_pair_joint_bank_jsonl_path"
            ),
            overfit_split_multi_subset_events=config["data"].get(
                "overfit_split_multi_subset_events", False
            ),
            overfit_full_path_control_mode=config["data"].get(
                "overfit_full_path_control_mode", False
            ),
            overfit_full_path_control_birthset_boundary_labels=config["data"].get(
                "overfit_full_path_control_birthset_boundary_labels", False
            ),
            overfit_full_path_control_seed=config["data"].get(
                "overfit_full_path_control_seed", 42
            ),
            overfit_full_path_control_use_discrete_phase_time=config["data"].get(
                "overfit_full_path_control_use_discrete_phase_time", False
            ),
            overfit_full_path_control_terminal_label_mode=config["data"].get(
                "overfit_full_path_control_terminal_label_mode", "phase_start"
            ),
            overfit_full_path_control_terminal_include_ar_states=config["data"].get(
                "overfit_full_path_control_terminal_include_ar_states", False
            ),
            overfit_full_path_control_terminal_include_target_one_split_off=config["data"].get(
                "overfit_full_path_control_terminal_include_target_one_split_off",
                False,
            ),
            overfit_full_path_control_extra_velocity_samples_json_path=config["data"].get(
                "overfit_full_path_control_extra_velocity_samples_json_path"
            ),
            overfit_full_path_control_extra_velocity_samples_json_paths=config["data"].get(
                "overfit_full_path_control_extra_velocity_samples_json_paths"
            ),
            overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id=config["data"].get(
                "overfit_full_path_control_extra_velocity_samples_json_paths_by_dataset_id"
            ),
            full_path_preparse_structural_trees=config["data"].get(
                "full_path_preparse_structural_trees", False
            ),
            overfit_oracle_prefix_start_prob=config["data"].get(
                "overfit_oracle_prefix_start_prob",
                config["data"].get(
                    "analysis_oracle_prefix_start_prob",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_start_prob", 0.0
                    ),
                ),
            ),
            overfit_oracle_prefix_max_fraction=config["data"].get(
                "overfit_oracle_prefix_max_fraction",
                config["data"].get(
                    "analysis_oracle_prefix_max_fraction",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_max_fraction", 0.5
                    ),
                ),
            ),
            overfit_fixed_pair_group_by_json_metadata=config["data"].get(
                "overfit_fixed_pair_group_by_json_metadata", False
            ),
            overfit_fixed_pair_reference_tree_from_target_bank=config["data"].get(
                "overfit_fixed_pair_reference_tree_from_target_bank", False
            ),
            overfit_virtual_epoch_size=None,
            overfit_fixed_pair_cache_virtual_index_selection=False,
            **topology_stream_val_dataset_kwargs,
            **trprobs_dataset_kwargs,
        )
        self.sample_metrics_data_module = None
        self.sample_metrics_dataset_train = None
        self.sample_metrics_dataset_val = None
        sample_metrics_config_path = (
            config["data"].get("sample_metrics_config_path")
            or config.get("trainer", {}).get("sample_metrics_config_path")
        )
        if sample_metrics_config_path:
            with open(sample_metrics_config_path, "r") as handle:
                sample_metrics_config = yaml.safe_load(handle)
            sample_ids = _coerce_id_list(
                sample_metrics_config.get("data", {}).get(
                    "posterior_dataset_ids",
                    sample_metrics_config.get("data", {}).get("short_run_dataset_ids"),
                )
            )
            if not sample_ids:
                sample_ids = _coerce_id_list(
                    sample_metrics_config.get("data", {}).get(
                        "posterior_dataset_id",
                        sample_metrics_config.get("data", {}).get("short_run_dataset_id"),
                    )
                )
            if not sample_ids:
                sample_ids = get_possible_ids(sample_metrics_config["data"]["nexus_root"])
            ran = random.Random(42)
            ran.shuffle(sample_ids)
            if len(sample_ids) < 2:
                sample_train_ids = sample_ids
                sample_test_ids = sample_ids
            else:
                sample_train_ids = sample_ids[: int(0.8 * len(sample_ids))]
                sample_test_ids = sample_ids[int(0.8 * len(sample_ids)) :]
            self.sample_metrics_data_module = PhylaDataModule(
                sample_metrics_config,
                train_ids=sample_train_ids,
                test_ids=sample_test_ids,
            )
            self.sample_metrics_dataset_train = (
                self.sample_metrics_data_module.dataset_train
            )
            self.sample_metrics_dataset_val = self.sample_metrics_data_module.dataset_val
        self.tree_tokenizer = TreeFeatureTokenizer(
            config["model"]["num_node_types"],
            config["model"]["num_edge_types"],
            config["model"]["hidden_dim"],
            n_layers=config["model"].get("tokenizer_n_layers", 6),
            lap_dim=config["model"].get("tokenizer_lap_dim", 16),
            lap_dropout=config["model"].get("tokenizer_lap_dropout", 0.2),
            branch_length_mode=config["model"].get(
                "tokenizer_branch_length_mode",
                "linear",
            ),
            branch_length_num_buckets=config["model"].get(
                "tokenizer_branch_length_num_buckets",
                64,
            ),
            branch_length_log_min=config["model"].get(
                "tokenizer_branch_length_log_min",
                -8.0,
            ),
            branch_length_log_max=config["model"].get(
                "tokenizer_branch_length_log_max",
                1.0,
            ),
        )
        self._collate_structural_tree_cache: Dict[str, Any] = {}
        self._collate_bhv_tree_cache: Dict[
            str,
            Tuple[Dict[int, float], int],
        ] = {}
        self.use_historical_collate = bool(
            config["data"].get("use_historical_collate", False)
        )
        self.msa_distance = True

    @property
    def chosen_tree(self):
        return self.dataset_train.chosen_tree

    @chosen_tree.setter
    def chosen_tree(self, value):
        self.dataset_train.chosen_tree = value

    @property
    def size_detector(self):
        return self.dataset_train.size_detector

    @property
    def name_to_seq(self):
        return self.dataset_train.name_to_seq

    def return_max_length(self, name_to_seq):
        return self.dataset_train.return_max_length(name_to_seq)

    def __getitem__(self, *args, **kwargs):
        return self.dataset_train.__getitem__(*args, **kwargs)

    def train_dataloader(self) -> DataLoader:
        generator = None
        if self.loader_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.loader_seed))
        if self.same_dataset_batch:
            return DataLoader(
                self.dataset_train,
                batch_sampler=SameDatasetBatchSampler(
                    self.dataset_train,
                    self.batch_size,
                    num_batches=self.same_dataset_batches_per_epoch,
                    seed=self.loader_seed,
                    shuffle=True,
                ),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def _collate_structural_tree(self, tree_newick):
        key = str(tree_newick)
        cached = self._collate_structural_tree_cache.get(key)
        if cached is None:
            cached = self.tree_tokenizer._newick_to_structural(key)
            self._collate_structural_tree_cache[key] = cached
        return cached

    def _collate_tokenizer_raw_graph(self, tree_newick, structural_tree=None):
        if structural_tree is None:
            structural_tree = _worker_newick_parser(str(tree_newick))
        raw_graph = self.tree_tokenizer.compute_raw_graph_cache([structural_tree])[0]
        return {
            key: value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else value
            for key, value in raw_graph.items()
        }

    def _select_velocity_samples_for_joint_tokenizer(self, samples):
        samples = list(samples or [])
        if self.velocity_probe_direct_set_anchor_only:
            anchor_samples = [
                sample for sample in samples if sample.get("anchor_family")
            ]
            if anchor_samples:
                return anchor_samples
        return samples

    def _pack_tokenizer_raw_graph_batch(self, raw_graphs, task_types=None):
        raw_graphs = [raw_graph for raw_graph in raw_graphs if raw_graph is not None]
        if not raw_graphs:
            return None

        def _array(raw_graph, key, dtype=None):
            value = raw_graph[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if dtype is not None:
                value = value.astype(dtype, copy=False)
            return value

        batch_size = len(raw_graphs)
        task_types = (
            [0] * batch_size
            if task_types is None
            else [int(task_type) for task_type in task_types]
        )
        node_counts = np.asarray(
            [int(raw_graph["node_num"]) for raw_graph in raw_graphs],
            dtype=np.int64,
        )
        edge_counts = np.asarray(
            [int(raw_graph["edge_num"]) for raw_graph in raw_graphs],
            dtype=np.int64,
        )
        token_counts = node_counts + edge_counts
        max_tokens = int(token_counts.max()) if token_counts.size else 0
        token_offsets = np.zeros((batch_size + 1,), dtype=np.int64)
        np.cumsum(token_counts, out=token_offsets[1:])
        node_offsets = np.zeros((batch_size + 1,), dtype=np.int64)
        np.cumsum(node_counts, out=node_offsets[1:])
        edge_offsets = np.zeros((batch_size + 1,), dtype=np.int64)
        np.cumsum(edge_counts, out=edge_offsets[1:])
        total_tokens = int(token_offsets[-1])
        total_nodes = int(node_offsets[-1])
        total_edges = int(edge_offsets[-1])

        node_data = (
            np.concatenate(
                [_array(raw_graph, "node_data", np.int64) for raw_graph in raw_graphs],
                axis=0,
            )
            if total_nodes > 0
            else np.zeros((0,), dtype=np.int64)
        )
        sin_embed_node = (
            np.concatenate(
                [
                    _array(raw_graph, "sin_embed_node", np.float32)
                    for raw_graph in raw_graphs
                ],
                axis=0,
            )
            if total_nodes > 0
            else np.zeros((0, 0), dtype=np.float32)
        )
        lap_pe = (
            np.concatenate(
                [_array(raw_graph, "lap_pe", np.float32) for raw_graph in raw_graphs],
                axis=0,
            )
            if total_nodes > 0
            else np.zeros((0, 0), dtype=np.float32)
        )
        edge_data = (
            np.concatenate(
                [_array(raw_graph, "edge_data", np.int64) for raw_graph in raw_graphs],
                axis=0,
            )
            if total_edges > 0
            else np.zeros((0,), dtype=np.int64)
        )
        branch_lengths = (
            np.concatenate(
                [
                    _array(raw_graph, "branch_lengths", np.float32)
                    for raw_graph in raw_graphs
                ],
                axis=0,
            )
            if total_edges > 0
            else np.zeros((0,), dtype=np.float32)
        )
        sin_embed_edge = (
            np.concatenate(
                [
                    _array(raw_graph, "sin_embed_edge", np.float32)
                    for raw_graph in raw_graphs
                ],
                axis=0,
            )
            if total_edges > 0
            else np.zeros((0, 0), dtype=np.float32)
        )

        padding_mask = np.ones((batch_size, max_tokens), dtype=np.bool_)
        padded_indices = np.zeros((batch_size, max_tokens, 2), dtype=np.int64)
        padded_leaf_masks = np.zeros((batch_size, max_tokens), dtype=np.bool_)
        padded_edge_masks = np.zeros((batch_size, max_tokens), dtype=np.bool_)
        flat_lap_indices = np.zeros((total_tokens, 2), dtype=np.int64)
        flat_type_ids = np.zeros((total_tokens,), dtype=np.int64)
        flat_batch_indices = np.zeros((total_tokens,), dtype=np.int64)
        flat_token_positions = np.zeros((total_tokens,), dtype=np.int64)
        node_token_positions = np.zeros((total_nodes,), dtype=np.int64)
        edge_token_positions = np.zeros((total_edges,), dtype=np.int64)
        leaf_indices = []
        edge_split_masks = []

        for batch_idx, raw_graph in enumerate(raw_graphs):
            node_count = int(node_counts[batch_idx])
            edge_count = int(edge_counts[batch_idx])
            token_count = int(token_counts[batch_idx])
            token_start = int(token_offsets[batch_idx])
            token_end = int(token_offsets[batch_idx + 1])
            node_start = int(node_offsets[batch_idx])
            node_end = int(node_offsets[batch_idx + 1])
            edge_start = int(edge_offsets[batch_idx])
            edge_end = int(edge_offsets[batch_idx + 1])

            local_indices = _array(raw_graph, "full_padded_index", np.int64)
            leaf_mask = _array(raw_graph, "leaf_mask", np.bool_)
            lap_offset = node_start
            padding_mask[batch_idx, :token_count] = False
            padded_indices[batch_idx, :token_count] = local_indices
            padded_leaf_masks[batch_idx, :token_count] = leaf_mask
            padded_edge_masks[batch_idx, node_count:token_count] = True
            flat_lap_indices[token_start:token_end] = local_indices + lap_offset
            flat_type_ids[token_start + node_count:token_end] = 1
            flat_batch_indices[token_start:token_end] = batch_idx
            flat_token_positions[token_start:token_end] = np.arange(
                token_count,
                dtype=np.int64,
            )
            if node_count > 0:
                node_token_positions[node_start:node_end] = np.arange(
                    token_start,
                    token_start + node_count,
                    dtype=np.int64,
                )
            if edge_count > 0:
                edge_token_positions[edge_start:edge_end] = np.arange(
                    token_start + node_count,
                    token_end,
                    dtype=np.int64,
                )
            leaf_indices.append(_array(raw_graph, "leaf_idx", np.int64))
            edge_split_masks.append(
                [int(mask) for mask in raw_graph.get("edge_split_masks", [])]
            )

        return {
            "_tree_tokenizer_raw_graph_batch_cache": True,
            "batch_size": int(batch_size),
            "task_types": np.asarray(task_types, dtype=np.int64),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "token_counts": token_counts,
            "max_tokens": int(max_tokens),
            "node_data": node_data,
            "edge_data": edge_data,
            "branch_lengths": branch_lengths,
            "lap_pe": lap_pe,
            "sin_embed_node": sin_embed_node,
            "sin_embed_edge": sin_embed_edge,
            "padding_mask": padding_mask,
            "padded_indices": padded_indices,
            "padded_leaf_masks": padded_leaf_masks,
            "padded_edge_masks": padded_edge_masks,
            "flat_lap_indices": flat_lap_indices,
            "flat_type_ids": flat_type_ids,
            "flat_batch_indices": flat_batch_indices,
            "flat_token_positions": flat_token_positions,
            "node_token_positions": node_token_positions,
            "edge_token_positions": edge_token_positions,
            "leaf_idx": leaf_indices,
            "edge_split_masks": edge_split_masks,
        }

    def _attach_tokenizer_raw_graph_to_sample(
        self,
        sample,
        *,
        newick_key,
        structural_key,
        out_key,
    ):
        if sample.get(out_key) is not None:
            return
        newick = sample.get(newick_key)
        if newick in {None, ""}:
            return
        structural_tree = sample.get(structural_key)
        if structural_tree is None:
            structural_tree = self._collate_structural_tree(newick)
            sample[structural_key] = structural_tree
        sample[out_key] = self._collate_tokenizer_raw_graph(
            newick,
            structural_tree=structural_tree,
        )

    def _attach_tokenizer_raw_graph_to_full_path_samples(
        self,
        velocity_samples,
        autoregressive_samples,
        terminal_samples,
    ):
        for sample in velocity_samples:
            self._attach_tokenizer_raw_graph_to_sample(
                sample,
                newick_key="newick_tree",
                structural_key="newick_tree_structural",
                out_key="newick_tree_tokenizer_raw_graph",
            )
        if self.full_path_use_terminal_samples:
            for sample in terminal_samples:
                self._attach_tokenizer_raw_graph_to_sample(
                    sample,
                    newick_key="newick_tree",
                    structural_key="newick_tree_structural",
                    out_key="newick_tree_tokenizer_raw_graph",
                )
        for sample in autoregressive_samples:
            self._attach_tokenizer_raw_graph_to_sample(
                sample,
                newick_key="newick",
                structural_key="newick_structural",
                out_key="newick_tokenizer_raw_graph",
            )

    def _attach_birthset_precompute_to_full_path_samples(
        self,
        autoregressive_samples,
    ):
        if not (
            self.full_path_precompute_birthset_targets
            or self.full_path_precompute_birthset_candidate_info
        ):
            return
        for sample in autoregressive_samples:
            sample["_birthset_precomputed_candidate_info_enabled"] = bool(
                self.full_path_precompute_birthset_candidate_info
            )
            if sample.get("birthset_precomputed") is not None:
                continue
            precomputed = _birthset_precompute_sample(
                sample,
                use_small_polytomy_enumeration=self.birthset_use_small_polytomy_enumeration,
                use_static_pair_triple_candidates=(
                    self.full_path_birthset_static_pair_triple_candidates
                ),
                max_enum_components=self.birthset_max_enum_components,
                max_candidates_per_polytomy=self.birthset_max_candidates_per_polytomy,
                proposal_pair_target_mode=self.birthset_proposal_pair_target_mode,
                proposal_max_expansion_examples=(
                    self.birthset_proposal_max_expansion_examples
                ),
                proposal_max_order_seed_pairs=(
                    self.birthset_proposal_max_order_seed_pairs
                ),
                proposal_train_topk=self.birthset_proposal_train_topk,
            )
            if precomputed is not None:
                sample["birthset_precomputed"] = precomputed

    def _collate_bhv_tree_entry(self, tree_newick):
        key = str(tree_newick)
        cached = self._collate_bhv_tree_cache.get(key)
        if cached is not None:
            return cached

        tree_obj = Tree(key)
        split_masks, split_lengths = BHVEncoder().return_BHV_encoding(tree_obj)
        length_map = {
            int(mask): float(length)
            for mask, length in zip(split_masks, split_lengths)
            if length is not None and float(length) > 1e-8
        }
        biological_bits = max(tree_obj.n_leaves - 1, 0)
        cached = (length_map, biological_bits)
        self._collate_bhv_tree_cache[key] = cached
        return cached

    def collate_fn(self, batch, preset_subtree_num=None):
        """Custom collate function if needed."""
        full_path_velocity_samples = []
        full_path_autoregressive_samples = []
        full_path_terminal_samples = []
        full_path_joint_tokenizer_raw_graph_batch = None
        full_path_control_mode = False
        for item in batch:
            if item is None:
                continue
            if item.get("_full_path_control_mode", False):
                full_path_control_mode = True
                full_path_velocity_samples.extend(
                    item.get("full_path_velocity_samples") or []
                )
                full_path_autoregressive_samples.extend(
                    item.get("full_path_autoregressive_samples") or []
                )
                if self.full_path_use_terminal_samples:
                    full_path_terminal_samples.extend(
                        item.get("full_path_terminal_samples") or []
                    )
        if (
            full_path_control_mode
            and self.full_path_precompute_tokenizer_raw_graphs
        ):
            self._attach_birthset_precompute_to_full_path_samples(
                full_path_autoregressive_samples
            )
            self._attach_tokenizer_raw_graph_to_full_path_samples(
                full_path_velocity_samples,
                full_path_autoregressive_samples,
                full_path_terminal_samples,
            )
            if self.full_path_joint_tokenizer_batch:
                selected_velocity_samples = (
                    self._select_velocity_samples_for_joint_tokenizer(
                        full_path_velocity_samples
                    )
                )
                joint_raw_graphs = [
                    sample.get("newick_tree_tokenizer_raw_graph")
                    for sample in selected_velocity_samples
                ] + [
                    sample.get("newick_tokenizer_raw_graph")
                    for sample in full_path_autoregressive_samples
                ]
                if joint_raw_graphs and all(
                    raw_graph is not None for raw_graph in joint_raw_graphs
                ):
                    full_path_joint_tokenizer_raw_graph_batch = (
                        self._pack_tokenizer_raw_graph_batch(
                            joint_raw_graphs,
                            task_types=(
                                [0] * len(selected_velocity_samples)
                                + [1] * len(full_path_autoregressive_samples)
                            ),
                        )
                    )
                    if full_path_joint_tokenizer_raw_graph_batch is not None:
                        full_path_joint_tokenizer_raw_graph_batch[
                            "velocity_count"
                        ] = int(len(selected_velocity_samples))
                        full_path_joint_tokenizer_raw_graph_batch[
                            "autoregressive_count"
                        ] = int(len(full_path_autoregressive_samples))
        elif full_path_control_mode:
            self._attach_birthset_precompute_to_full_path_samples(
                full_path_autoregressive_samples
            )

        if self.use_historical_collate:
            flat_batch = []
            for item in batch:
                if item is None:
                    continue
                multi_step_samples = item.get("multi_step_samples")
                if multi_step_samples:
                    flat_batch.extend(multi_step_samples)
                else:
                    flat_batch.append(item)
            batch = flat_batch

            if "posterior_trees" in batch[0]:
                ids = [item["id"] for item in batch]
                posterior_trees = [item["posterior_trees"] for item in batch]
                mappings = [item["num_to_name"] for item in batch]
                return {
                    "ids": ids,
                    "posterior_trees": posterior_trees,
                    "phyla_embeddings": None,
                    "mappings": mappings,
                    "nexus_filepaths": [item["nexus_path"] for item in batch],
                    "tree_paths": [item["tree_paths"] for item in batch],
                }

            trees_to_tokenize = [item["newick_tree"] for item in batch]
            structural_trees = [
                self._collate_structural_tree(tree)
                for tree in trees_to_tokenize
            ]
            with torch.no_grad():
                tokenized_trees = _detach_tensors(
                    self.tree_tokenizer(structural_trees)
                )
            num_leaves = [len(batch[i]["sequences"]) for i in range(len(batch))]
            autoregressive_trees_to_tokenize = [
                item["autoregressive_newick"] for item in batch
            ]
            autoregressive_structural_trees = [
                self._collate_structural_tree(tree)
                for tree in autoregressive_trees_to_tokenize
            ]
            with torch.no_grad():
                autoregressive_tokenized_trees = _detach_tensors(
                    self.tree_tokenizer(autoregressive_structural_trees)
                )
            mappings = [item["num_to_name"] for item in batch]
            ids = [item["id"] for item in batch]
            dataset_ids = [
                str(item.get("dataset_id", item["id"])).upper()
                for item in batch
            ]
            batched_autoregressive_time = torch.tensor(
                [item["autoregressive_newick_time"] for item in batch],
                dtype=torch.float32,
            )

            to_run = {
                "tokenized_trees": tokenized_trees,
                "tokenized_autoregressive_trees": autoregressive_tokenized_trees,
                "newick_autoregressive_trees": autoregressive_trees_to_tokenize,
                "nexus_filepaths": [item["nexus_path"] for item in batch],
                "tree_paths": [item["tree_paths"] for item in batch],
                "original_trees": [item["newick_tree"] for item in batch],
                "start_trees": [
                    item.get("start_tree", item["newick_tree"]) for item in batch
                ],
                "target_trees": [item["target_tree"] for item in batch],
                "batched_velocity": [item["velocity"] for item in batch],
                "velocity_next_boundary_trees": [
                    item.get("velocity_next_boundary_tree") for item in batch
                ],
                "batched_autoregressive_time": batched_autoregressive_time,
                "batched_autoregressive_labels": [
                    item["autoregressive_labels"] for item in batch
                ],
                "batched_autoregressive_stop_after_merge": torch.tensor(
                    [
                        1.0
                        if item.get("autoregressive_stop_after_merge", False)
                        else 0.0
                        for item in batch
                    ],
                    dtype=torch.float32,
                ),
                "batched_time": torch.tensor(
                    [item["timepoint"] for item in batch], dtype=torch.float32
                ),
                "phyla_embeddings": None,
                "num_leaves": num_leaves,
                "ids": ids,
                "dataset_ids": dataset_ids,
                "mappings": mappings,
                "selected_sequences": [
                    item.get("selected_sequences") for item in batch
                ],
                "selected_sequence_names": [
                    item.get("selected_sequence_names") for item in batch
                ],
                "bank_group_key": [
                    item.get("bank_group_key") for item in batch
                ],
            }
            if full_path_control_mode:
                to_run["full_path_velocity_samples"] = list(
                    full_path_velocity_samples
                )
                to_run["full_path_autoregressive_samples"] = list(
                    full_path_autoregressive_samples
                )
                to_run["full_path_terminal_samples"] = list(
                    full_path_terminal_samples
                )
                if full_path_joint_tokenizer_raw_graph_batch is not None:
                    to_run["full_path_joint_tokenizer_raw_graph_batch"] = (
                        full_path_joint_tokenizer_raw_graph_batch
                    )
                to_run["_birthset_precomputed_candidate_info_enabled"] = bool(
                    self.full_path_precompute_birthset_candidate_info
                )
                to_run["_full_path_control_mode"] = True
            return to_run

        flat_batch = []
        for item in batch:
            if item is None:
                continue
            multi_step_samples = item.get("multi_step_samples")
            if multi_step_samples:
                flat_batch.extend(multi_step_samples)
            else:
                flat_batch.append(item)
        batch = flat_batch

        if "posterior_trees" in batch[0]:
            ids = [item["id"] for item in batch]
            posterior_trees = [item["posterior_trees"] for item in batch]
            mappings = [item["num_to_name"] for item in batch]
            phyla_embeddings = None

            return {
                "ids": ids,
                "posterior_trees": posterior_trees,
                "phyla_embeddings": phyla_embeddings,
                "mappings": mappings,
                "nexus_filepaths": [item["nexus_path"] for item in batch],
                "tree_paths": [item["tree_paths"] for item in batch],
            }

        # preset_subtree_num is accepted but currently unused in logic below
        # Just ensuring signature matches call site

        trees_to_tokenize = [item["newick_tree"] for item in batch]
        structural_trees = [
            self._collate_structural_tree(tree)
            for tree in trees_to_tokenize
        ]
        # Tokenizer runs in worker if num_workers > 0, so must disable gradients
        # to avoid pickling errors (grad_fn cannot be pickled).
        
        try:
            with torch.no_grad():
                tokenized_trees = _detach_tensors(self.tree_tokenizer(structural_trees))
        except Exception as e:
            print(f"Error in tree tokenization: {e}")
            return None 

        def _aligned_true_edge_lengths(tree_newick, token_masks):
            true_length_map, biological_bits = self._collate_bhv_tree_entry(
                tree_newick
            )
            full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
            aligned_lengths = []
            for raw_mask in token_masks:
                raw_mask = int(raw_mask)
                if raw_mask == 0:
                    aligned_lengths.append(0.0)
                    continue

                edge_length = 0.0
                if raw_mask in true_length_map:
                    edge_length = float(true_length_map[raw_mask])
                elif full_model_mask and (full_model_mask ^ raw_mask) in true_length_map:
                    edge_length = float(true_length_map[int(full_model_mask ^ raw_mask)])
                aligned_lengths.append(edge_length)
            return torch.as_tensor(aligned_lengths, dtype=torch.float32)

        tokenized_tree_edge_lengths = [
            _aligned_true_edge_lengths(tree_newick, tokenized_trees[-1][idx])
            for idx, tree_newick in enumerate(trees_to_tokenize)
        ]

        velocity_next_boundary_active_masks = []
        for batch_idx, item in enumerate(batch):
            next_boundary_tree = item.get("velocity_next_boundary_tree")
            if not next_boundary_tree:
                velocity_next_boundary_active_masks.append(None)
                continue

            _, biological_bits = self._collate_bhv_tree_entry(item["newick_tree"])
            boundary_length_map, _ = self._collate_bhv_tree_entry(next_boundary_tree)
            full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
            current_masks = [int(mask) for mask in tokenized_trees[-1][batch_idx]]
            active_masks = set()
            for raw_mask in current_masks:
                raw_mask = int(raw_mask)
                if raw_mask == 0:
                    continue
                if raw_mask in boundary_length_map or (
                    full_model_mask and (full_model_mask ^ raw_mask) in boundary_length_map
                ):
                    active_masks.add(raw_mask)
            velocity_next_boundary_active_masks.append(active_masks)

        num_leaves = [len(batch[i]["sequences"]) for i in range(len(batch))]

        autoregressive_trees_to_tokenize = [
            item["autoregressive_newick"] for item in batch
        ]
        autoregressive_structural_trees = [
            self._collate_structural_tree(tree)
            for tree in autoregressive_trees_to_tokenize
        ]

        try:
            with torch.no_grad():
                autoregressive_tokenized_trees = _detach_tensors(
                    self.tree_tokenizer(autoregressive_structural_trees)
                )
        except Exception as e:
            print(f"Error in autoregressive tree tokenization: {e}")
            return None
            
        mappings = [item['num_to_name'] for item in batch]
        ids = [item["id"] for item in batch]
        dataset_ids = [
            str(item.get("dataset_id", item["id"])).upper()
            for item in batch
        ]

        batched_autoregressive_time = torch.tensor(
            [item["autoregressive_newick_time"] for item in batch], dtype=torch.float32
        )

        to_run = {
            "tokenized_trees": tokenized_trees,
            "tokenized_autoregressive_trees": autoregressive_tokenized_trees,
            "newick_autoregressive_trees": autoregressive_trees_to_tokenize,
            "nexus_filepaths": [item["nexus_path"] for item in batch],
            "tree_paths": [item["tree_paths"] for item in batch],
            "original_trees": [item["newick_tree"] for item in batch],
            "start_trees": [
                item.get("start_tree", item["newick_tree"]) for item in batch
            ],
            "target_trees": [item["target_tree"] for item in batch],
            "batched_velocity": [item["velocity"] for item in batch],
            "tokenized_tree_edge_lengths": tokenized_tree_edge_lengths,
            "velocity_next_boundary_trees": [
                item.get("velocity_next_boundary_tree") for item in batch
            ],
            "velocity_next_boundary_active_masks": velocity_next_boundary_active_masks,
            "batched_autoregressive_time": batched_autoregressive_time,
            "batched_autoregressive_labels": [
                item["autoregressive_labels"] for item in batch
            ],
            "batched_autoregressive_stop_after_merge": torch.tensor(
                [
                    1.0 if item.get("autoregressive_stop_after_merge", False) else 0.0
                    for item in batch
                ],
                dtype=torch.float32,
            ),
            "batched_time": torch.tensor(
                [item["timepoint"] for item in batch], dtype=torch.float32
            ),
            # "phyla_embeddings": torch.tensor([item['phyla_embedding'] for item in batch], dtype=torch.float32),
            "phyla_embeddings": None,
            "num_leaves": num_leaves,
            "ids": ids,
            "dataset_ids": dataset_ids,
            "mappings": mappings,
            "sequence_ordering_maps": [item["seq_ordering_map"] for item in batch],
            "selected_sequences": [
                item.get("selected_sequences") for item in batch
            ],
            "selected_sequence_names": [
                item.get("selected_sequence_names") for item in batch
            ],
            "bank_group_key": [item.get("bank_group_key") for item in batch],
        }
        if full_path_control_mode:
            to_run["full_path_velocity_samples"] = list(full_path_velocity_samples)
            to_run["full_path_autoregressive_samples"] = list(
                full_path_autoregressive_samples
            )
            to_run["full_path_terminal_samples"] = list(
                full_path_terminal_samples
            )
            if full_path_joint_tokenizer_raw_graph_batch is not None:
                to_run["full_path_joint_tokenizer_raw_graph_batch"] = (
                    full_path_joint_tokenizer_raw_graph_batch
                )
            to_run["_birthset_precomputed_candidate_info_enabled"] = bool(
                self.full_path_precompute_birthset_candidate_info
            )
            to_run["_full_path_control_mode"] = True
        return to_run


def test():
    dm = TreeDataset(
        nexus_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/nexus/",
        mrbayes_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/runs/",
    )
    res_one = dm[0]
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    test()
