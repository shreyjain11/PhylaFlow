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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils.bhv_utils import (
    BHVEncoder,
    _split_multi_label_training_events,
    return_sampled_tree_orthant_velocity,
    return_sampled_tree_boundary_decisions,
    return_tree_boundary_merge_paths,
)
import random
from model.treeTokenizer import TreeFeatureTokenizer
from utils.random_tree import Tree
from ete3 import Tree as EteTree
from utils.utils import remove_bit


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


def _include_fixed_pair_json_path_for_role(path: Path, role: str) -> bool:
    name = path.name
    if name.startswith("._") or name == ".DS_Store":
        return False
    lowered = name.lower()
    if "anchor" in lowered:
        return False
    if role == "start":
        return "_target" not in lowered
    if role == "target":
        return "_start" not in lowered
    return True


class SizeDetector:
    def __init__(self, max_aa=None):
        self.max_aa = max_aa

    def update_max_aa(self, new_max_aa):
        self.max_aa = new_max_aa


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
    """Dataset for release Nexus/MrBayes data and DS posterior references.

    The final release path uses `.trprobs` posterior references plus fixed
    start/target path artifacts. The Nexus/MrBayes directory layout remains
    supported for lightweight fixtures and validation utilities.
    """

    def __init__(
        self,
        nexus_root: str,
        mrbayes_root: str,
        filter_ids: Optional[List[str]] = None,
        validation=False,
        overfit_fixed_pair: bool = False,
        overfit_fixed_pair_start_tree_newick: Optional[str] = None,
        overfit_fixed_pair_start_tree_json_path: Optional[str] = None,
        overfit_fixed_pair_start_tree_json_paths: Optional[List[str]] = None,
        overfit_fixed_pair_start_tree_json_dir: Optional[str] = None,
        overfit_fixed_pair_target_tree_newick: Optional[str] = None,
        overfit_fixed_pair_target_tree_json_path: Optional[str] = None,
        overfit_fixed_pair_target_tree_json_paths: Optional[List[str]] = None,
        overfit_fixed_pair_target_tree_json_dir: Optional[str] = None,
        overfit_full_path_control_seed: int = 42,
        overfit_virtual_epoch_size: Optional[int] = None,
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
        posterior_ids = _coerce_id_list(posterior_dataset_ids)
        if not posterior_ids:
            posterior_ids = _coerce_id_list(posterior_dataset_id)
        self.posterior_dataset_ids = posterior_ids
        self.use_random_sequence_distribution = bool(
            use_random_sequence_distribution or self.posterior_trprobs_root
        )
        self.random_distribution_sequence_length = max(
            1, int(random_distribution_sequence_length)
        )
        self.random_distribution_sequence_seed = int(random_distribution_sequence_seed)
        alphabet = str(random_distribution_alphabet or "ACGT").strip()
        self.random_distribution_alphabet = alphabet or "ACGT"
        self.trprobs_sample_count_per_file = max(0, int(trprobs_sample_count_per_file))
        self.overfit_fixed_pair = bool(overfit_fixed_pair)
        self.overfit_full_path_control_seed = int(overfit_full_path_control_seed)
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
                if not _include_fixed_pair_json_path_for_role(raw_path, "start"):
                    continue
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
                if not _include_fixed_pair_json_path_for_role(raw_path, "target"):
                    continue
                override_payload = json.loads(raw_path.read_text())
                override_tree = (
                    override_payload.get("target_tree")
                    or override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                    or override_payload.get("tree")
                )
                if override_tree:
                    override_target_tree_bank.append(dict(override_payload))
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
        self.overfit_split_multi_subset_events = False
        self.size_detector = SizeDetector()
        # Tracks the most recently sampled dataset id and leaf count.
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
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]],
        ] = {}
        self._frozen_full_path_control_selections: List[Dict[str, Any]] = []
        self._cached_posterior_trees_by_key: Dict[Tuple[Any, ...], List[str]] = {}
        self.random_tree = None

        # Build index immediately; optionally preload
        self.build_index()
        self.set_overfit_fixed_pair_start_tree_bank(override_start_tree_bank)
        self.set_overfit_fixed_pair_target_tree_bank(override_target_tree_bank)
        if (
            self.overfit_fixed_pair
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
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        start_items = list(self.overfit_fixed_pair_start_tree_bank_items or [])
        target_items = list(self.overfit_fixed_pair_target_tree_bank_items or [])
        if not start_items or not target_items:
            return None, None

        compatible_group_keys = sorted(
            set(self._overfit_fixed_pair_start_tree_groups.keys())
            & set(self._overfit_fixed_pair_target_tree_groups.keys())
        )
        if not compatible_group_keys:
            return None, None

        chosen_group_key = random.choice(compatible_group_keys)
        chosen_start = random.choice(
            self._overfit_fixed_pair_start_tree_groups[chosen_group_key]
        )
        chosen_target = random.choice(
            self._overfit_fixed_pair_target_tree_groups[chosen_group_key]
        )
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

    def _sample_overfit_fixed_pair_bank_selection(
        self,
    ) -> Optional[Dict[str, Any]]:
        chosen_start_item, chosen_target_item = (
            self._sample_matching_overfit_fixed_pair_bank_items()
        )
        if chosen_start_item is None or chosen_target_item is None:
            return None

        chosen_start_tree = str(chosen_start_item["tree"])
        chosen_target_tree = str(chosen_target_item["tree"])
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
                selection = self._sample_overfit_fixed_pair_bank_selection()
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
        return _remap_tree_leaf_names_to_match_reference(
            target_tree_newick,
            start_tree_newick,
        )


    @staticmethod
    def _clone_full_path_control_sample_groups(
        groups: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        return cloned_groups[0], cloned_groups[1]

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
        )

    def _build_full_path_control_samples(
        self,
        pair: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            return sample

        velocity_samples: List[Dict[str, Any]] = []
        autoregressive_samples: List[Dict[str, Any]] = []
        for path_index, path in enumerate(boundary_paths):
            source_tree = (
                start_tree
                if int(path_index) == 0
                else str(boundary_paths[int(path_index) - 1]["end_newick"])
            )
            path_start_time = float(path_index)
            velocity_newick, velocity = return_sampled_tree_orthant_velocity(
                source_tree,
                target_tree,
                0.0,
                legacy_training_semantics=False,
            )
            velocity_samples.append(
                _attach_pair_group(
                    {
                        "path_index": int(path_index),
                        "newick_tree": str(velocity_newick),
                        "target_tree": target_tree,
                        "velocity": {
                            int(k): float(v)
                            for k, v in velocity.items()
                        },
                        "velocity_next_boundary_tree": str(path["start_newick"]),
                        "timepoint": path_start_time,
                        "num_leaves": int(Tree(source_tree).n_leaves),
                    }
                )
            )
            boundary_time = float(path_index)
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
                    _attach_pair_group(
                        {
                            "path_index": int(path_index),
                            "newick": str(event["newick"]),
                            "target_tree": target_tree,
                            "labels": list(event["labels"]),
                            "stop_after_merge": bool(
                                event.get("stop_after_merge", False)
                            ),
                            "time": boundary_time,
                        }
                    )
                )
        result = (velocity_samples, autoregressive_samples)
        self._cached_full_path_control_samples_by_key[cache_key] = (
            self._clone_full_path_control_sample_groups(result)
        )
        return self._clone_full_path_control_sample_groups(result)

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
            return (
                self._random_distribution_sequences(str(meta["id"]), taxa_order),
                taxa_order,
            )
        return self.parse_nexus(meta["nexus_path"])

    def sample_random_tree_with_base(
        self,
        real_tree,
        subtree_size: Optional[int] = None,
    ) -> Tuple[str, str]:
        start_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
        return start_tree, start_tree

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
            return {i: name for i, name in enumerate(meta.get("taxa_order", []))}
        _, taxa_order = self.parse_nexus(meta["nexus_path"])
        num_to_name = {i: name for i, name in enumerate(taxa_order)}
        return num_to_name

    def __getitem__(
        self, index: int, preset_subtree_size: Optional[int] = None
    ) -> Dict[str, Any]:  # Required for torch Dataset
        requested_index = int(index)
        if self.overfit_virtual_epoch_size is not None and len(self._index) > 0:
            index = requested_index % len(self._index)
        meta = self._index[index]
        if self.validation:
            return {
                "id": meta["id"],
                "posterior_trees": self.return_posterior_trees(index),
                "nexus_path": meta["nexus_path"],
                "tree_paths": meta["tree_paths"],
                "num_to_name": self.return_nexus_number_to_name(index),
            }

        seqs, taxa_order = self._sequences_for_meta(meta)

        # Update name_to_seq cache (dumb update for now)
        self.name_to_seq = seqs

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
        real_tree_newick = random.sample(trees, 1)[0]

        t = EteTree(real_tree_newick, format=1)
        leaves = t.get_leaves()

        if preset_subtree_size is not None and len(leaves) > preset_subtree_size:
            kept_leaves = random.sample(leaves, preset_subtree_size)
            t.prune(kept_leaves, preserve_branch_length=True)
            # real_tree_newick = t.write(format=1) # Don't write yet, wait for re-indexing
            # Update leaves for size tracking
            leaves = t.get_leaves()

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
                cache_key = (random_tree, target_tree_newick)
                cached_pair = self._cached_overfit_bank_pairs_by_key.get(cache_key)
                if cached_pair is not None:
                    return dict(cached_pair)

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
                split_multi_label_events=False,
                legacy_training_semantics=False,
            )
            allow_velocity_only_pair = bool(boundary_paths) and not final_labels

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
                    split_multi_label_events=False,
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

            if pair is None and len(self.overfit_fixed_pair_target_tree_newick_bank) > 1:
                selection_cache_key = int(requested_index)
                cached_selection = (
                    self._cached_overfit_bank_selection_by_virtual_index.get(
                        selection_cache_key
                    )
                )
                selection = dict(cached_selection) if cached_selection is not None else None

                if selection is None:
                    selection = self._sample_overfit_fixed_pair_bank_selection()
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
            }
            if "bank_group_key" in pair:
                sample["bank_group_key"] = pair["bank_group_key"]
            (
                sample["full_path_velocity_samples"],
                sample["full_path_autoregressive_samples"],
            ) = self._build_full_path_control_samples(pair)
            return sample

        horizon = 1
        max_start_index = max(0, len(final_labels) - horizon)
        random_index = random.randint(0, max_start_index)

        def _build_step_sample(event_index: int) -> Dict[str, Any]:
            chosen_autoregressive_event = final_labels[event_index]
            autoregressive_time = (
                0.0
                if len(final_labels) <= 1
                else event_index / float(len(final_labels) - 1)
            )
            velocity_source_tree = random_tree
            velocity_next_boundary_tree = None
            timepoint = random.uniform(0, 1)
            newick, velocity = return_sampled_tree_orthant_velocity(
                velocity_source_tree,
                effective_target_tree,
                timepoint,
                legacy_training_semantics=False,
            )

            return {
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
            }

        step_samples = [
            _build_step_sample(event_index)
            for event_index in range(random_index, random_index + horizon)
        ]

        num_to_name = self.return_nexus_number_to_name(index)
        sample = dict(step_samples[0])
        sample["num_to_name"] = original_names_map
        if "bank_group_key" in pair:
            sample["bank_group_key"] = pair["bank_group_key"]
        sample["seq_ordering_map"] = seq_ordering_map
        (
            sample["full_path_velocity_samples"],
            sample["full_path_autoregressive_samples"],
        ) = self._build_full_path_control_samples(pair)

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

    def sample_overfit_fixed_pair_bank_pair(self) -> Optional[Dict[str, Any]]:
        if not self.overfit_fixed_pair:
            return None
        chosen_start_item, chosen_target_item = (
            self._sample_matching_overfit_fixed_pair_bank_items()
        )
        if chosen_start_item is None or chosen_target_item is None:
            return None

        chosen_start_tree = str(chosen_start_item["tree"])
        chosen_target_tree = str(chosen_target_item["tree"])
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
            split_multi_label_events=False,
            legacy_training_semantics=False,
        )
        return {
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
            if not path.name.startswith("._") and path.name != ".DS_Store"
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


class PhylaDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for managing TreeDataset splits.

    Responsibilities:
    - create the DS train/validation datasets used by the release configs
    - expose train_dataloader()/val_dataloader()
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
        self.pin_memory = config["data"].get("pin_memory", False)
        self.loader_seed = config["data"].get(
            "loader_seed",
            config.get("trainer", {}).get("seed"),
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
            self.train_ids = filtered_train_ids or list(configured_posterior_ids)
            self.test_ids = filtered_test_ids or list(configured_posterior_ids)
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
            posterior_trprobs_root = os.environ.get("PHYLAFLOW_SHORT_RUN_ROOT")
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

        self.dataset_train = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.train_ids,
            overfit_fixed_pair=True,
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
            overfit_full_path_control_seed=config["data"].get(
                "overfit_full_path_control_seed", 42
            ),
            overfit_virtual_epoch_size=config["data"].get(
                "overfit_virtual_epoch_size"
            ),
            **trprobs_dataset_kwargs,
        )
        self.dataset_val = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.test_ids, validation=True,
            overfit_fixed_pair=True,
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
            overfit_full_path_control_seed=config["data"].get(
                "overfit_full_path_control_seed", 42
            ),
            overfit_virtual_epoch_size=None,
            **trprobs_dataset_kwargs,
        )
        self.tree_tokenizer = TreeFeatureTokenizer(
            config["model"]["num_node_types"],
            config["model"]["num_edge_types"],
            config["model"].get("hidden_dim", config["model"]["embed_dim"]),
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
        raise NotImplementedError(
            "Standalone test dataloaders are not part of the initial PhylaFlow "
            "release. Use val_dataloader() for the configured posterior "
            "reference split."
        )

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError(
            "Standalone prediction dataloaders are not part of the initial "
            "PhylaFlow release. Sampling is exposed through the training module "
            "and release configs."
        )

    def collate_fn(self, batch, preset_subtree_num=None):
        """Custom collate function if needed."""
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

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

        def _attach_full_path_sample_context(item, samples):
            enriched = []
            mapping = item.get("num_to_name")
            sequences = item.get("sequences")
            num_leaves = len(sequences) if isinstance(sequences, dict) else None
            for sample in samples or []:
                sample_with_context = dict(sample)
                if mapping is not None:
                    sample_with_context.setdefault("num_to_name", mapping)
                if num_leaves is not None:
                    sample_with_context.setdefault("num_leaves", int(num_leaves))
                if sequences is not None:
                    sample_with_context.setdefault("sequences", sequences)
                if "id" in item:
                    sample_with_context.setdefault("id", item["id"])
                if "nexus_path" in item:
                    sample_with_context.setdefault("nexus_path", item["nexus_path"])
                if "tree_paths" in item:
                    sample_with_context.setdefault("tree_paths", item["tree_paths"])
                enriched.append(sample_with_context)
            return enriched

        full_path_velocity_samples = []
        full_path_autoregressive_samples = []
        for item in batch:
            if (
                item.get("full_path_velocity_samples") is None
                or item.get("full_path_autoregressive_samples") is None
            ):
                raise RuntimeError(
                    "Production DS training requires full-path velocity and autoregressive samples."
                )
            full_path_velocity_samples.extend(
                _attach_full_path_sample_context(
                    item,
                    item["full_path_velocity_samples"],
                )
            )
            full_path_autoregressive_samples.extend(
                _attach_full_path_sample_context(
                    item,
                    item["full_path_autoregressive_samples"],
                )
            )

        # preset_subtree_num is accepted but currently unused in logic below
        # Just ensuring signature matches call site

        trees_to_tokenize = [item["newick_tree"] for item in batch]
        structural_trees = [
            self.tree_tokenizer._newick_to_structural(tree)
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
            tree_obj = Tree(tree_newick)
            split_masks, split_lengths = BHVEncoder().return_BHV_encoding(tree_obj)
            true_length_map = {
                int(mask): float(length)
                for mask, length in zip(split_masks, split_lengths)
                if length is not None and float(length) > 1e-8
            }
            biological_bits = max(tree_obj.n_leaves - 1, 0)
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

            current_tree_obj = Tree(item["newick_tree"])
            boundary_tree_obj = Tree(next_boundary_tree)
            boundary_masks, boundary_lengths = BHVEncoder().return_BHV_encoding(
                boundary_tree_obj
            )
            boundary_length_map = {
                int(mask): float(length)
                for mask, length in zip(boundary_masks, boundary_lengths)
                if length is not None and float(length) > 1e-8
            }
            biological_bits = max(current_tree_obj.n_leaves - 1, 0)
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

        try:
            with torch.no_grad():
                autoregressive_tokenized_trees = _detach_tensors(
                    self.tree_tokenizer(autoregressive_trees_to_tokenize)
                )
        except Exception as e:
            print(f"Error in autoregressive tree tokenization: {e}")
            return None
            
        mappings = [item['num_to_name'] for item in batch]
        ids = [item["id"] for item in batch]

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
            "mappings": mappings,
            "sequence_ordering_maps": [item["seq_ordering_map"] for item in batch],
            "bank_group_key": [item.get("bank_group_key") for item in batch],
            "full_path_velocity_samples": list(full_path_velocity_samples),
            "full_path_autoregressive_samples": list(
                full_path_autoregressive_samples
            ),
        }
        return to_run


if __name__ == "__main__":
    raise SystemExit(
        "data.dataset defines Dataset/DataModule classes and is not a standalone entry point."
    )
